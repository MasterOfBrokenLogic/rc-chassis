import asyncio
import re
import time
import random
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Constants ────────────────────────────────────────────────────────────────

SMC_HOMEPAGE   = "https://www.smcinsurance.com/"
SMC_API        = "https://www.smcinsurance.com/central/centralcall/CallReqWithHeader"
PROXIES_FILE   = Path(__file__).parent / "proxies.txt"

HEADERS = {
    "User-Agent":      "okhttp/4.9.2",
    "Accept-Encoding": "gzip, deflate",
}

# ── Proxy loader ─────────────────────────────────────────────────────────────
# Reads proxies.txt on startup (and on reload).
# Format per line — any of:
#   host:port:user:pass          ← owlproxy style
#   http://user:pass@host:port
#   host:port                    ← no-auth proxy

def _parse_proxy_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("http://") or line.startswith("https://"):
        return line
    parts = line.split(":")
    if len(parts) == 4:
        host, port, user, pw = parts
        return f"http://{user}:{pw}@{host}:{port}"
    if len(parts) == 2:
        return f"http://{line}"
    return None


def load_proxies() -> list[str]:
    if not PROXIES_FILE.exists():
        return []
    lines   = PROXIES_FILE.read_text().splitlines()
    proxies = [_parse_proxy_line(l) for l in lines]
    return [p for p in proxies if p]


# ── Proxy state ───────────────────────────────────────────────────────────────
# We keep a small pool of (proxy_url, client) pairs so connections are reused.
# Bad proxies get removed from the pool at runtime; reloaded on /reload-proxies.

class ProxyPool:
    def __init__(self):
        self._lock    = asyncio.Lock()
        self._proxies: list[str] = []
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._dead:    set[str]  = set()

    async def load(self, proxy_list: list[str]):
        async with self._lock:
            # Close removed clients
            removed = set(self._clients) - set(proxy_list)
            for p in removed:
                await self._clients[p].aclose()
                del self._clients[p]

            # Open new clients
            for p in proxy_list:
                if p not in self._clients:
                    self._clients[p] = httpx.AsyncClient(
                        proxy=p,
                        timeout=httpx.Timeout(12.0, connect=5.0),
                        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                        follow_redirects=True,
                        verify=False,
                    )

            self._proxies = [p for p in proxy_list if p in self._clients]
            self._dead.clear()
            print(f"[proxy] loaded {len(self._proxies)} proxies")

    async def close_all(self):
        for c in self._clients.values():
            await c.aclose()
        self._clients.clear()
        self._proxies.clear()

    def pick(self) -> tuple[str, httpx.AsyncClient] | tuple[None, None]:
        alive = [p for p in self._proxies if p not in self._dead]
        if not alive:
            return None, None
        p = random.choice(alive)
        return p, self._clients[p]

    def mark_dead(self, proxy: str):
        self._dead.add(proxy)
        print(f"[proxy] marked dead: {proxy}")

    @property
    def alive_count(self) -> int:
        return len([p for p in self._proxies if p not in self._dead])

    @property
    def total_count(self) -> int:
        return len(self._proxies)


pool = ProxyPool()

# ── Fallback direct client (no proxy) ────────────────────────────────────────
# Used only when proxies.txt is empty — e.g. running locally.

_direct_client: httpx.AsyncClient | None = None

# ── Cookie cache ──────────────────────────────────────────────────────────────
# Per-proxy cookie cache so we don't hit the homepage on every request.
# key = proxy_url (or "__direct__"), value = {mcbc, expires_at}

_cookie_cache: dict[str, dict] = {}
_cookie_lock  = asyncio.Lock()


async def get_mcbc(client: httpx.AsyncClient, key: str) -> str:
    async with _cookie_lock:
        now   = time.monotonic()
        entry = _cookie_cache.get(key)
        if entry and entry["mcbc"] and now < entry["expires_at"]:
            return entry["mcbc"]

        resp = await client.get(SMC_HOMEPAGE, headers=HEADERS)
        resp.raise_for_status()
        sc   = resp.headers.get("set-cookie", "")
        m    = re.search(r"MCBC=([^;]+)", sc)
        if not m:
            raise RuntimeError("MCBC cookie not found")

        _cookie_cache[key] = {"mcbc": m.group(1), "expires_at": now + 300}
        return _cookie_cache[key]["mcbc"]


def _invalidate_cookie(key: str):
    _cookie_cache.pop(key, None)


# ── SMC fetch ─────────────────────────────────────────────────────────────────

async def fetch_smc(vehicle_number: str) -> dict | None:
    proxy_url, client = pool.pick()

    # Fallback to direct client if no proxies configured
    if client is None:
        client    = _direct_client
        proxy_url = None
    cache_key = proxy_url or "__direct__"

    payload = {
        "url":   "GetVaahanDetailsByVehicleNo",
        "props": [vehicle_number, "", "0"],
    }

    async def _post(c: httpx.AsyncClient, mcbc: str):
        return await c.post(
            SMC_API,
            json=payload,
            headers={**HEADERS, "Content-Type": "application/json", "Cookie": f"MCBC={mcbc}"},
        )

    try:
        mcbc = await get_mcbc(client, cache_key)
        resp = await _post(client, mcbc)

        # Cookie expired early → refresh once
        if resp.status_code in (401, 403):
            _invalidate_cookie(cache_key)
            mcbc = await get_mcbc(client, cache_key)
            resp = await _post(client, mcbc)

        resp.raise_for_status()
        data = resp.json()

        if data.get("statusCode") == 200 and data.get("response"):
            return data["response"]
        return None

    except Exception as e:
        # Mark proxy dead so next request picks a different one
        if proxy_url:
            pool.mark_dead(proxy_url)
            _invalidate_cookie(cache_key)
        raise


# ── Vehicle number normalisation ──────────────────────────────────────────────

_VH_RE = re.compile(
    r'^([A-Z]{2})'
    r'(\d{1,2})'
    r'([A-Z]{1,3})'
    r'(\d{1,4})$'
)

def normalize_vehicle_number(raw: str) -> tuple[str, str]:
    clean = re.sub(r"[^A-Z0-9]", "", raw.upper())
    m = _VH_RE.match(clean)
    if not m:
        raise ValueError(f"Cannot parse vehicle number: {raw!r}")
    state, district, series, number = m.groups()
    canonical = f"{state}{district.zfill(2)}{series}{number.zfill(4)}"
    return canonical, clean


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _direct_client
    _direct_client = httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        follow_redirects=True,
    )

    proxies = load_proxies()
    if proxies:
        await pool.load(proxies)
    else:
        print("[proxy] proxies.txt empty or missing — using direct connection")

    yield

    await pool.close_all()
    await _direct_client.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    return {
        "status":       "ok",
        "proxies_alive": pool.alive_count,
        "proxies_total": pool.total_count,
    }


@app.get("/reload-proxies")
async def reload_proxies():
    """Hot-reload proxies.txt without restarting the server."""
    proxies = load_proxies()
    await pool.load(proxies)
    return {"loaded": len(proxies), "proxies": proxies}


@app.get("/fetch")
async def fetch_vehicle(
    vehicle: str = Query(default=None),
    vehicle_number: str = Query(default=None),
):
    raw = (vehicle or vehicle_number or "").strip()
    if not raw:
        return JSONResponse({"success": False, "error": "vehicle number is required"}, status_code=400)

    try:
        canonical, original = normalize_vehicle_number(raw)
    except ValueError:
        original  = re.sub(r"[^A-Z0-9]", "", raw.upper())
        canonical = original
        if not original or not (6 <= len(original) <= 12):
            return JSONResponse({"success": False, "error": "Invalid vehicle number format"}, status_code=400)

    start = time.perf_counter()

    try:
        vehicle_data = await fetch_smc(canonical)
    except httpx.TimeoutException:
        elapsed = round(time.perf_counter() - start, 3)
        return JSONResponse({"success": False, "error": "SMC request timed out", "fetch_time_seconds": elapsed}, status_code=504)
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 3)
        return JSONResponse({"success": False, "error": str(e), "fetch_time_seconds": elapsed}, status_code=500)

    elapsed = round(time.perf_counter() - start, 3)

    if not vehicle_data:
        return JSONResponse({
            "success": False, "error": "Vehicle not found",
            "queried_as": canonical, "fetch_time_seconds": elapsed,
        }, status_code=404)

    chassis       = re.sub(r"\s+", "", vehicle_data.get("chassis", ""))
    engine_number = vehicle_data.get("engine", "")

    return JSONResponse({
        "success":            True,
        "vehicle_number":     original,
        "queried_as":         canonical,
        "engine_number":      engine_number,
        "chassis_number":     chassis,
        "vehicle_data":       vehicle_data,
        "fetch_time_seconds": elapsed,
    })
