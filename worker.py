import asyncio
import re
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Constants ────────────────────────────────────────────────────────────────

SMC_HOMEPAGE = "https://www.smcinsurance.com/"
SMC_API      = "https://www.smcinsurance.com/central/centralcall/CallReqWithHeader"

HEADERS = {
    "User-Agent": "okhttp/4.9.2",
    "Accept-Encoding": "gzip, deflate",
}

# ── Cookie cache ─────────────────────────────────────────────────────────────

_cookie_cache: dict = {"mcbc": None, "expires_at": 0.0}
_cookie_lock = asyncio.Lock()

_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        follow_redirects=True,
    )
    yield
    await _client.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Vehicle number normalisation ──────────────────────────────────────────────

# Indian vehicle number format: SS DD AAA NNNN
#   SS  = 2-letter state code       (alpha,  fixed width 2)
#   DD  = district code             (digits, pad to 2)
#   AAA = series                    (alpha,  1–3 letters, no padding needed)
#   NNNN= vehicle number            (digits, pad to 4)
#
# Examples:
#   KL7BX2273  → KL07BX2273
#   KL1B23     → KL01B0023
#   MH4F1234   → MH04F1234  (already correct, unchanged)

_VH_RE = re.compile(
    r'^([A-Z]{2})'       # state code  (exactly 2 alpha)
    r'(\d{1,2})'         # district    (1–2 digits)
    r'([A-Z]{1,3})'      # series      (1–3 alpha)
    r'(\d{1,4})$'        # number      (1–4 digits)
)

def normalize_vehicle_number(raw: str) -> tuple[str, str]:
    """
    Accepts a raw (already uppercased, stripped) vehicle number and returns
    (canonical, original_clean).

    canonical  — padded form sent to SMC  e.g. "KL07BX2273"
    original   — what the user typed      e.g. "KL7BX2273"

    Raises ValueError if the string doesn't look like an Indian vehicle number.
    """
    clean = re.sub(r"[^A-Z0-9]", "", raw.upper())

    m = _VH_RE.match(clean)
    if not m:
        raise ValueError(f"Cannot parse vehicle number: {raw!r}")

    state, district, series, number = m.groups()
    canonical = f"{state}{district.zfill(2)}{series}{number.zfill(4)}"
    return canonical, clean


# ── Cookie helper ─────────────────────────────────────────────────────────────

async def get_mcbc_cookie() -> str:
    async with _cookie_lock:
        now = time.monotonic()
        if _cookie_cache["mcbc"] and now < _cookie_cache["expires_at"]:
            return _cookie_cache["mcbc"]

        resp = await _client.get(SMC_HOMEPAGE, headers=HEADERS)
        resp.raise_for_status()

        sc = resp.headers.get("set-cookie", "")
        m = re.search(r"MCBC=([^;]+)", sc)
        if not m:
            raise RuntimeError("MCBC cookie not found in SMC homepage response")

        _cookie_cache["mcbc"]       = m.group(1)
        _cookie_cache["expires_at"] = now + 300
        return _cookie_cache["mcbc"]


# ── SMC fetch ─────────────────────────────────────────────────────────────────

async def fetch_smc(vehicle_number: str) -> dict | None:
    mcbc = await get_mcbc_cookie()

    payload = {
        "url":   "GetVaahanDetailsByVehicleNo",
        "props": [vehicle_number, "", "0"],
    }

    headers = {
        **HEADERS,
        "Content-Type": "application/json",
        "Cookie":        f"MCBC={mcbc}",
    }

    resp = await _client.post(SMC_API, json=payload, headers=headers)

    if resp.status_code in (401, 403):
        async with _cookie_lock:
            _cookie_cache["mcbc"]       = None
            _cookie_cache["expires_at"] = 0.0
        mcbc = await get_mcbc_cookie()
        headers["Cookie"] = f"MCBC={mcbc}"
        resp = await _client.post(SMC_API, json=payload, headers=headers)

    resp.raise_for_status()
    data = resp.json()

    if data.get("statusCode") == 200 and data.get("response"):
        return data["response"]
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.get("/fetch")
async def fetch_vehicle(
    vehicle: str = Query(default=None),
    vehicle_number: str = Query(default=None),
):
    raw = (vehicle or vehicle_number or "").strip()

    if not raw:
        return JSONResponse(
            {"success": False, "error": "vehicle number is required"},
            status_code=400,
        )

    # ── Normalise ──────────────────────────────────────────────────────────
    try:
        canonical, original = normalize_vehicle_number(raw)
    except ValueError:
        # Fallback: just strip non-alphanum and try as-is (handles edge cases
        # like temporary/special numbers that don't fit the standard pattern)
        original  = re.sub(r"[^A-Z0-9]", "", raw.upper())
        canonical = original
        if not original or not (6 <= len(original) <= 12):
            return JSONResponse(
                {"success": False, "error": "Invalid or unrecognised vehicle number format"},
                status_code=400,
            )

    start = time.perf_counter()

    try:
        vehicle_data = await fetch_smc(canonical)
    except httpx.TimeoutException:
        elapsed = round(time.perf_counter() - start, 3)
        return JSONResponse(
            {"success": False, "error": "SMC request timed out", "fetch_time_seconds": elapsed},
            status_code=504,
        )
    except Exception as e:
        elapsed = round(time.perf_counter() - start, 3)
        return JSONResponse(
            {"success": False, "error": str(e), "fetch_time_seconds": elapsed},
            status_code=500,
        )

    elapsed = round(time.perf_counter() - start, 3)

    if not vehicle_data:
        return JSONResponse(
            {
                "success":              False,
                "error":                "Vehicle not found",
                "queried_as":           canonical,
                "fetch_time_seconds":   elapsed,
            },
            status_code=404,
        )

    chassis       = re.sub(r"\s+", "", vehicle_data.get("chassis", ""))
    engine_number = vehicle_data.get("engine", "")

    return JSONResponse({
        "success":            True,
        "vehicle_number":     original,   # what user typed (cleaned)
        "queried_as":         canonical,  # what we actually sent to SMC
        "engine_number":      engine_number,
        "chassis_number":     chassis,
        "vehicle_data":       vehicle_data,
        "fetch_time_seconds": elapsed,
    })