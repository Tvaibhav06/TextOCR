"""
python-backend/api/main.py
────────────────────────────────────────────────────────────────────────────────
Phase 2: FastAPI wrapper around the existing TextRemovalPipeline.

Endpoints:
  GET  /api/health    → { "status": "ok", "version": "2.0" }
  POST /api/process   → multipart/form-data with 'image' field
                        Returns processed image as image/jpeg stream.

mDNS advertisement:
  Announces _textremover._tcp.local. so the mobile app can discover
  the server automatically on the same LAN using react-native-zeroconf.

Usage:
  uvicorn python-backend.api.main:app --host 0.0.0.0 --port 8000 --reload
  # or from within python-backend/api/:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import logging
import sys
import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("textremover.api")

# ── Pipeline import (resolve path regardless of CWD) ─────────────────────────
# Support running from repo root:  uvicorn python-backend.api.main:app
# or from python-backend/:         uvicorn api.main:app
_HERE = os.path.dirname(__file__)
_BACKEND_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from pipeline.core import TextRemovalPipeline  # noqa: E402

# ── Lazy pipeline singleton ────────────────────────────────────────────────────
_pipeline: TextRemovalPipeline | None = None


def get_pipeline() -> TextRemovalPipeline:
    global _pipeline
    if _pipeline is None:
        log.info("Initialising TextRemovalPipeline (first request) …")
        _pipeline = TextRemovalPipeline(lang="en", use_gpu=False)
        log.info("Pipeline ready.")
    return _pipeline


# ── mDNS advertisement (optional — graceful if zeroconf not installed) ────────

def _advertise_mdns(port: int = 8000) -> None:
    """
    Announce _textremover._tcp.local. so mobile clients can auto-discover.
    Requires: pip install zeroconf
    Skipped gracefully if zeroconf is not installed.
    """
    try:
        import socket
        from zeroconf import ServiceInfo, Zeroconf

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        ip_bytes = socket.inet_aton(local_ip)

        info = ServiceInfo(
            "_textremover._tcp.local.",
            f"TextRemover Server._textremover._tcp.local.",
            addresses=[ip_bytes],
            port=port,
            properties={b"version": b"2"},
            server=f"{hostname}.local.",
        )
        zc = Zeroconf()
        zc.register_service(info)
        log.info(
            "mDNS: advertised _textremover._tcp.local. at %s:%d", local_ip, port
        )
        # Keep a reference so GC doesn't destroy it
        app.state.zeroconf = zc
        app.state.zeroconf_info = info
    except ImportError:
        log.warning(
            "zeroconf package not installed — mDNS discovery disabled. "
            "Run: pip install zeroconf"
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("mDNS advertisement failed: %s", exc)


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("═══════════════════════════════════════════")
    log.info("  TextRemover API  —  Phase 2 Backend")
    log.info("═══════════════════════════════════════════")
    _advertise_mdns(port=8000)
    yield
    # Shutdown: unregister mDNS if active
    zc = getattr(app.state, "zeroconf", None)
    if zc is not None:
        zc.unregister_service(app.state.zeroconf_info)
        zc.close()
        log.info("mDNS: service unregistered.")
    log.info("TextRemover API shut down cleanly.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TextRemover API",
    version="2.0.0",
    description=(
        "Phase 2 hybrid-fallback backend. "
        "Accepts raw images from the mobile app and returns LaMa-inpainted results."
    ),
    lifespan=lifespan,
)

# CORS — allow all origins during development (mobile + emulator both hit this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t_start = time.perf_counter()
    log.info("→ %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
        elapsed = round((time.perf_counter() - t_start) * 1000, 1)
        log.info(
            "← %s %s  [%d]  %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response
    except Exception as exc:
        elapsed = round((time.perf_counter() - t_start) * 1000, 1)
        log.error(
            "✗ %s %s  ERROR  %.1fms  —  %s",
            request.method,
            request.url.path,
            elapsed,
            exc,
        )
        raise


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health", summary="Health check")
async def health():
    """
    Returns 200 when the server is running.
    The mobile app polls this to validate a discovered URL before caching it.
    """
    return JSONResponse({"status": "ok", "version": "2.0"})


@app.post("/api/process", summary="Process image — remove text via LaMa pipeline")
async def process_image(image: UploadFile = File(...)):
    """
    Accepts a raw image (JPEG/PNG/WEBP) as multipart/form-data.

    Processing:
      Stage 1 — PaddleOCR DBNet++ detection
      Stage 2 — Mask generation (fillPoly + dilation)
      Stage 3 — LaMa inpainting (Telea fallback)
      Stage 4 — SSIM/PSNR quality assessment

    Returns:
      • HTTP 200 + image/jpeg body — the inpainted result image
      • X-SSIM, X-PSNR, X-Coverage, X-InpaintTier, X-TotalMs response headers
        (so the mobile client can display quality metadata without parsing the body)

    On failure:
      • HTTP 400 if the file cannot be decoded as an image
      • HTTP 500 for pipeline errors
    """
    log.info(
        "Processing request: filename=%r  content_type=%r",
        image.filename,
        image.content_type,
    )

    # ── Read and decode ──────────────────────────────────────────────────────
    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        log.error("cv2.imdecode failed — not a valid image (%d bytes)", len(raw_bytes))
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Ensure the file is a valid JPEG/PNG.",
        )

    h, w = img_bgr.shape[:2]
    log.info("Image decoded: %d×%d px, %d bytes", w, h, len(raw_bytes))

    # ── Run pipeline ─────────────────────────────────────────────────────────
    try:
        pipe = get_pipeline()
        t_start = time.perf_counter()
        result = pipe.run(img_bgr)
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)
    except Exception as exc:
        log.exception("Pipeline error for %r: %s", image.filename, exc)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    log.info(
        "Pipeline done: SSIM=%.3f  coverage=%.1f%%  tier=%s  total=%.1fms",
        result.quality.get("ssim", 0),
        result.mask.coverage * 100,
        result.timing.get("inpaint_tier", "unknown"),
        elapsed_ms,
    )

    # ── Encode result as JPEG ────────────────────────────────────────────────
    ok, jpeg_buf = cv2.imencode(
        ".jpg", result.output, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not ok:
        log.error("cv2.imencode failed")
        raise HTTPException(status_code=500, detail="Failed to encode result image.")

    jpeg_bytes = jpeg_buf.tobytes()

    # ── Build response with quality metadata headers ──────────────────────────
    headers = {
        "X-SSIM":          str(round(result.quality.get("ssim", 0), 4)),
        "X-PSNR":          str(result.quality.get("psnr", 0)),
        "X-Coverage":      str(round(result.mask.coverage, 4)),
        "X-InpaintTier":   result.timing.get("inpaint_tier", "unknown"),
        "X-TotalMs":       str(elapsed_ms),
        "X-Detections":    str(len(result.detections)),
        # Allow mobile to read these headers cross-origin
        "Access-Control-Expose-Headers": (
            "X-SSIM, X-PSNR, X-Coverage, X-InpaintTier, X-TotalMs, X-Detections"
        ),
    }

    return StreamingResponse(
        io.BytesIO(jpeg_bytes),
        media_type="image/jpeg",
        headers=headers,
    )
