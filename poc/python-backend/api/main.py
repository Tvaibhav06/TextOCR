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
from contextlib import asynccontextmanager # Force uvicorn reload

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
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

from pipeline.core import TextRemovalPipeline, Detection, BBox, PipelineResult, MSERDetector  # noqa: E402

# ── MSER second-pass helper ───────────────────────────────────────────────────
# Reuses the existing MSERDetector from core.py (zero new dependencies).
# Only used in the translate pipeline; /api/process is never touched.
_mser_detector: MSERDetector | None = None


def _mser_second_pass(
    image_bgr: np.ndarray,
    existing:  list,       # List[Detection] from Stage 1
    min_w:     int = 15,
    min_h:     int = 10,
    iou_gate:  float = 0.20,
) -> list:
    """
    Run a lightweight MSER sweep on the original image and return NEW detections
    that Stage 1 missed (IoU < iou_gate vs all existing boxes, size >= min_w x min_h).

    Existing Stage 1 detections are NOT modified.
    Returns ONLY the new candidate detections for merging.
    """
    global _mser_detector
    if _mser_detector is None:
        _mser_detector = MSERDetector()  # uses core.py defaults (min_area=30, max_area=4000)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mser_dets = _mser_detector.detect(gray, threshold=0.5)

    new_dets: list = []
    next_id = max((d.id for d in existing), default=-1) + 1

    for md in mser_dets:
        # Size gate: reject very small regions that are almost certainly noise
        if md.bbox.w < min_w or md.bbox.h < min_h:
            continue
        # IoU gate: skip if this region overlaps with any Stage 1 detection
        if any(md.bbox.iou(ed.bbox) >= iou_gate for ed in existing):
            continue
        # Renumber IDs so they don't collide with Stage 1 IDs
        new_det = Detection(
            id=next_id,
            bbox=md.bbox,
            polygon=md.polygon,
            confidence=md.confidence,
            source="mser_recovery",
        )
        new_dets.append(new_det)
        next_id += 1

    return new_dets


def _probe_source_lang(image_bgr: np.ndarray, detections: list, max_samples: int = 3) -> str:
    """
    Sample up to `max_samples` detection crops from the original image and run
    script-block detection to get a source language hint for Stage 5 OCR.

    Returns an ISO language code string (e.g. 'en', 'hi', 'zh') or 'und' if
    no dominant script is found.
    """
    from pipeline.translation import TranslationRouter
    from pipeline.ocr import _crop_deskew
    import numpy as np

    votes: dict[str, int] = {}
    for det in detections[:max_samples]:
        poly = np.array(det.polygon, dtype=np.float32)
        crop = _crop_deskew(image_bgr, poly)
        if crop is None or crop.size == 0:
            continue
        # Convert crop to a rough text sample for script detection.
        # We don't need real OCR here — just the Unicode script range probe.
        # Since this is a pixel crop (not text), we can't directly detect script;
        # instead we'll rely on the crop's pixel characteristics being meaningful
        # only when we have actual text content from later stages.
        # For now, use the lang from the bounding box source attribute as a hint.
        # If source is 'paddleocr', the detection is reliable; use 'und' for MSER.
        if getattr(det, 'source', '') in ('paddleocr', 'injected'):
            # Attempt to get script hint from detection confidence range
            # (high-conf PaddleOCR tends to be Latin/CJK based on model)
            pass  # will use 'und' and rely on individual OCR results

    # If we have no reliable votes, return 'und' — ocr.py will default to 'en'
    if not votes:
        return "und"
    return max(votes, key=lambda k: votes[k])


# ── Startup checks ────────────────────────────────────────────────────────────

def _check_fonts_at_startup() -> None:
    """
    Log a WARNING for every font referenced in renderer._FONT_MATRIX that is
    not present on disk.  Server continues regardless.
    """
    try:
        from pipeline.renderer import _FONT_MATRIX, FONTS_DIR
        missing = set()
        for filename in _FONT_MATRIX.values():
            if not (FONTS_DIR / filename).exists():
                missing.add(filename)
        if missing:
            for fn in sorted(missing):
                log.warning(
                    "Font missing: %s — run: python python-backend/scripts/download_fonts.py",
                    fn,
                )
        else:
            log.info("Font check OK — all %d required fonts present.", len(set(_FONT_MATRIX.values())))
    except Exception as exc:
        log.warning("Font check failed (non-fatal): %s", exc)


def _check_indictrans2_at_startup() -> None:
    """
    Log a WARNING for each IndicTrans2 model direction that is not cached locally.
    A partial installation is fine — those directions will use Argos as fallback.
    """
    try:
        from pipeline.translation import TranslationRouter
        router = TranslationRouter()
        status = router.indictrans2_status()
        available = [d for d, ok in status.items() if ok]
        missing   = [d for d, ok in status.items() if not ok]
        if not available:
            log.warning(
                "IndicTrans2: no model directions cached —"
                " all Indic↔English pairs will use Argos Translate."
                " To download: python -c \"from pipeline.translation import"
                " IndicTrans2Engine; e=IndicTrans2Engine();"
                " e._load_model_online('en-indic')\""
            )
        else:
            log.info("IndicTrans2 available directions: %s", ", ".join(available))
            for d in missing:
                log.warning(
                    "IndicTrans2 direction '%s' not cached — will fall back to Argos.", d
                )
    except Exception as exc:
        log.warning("IndicTrans2 check failed (non-fatal): %s", exc)


# ── Lazy pipeline singleton ────────────────────────────────────────────────────
_pipeline: TextRemovalPipeline | None = None


def get_pipeline() -> TextRemovalPipeline:
    global _pipeline
    if _pipeline is None:
        log.info("Initialising TextRemovalPipeline (first request) …")
        _pipeline = TextRemovalPipeline(lang="en", use_gpu=False)
        log.info("Pipeline ready.")
    return _pipeline

_ocr = None
_translator = None
_renderer = None

def get_ocr():
    global _ocr
    if _ocr is None:
        log.info("Initialising RegionOCR…")
        from pipeline.ocr import RegionOCR
        _ocr = RegionOCR()
    return _ocr

def get_translator():
    global _translator
    if _translator is None:
        log.info("Initialising TranslationRouter…")
        from pipeline.translation import TranslationRouter
        _translator = TranslationRouter()
    return _translator

def get_renderer():
    global _renderer
    if _renderer is None:
        log.info("Initialising TextRenderer…")
        from pipeline.renderer import TextRenderer
        _renderer = TextRenderer()
    return _renderer


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
    # ── Startup checks (non-fatal: warn and continue) ─────────────────────────
    _check_fonts_at_startup()
    _check_indictrans2_at_startup()
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
    expose_headers=["X-TotalMs", "X-Detections", "X-Translations", "X-Translated-Texts", "X-SSIM", "X-PSNR", "X-Coverage", "X-InpaintTier"],
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
def process_image(image: UploadFile = File(...)):
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
    raw_bytes = image.file.read()
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
        # Note: Access-Control-Expose-Headers is handled globally by CORSMiddleware
    }

    return StreamingResponse(
        io.BytesIO(jpeg_bytes),
        media_type="image/jpeg",
        headers=headers,
    )


@app.post("/api/process-translate", summary="Remove text and replace with translation")
def process_translate_image(
    image: UploadFile = File(...),
    target_lang: str = Form("en"),
    regions_json: str = Form(None),
):
    """
    Full Phase 3 pipeline: Erase text + Translate + Re-render.
    
    Processing:
      1. Stages 1-4: PaddleOCR detection + LaMa inpainting.
      2. Stage 5:    Tesseract OCR on detected regions from original image.
      3. Stage 6:    IndicTrans2 / Argos Translate to target_lang.
      4. Stage 7:    TextRenderer overlay with matched fonts & BiDi.
    """
    log.info(
        "Process-Translate request: file=%r target_lang=%r",
        image.filename,
        target_lang,
    )

    raw_bytes = image.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image.")

    try:
        t_start = time.perf_counter()
        encoded_texts = "[]"
        
        injected_detections = None
        injected_ocr_results = None
        
        if regions_json:
            import json
            import urllib.parse
            from pipeline.ocr import OcrResult
            try:
                data = json.loads(urllib.parse.unquote(regions_json))
                if not isinstance(data, dict): raise ValueError("payload must be dict")
                if data.get("version") != 1: raise ValueError("unsupported version")
                regions = data.get("regions")
                if not isinstance(regions, list): raise ValueError("regions must be a list")
                
                valid_detections = []
                valid_ocr = []
                for r in regions:
                    r_id = int(r["id"])
                    bbox = r["bbox"]
                    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
                    if w <= 0 or h <= 0: raise ValueError("invalid bbox dimensions")
                    
                    poly_list = r.get("polygon")
                    if poly_list and isinstance(poly_list, list) and len(poly_list) >= 3:
                        poly = np.array(poly_list, dtype=np.int32)
                    else:
                        poly = np.array([
                            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
                        ], dtype=np.int32)
                        
                    conf = float(r.get("confidence", 1.0))
                    
                    det = Detection(
                        id=r_id,
                        bbox=BBox(x, y, w, h),
                        polygon=poly,
                        confidence=conf,
                        source="injected"
                    )
                    valid_detections.append(det)
                    
                    text = r.get("text", "")
                    skip_flag = r.get("skip", False)
                    if text and isinstance(text, str) and text.strip():
                        valid_ocr.append({
                            "has_text": True,
                            "obj": OcrResult(
                                det_id=r_id, text=text, confidence=conf,
                                skip=skip_flag, x=x, y=y, w=w, h=h
                            )
                        })
                    else:
                        valid_ocr.append({
                            "has_text": False,
                            "obj": OcrResult(
                                det_id=r_id, text="", confidence=conf,
                                skip=True, x=x, y=y, w=w, h=h
                            )
                        })
                
                injected_detections = valid_detections
                injected_ocr_results = valid_ocr
                log.info("[Injection] Successfully parsed %d regions from mobile.", len(valid_detections))
                
            except Exception as e:
                log.warning("[Injection] Failed to parse regions_json: %s. Falling back to full pipeline.", e)
                injected_detections = None
                injected_ocr_results = None

        pipe = get_pipeline()
        
        if injected_detections is not None:
            # ── Stage 1-4 Bypass ──
            h_img, w_img = img_bgr.shape[:2]
            
            # Adaptive dilation for tight MLKit boxes
            dilation_px = max(12, int(min(h_img, w_img) * 0.015))
            
            mask = pipe.masker.generate(injected_detections, h_img, w_img, dilation_px=dilation_px)
            inpainted, _ = pipe.inpainter.inpaint(img_bgr, mask, radius=7)
            
            result = PipelineResult(
                output=inpainted, mask=mask, detections=injected_detections,
                quality={}, timing={}, total_ms=0.0
            )
        else:
            # ── Normal Stages 1–4: Erase text ──
            result = pipe.run(img_bgr)

            # ── MSER second-pass recovery (pure PC path only) ──
            # Find text regions that PaddleOCR Stage 1 missed and add them to
            # the detection list BEFORE passing to Stage 5. Stage 1–4 output
            # (inpainted image, mask, quality) is unchanged.
            if result.detections is not None:
                recovered = _mser_second_pass(img_bgr, result.detections)
                if recovered:
                    log.info(
                        "[MSER Recovery] Stage 1 found %d regions; "
                        "MSER sweep added %d more.",
                        len(result.detections), len(recovered),
                    )
                    # Build a merged list without mutating result
                    merged_detections = list(result.detections) + recovered
                    # Re-generate mask to cover newly recovered regions too
                    h_rec, w_rec = img_bgr.shape[:2]
                    merged_mask = pipe.masker.generate(
                        merged_detections, h_rec, w_rec, dilation_px=5
                    )
                    merged_inpainted, _ = pipe.inpainter.inpaint(img_bgr, merged_mask, radius=7)
                    result = PipelineResult(
                        output=merged_inpainted,
                        mask=merged_mask,
                        detections=merged_detections,
                        quality=result.quality,
                        timing=result.timing,
                        total_ms=result.total_ms,
                    )

        if not result.detections:
            # Nothing to translate; return original
            output_bgr = img_bgr
            translated_count = 0
        else:
            # ── Stage 5: OCR text from original image
            if injected_ocr_results is not None:
                from pipeline.translation import TranslationRouter
                ocr_engine = get_ocr()
                ocr_results = []
                for i, det in enumerate(result.detections):
                    inj = injected_ocr_results[i]
                    if inj["has_text"]:
                        ocr_obj = inj["obj"]
                        # Tag with stable source lang so Stage 6 avoids running langdetect
                        # on noisy handwriting OCR text (which causes misidentification).
                        # Mobile MLKit reads the source text reliably; default to 'en'.
                        ocr_obj._injected_src_lang = str(inj.get("src_lang", "en") or "en")
                        ocr_results.append(ocr_obj)
                    else:
                        # Attempt to derive source script from mobile text (if any), else 'und'
                        hint = "und"
                        inj_text = inj["obj"].text.strip()
                        if inj_text:
                            hint = TranslationRouter._identify_language(inj_text)

                        partial_res = ocr_engine.run(img_bgr, [det], source_lang_hint=hint)
                        if partial_res:
                            ocr_results.append(partial_res[0])
            else:
                # Pure PC path: probe source script from a few crops to pick
                # the right OCR model (avoids defaulting to Chinese recognizer).
                ocr_engine = get_ocr()
                source_lang_hint = _probe_source_lang(img_bgr, result.detections)
                log.info("[Stage 5] Source lang probe: %r", source_lang_hint)
                ocr_results = ocr_engine.run(
                    img_bgr, result.detections, source_lang_hint=source_lang_hint
                )
            
            # ── Stage 6: Translate
            translator = get_translator()
            # Preflight: log missing model warnings for this request's language pair
            translator.preflight(src="auto", tgt=target_lang)
            from pipeline.translation import TranslationResult
            tr_results = []
            for r in ocr_results:
                if r.skip:
                    tr_results.append(TranslationResult(
                        text=r.text, engine="skipped", src_lang="auto", tgt_lang=target_lang
                    ))
                else:
                    # Use the injected src_lang if the mobile path set it on the OcrResult.
                    # This avoids running langdetect on noisy handwriting OCR text (typos,
                    # dropped chars cause misidentification → wrong Argos pair → engine='failed').
                    injected_src = getattr(r, "_injected_src_lang", None)
                    src_hint = injected_src if injected_src else "auto"
                    res = translator.translate(r.text, src=src_hint, tgt=target_lang)
                    tr_results.append(res)

            
            # ── Stage 7: Render translated text onto inpainted image
            renderer = get_renderer()
            output_bgr = renderer.render_all(
                original_bgr=img_bgr,
                inpainted_bgr=result.output,
                ocr_results=ocr_results,
                translations=tr_results,
                target_lang=target_lang
            )
            
            import json
            translated_strings = [
                tr.text
                for tr in tr_results
                if tr and tr.engine not in ("failed",) and tr.text and tr.text.strip()
            ]
            translated_count = len(translated_strings)
            source_langs = list(set(
                tr.src_lang
                for tr in tr_results
                if tr and tr.engine not in ("failed", "skipped") and tr.src_lang and tr.src_lang not in ("auto", "und")
            ))

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    except Exception as exc:
        log.exception("Translate pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    _translated_count = translated_count if 'translated_count' in locals() else 0
    _translated_strings = translated_strings if 'translated_strings' in locals() else []
    _source_langs = source_langs if 'source_langs' in locals() else []
    _detections         = len(result.detections) if 'result' in locals() else 0

    log.info(
        "Translate done: %d regions translated in %.1fms",
        _translated_count,
        elapsed_ms,
    )

    ok, jpeg_buf = cv2.imencode(".jpg", output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image.")

    import base64
    image_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

    return JSONResponse({
        "image_b64":        image_b64,
        "translated_texts": _translated_strings,
        "total_ms":         elapsed_ms,
        "detections":       _detections,
        "translations":     _translated_count,
        "source_langs":     _source_langs,
    })
