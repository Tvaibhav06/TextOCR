"""
python-backend/pipeline/ocr.py
────────────────────────────────────────────────────────────────────────────────
Phase 3 — Stage 5: Per-Region OCR Text Extraction

This module is ADDITIVE. It does NOT modify or import from core.py's internal
classes. It receives data from the existing pipeline as plain Python types.

Design rules:
  - OCR always runs on the ORIGINAL (pre-inpainting) image crops.
  - Detection already ran in Stage 1 (core.py). We receive polygon coords
    and run only PaddleOCR recognition (det=False).
  - Regions with confidence < 0.60 or text shorter than 3 chars are flagged
    skip=True. They appear as "erased_only" in the API response.
  - This class is imported by api/main.py's NEW /api/process-translate
    endpoint only. The existing /api/process endpoint is NOT touched.

Usage:
    from pipeline.ocr import RegionOCR, OcrResult
    ocr_engine = RegionOCR()
    results = ocr_engine.run(original_bgr, detections)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("textremover.ocr")

# ── Confidence gate constants ─────────────────────────────────────────────────
MIN_CONFIDENCE: float = 0.40   # Regions below this are skipped (noise threshold)
MIN_TEXT_LEN:   int   = 2      # Single-char noise skipped; 2+ chars are kept


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class OcrResult:
    """OCR result for a single detection region."""
    det_id:     int
    text:       str
    confidence: float
    skip:       bool    # True → keep erased only; do not translate or render
    x:          int     # bbox from Detection (pixel coords)
    y:          int
    w:          int
    h:          int


# ── Helper: crop + deskew a polygon region ────────────────────────────────────

def _crop_deskew(img: np.ndarray, polygon: np.ndarray,
                 padding: int = 2) -> Optional[np.ndarray]:
    """
    Crop a quadrilateral polygon region from an image and de-skew it to
    a horizontal rectangle using a perspective transform.

    Args:
        img:     Full BGR image (the ORIGINAL, pre-inpainting frame).
        polygon: Quadrilateral polygon as shape (N, 2) float or int array.
        padding: Extra pixels added around the bounding rect before crop.

    Returns:
        Cropped, de-skewed BGR image, or None if the crop is degenerate.
    """
    pts = polygon.astype(np.float32)

    # core.py stores polygons as (N, 1, 2) for cv2.fillPoly — squeeze to (N, 2)
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts.reshape(-1, 2)

    if pts.ndim != 2 or pts.shape[0] < 3:
        log.debug("_crop_deskew: degenerate polygon shape %s", pts.shape)
        return None

    # Ensure it's exactly 4 points for perspective transform
    if pts.shape[0] != 4:
        x, y, w, h = cv2.boundingRect(pts)
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

    # Compute target dimensions
    w1 = np.linalg.norm(pts[0] - pts[1])
    w2 = np.linalg.norm(pts[2] - pts[3])
    target_w = int(max(w1, w2))

    h1 = np.linalg.norm(pts[0] - pts[3])
    h2 = np.linalg.norm(pts[1] - pts[2])
    target_h = int(max(h1, h2))

    if target_w < 4 or target_h < 4:
        log.debug("_crop_deskew: near-zero-area polygon (w=%d, h=%d)", target_w, target_h)
        return None

    # Apply padding to target dimensions
    target_w += padding * 2
    target_h += padding * 2

    # Destination points with padding offset
    dst_pts = np.array([
        [padding, padding],
        [target_w - 1 - padding, padding],
        [target_w - 1 - padding, target_h - 1 - padding],
        [padding, target_h - 1 - padding]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst_pts)
    crop = cv2.warpPerspective(img, M, (target_w, target_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

    return crop


# ── Main OCR Class ────────────────────────────────────────────────────────────

class RegionOCR:
    """
    Per-region OCR runner for Stage 5.

    Uses PaddleOCR with det=False (recognition only). Text detection polygon
    coordinates are already provided by Stage 1 (core.py PaddleOCRDetector).

    Lazy-loaded: PaddleOCR is only imported and initialized when run() is
    called for the first time, keeping startup fast when translation is not used.
    """

    def __init__(self) -> None:
        self._recs: dict[str, object] = {}   # lazy-loaded on first use per lang

    def _ensure_loaded(self, paddle_lang: str) -> None:
        if paddle_lang in self._recs:
            return
        log.info("[Stage 5] Loading PaddleOCR recognizer (lang=%s, rec=True, det=False) …", paddle_lang)
        try:
            # Disable PIR / OneDNN to match the same flags as Stage 1
            import paddle
            paddle.set_flags({"FLAGS_enable_pir_api": False})
            from paddleocr import PaddleOCR
            self._recs[paddle_lang] = PaddleOCR(
                lang=paddle_lang,
                det=False,        # Detection already done in Stage 1
                rec=True,
                cls=True,
                use_angle_cls=True,
                show_log=False,
            )
            log.info("[Stage 5] PaddleOCR recognizer (%s) loaded.", paddle_lang)
        except Exception as exc:
            log.error("[Stage 5] Failed to load PaddleOCR (%s): %s", paddle_lang, exc)
            raise RuntimeError(f"Stage 5 OCR init failed: {exc}") from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self,
            original_bgr: np.ndarray,
            detections: list,
            source_lang_hint: str = "und") -> List[OcrResult]:
        """
        Run recognition on each detected region from the ORIGINAL image.

        Args:
            original_bgr: The full-resolution original image BEFORE inpainting.
                          This is critical — OCR must NOT run on erased pixels.
            detections:   List of Detection objects from Stage 1.
                          Expected attributes: .id, .polygon (np.ndarray N×2),
                          .bbox (.x .y .w .h).
            source_lang_hint: ISO language code hint for the source text.

        Returns:
            List of OcrResult, one per detection, in the same order.
        """
        # Map hint to paddle language
        # PaddleOCR supported: ch, en, french, german, korean, japan, arabic, devanagari, ta, te, ka...
        hint_map = {
            "hi": "devanagari", "mr": "devanagari", "ne": "devanagari", "sa": "devanagari",
            "ur": "arabic", "ar": "arabic", "fa": "arabic",
            "ta": "ta", "te": "te", "kn": "ka",
            "pa": "en", # Fallback to en/latin as paddleocr v4 default doesn't explicitly bundle pa
            "en": "en",
        }
        
        if source_lang_hint == "und":
            paddle_lang = "ch" # Default multilingual model (English + Chinese) if completely unknown
        elif source_lang_hint in hint_map:
            paddle_lang = hint_map[source_lang_hint]
        else:
            paddle_lang = "en" # Fallback to english for pure latin/unknown
            
        self._ensure_loaded(paddle_lang)
        results: List[OcrResult] = []

        for det in detections:
            polygon = np.array(det.polygon, dtype=np.float32)
            crop = _crop_deskew(original_bgr, polygon)

            text, conf = self._recognize_crop(crop, det.id, paddle_lang)
            skip = (conf < MIN_CONFIDENCE or len(text.strip()) < MIN_TEXT_LEN)

            results.append(OcrResult(
                det_id=det.id,
                text=text,
                confidence=round(conf, 4),
                skip=skip,
                x=det.bbox.x,
                y=det.bbox.y,
                w=det.bbox.w,
                h=det.bbox.h,
            ))

            if skip:
                log.debug(
                    "[Stage 5] Region %d skipped (conf=%.2f, len=%d): %r",
                    det.id, conf, len(text.strip()), text[:40]
                )
            else:
                log.debug(
                    "[Stage 5] Region %d: %r (conf=%.2f)",
                    det.id, text[:60], conf
                )

        log.info(
            "[Stage 5] OCR complete — %d regions, %d readable, %d skipped.",
            len(results),
            sum(1 for r in results if not r.skip),
            sum(1 for r in results if r.skip),
        )
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _recognize_crop(self,
                        crop: Optional[np.ndarray],
                        det_id: int,
                        paddle_lang: str) -> Tuple[str, float]:
        """Run PaddleOCR rec on a single cropped image. Returns (text, conf)."""
        if crop is None or crop.size == 0:
            log.debug("[Stage 5] Region %d: empty crop.", det_id)
            return "", 0.0

        # Minimum size guard — PaddleOCR crashes on tiny images
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            log.debug("[Stage 5] Region %d: crop too small (%s).",
                      det_id, crop.shape[:2])
            return "", 0.0

        try:
            result = self._recs[paddle_lang].ocr(crop, det=False, cls=True)
            # PaddleOCR rec-only format (det=False):
            #   result = [ [text, score], [text, score], ... ]
            #   result[0] = [text, score]
            #   result[0][0] = text string
            #   result[0][1] = confidence float
            # (Unlike det=True mode where result = [[[polygon, [text, score]], ...]])
            if not result or not result[0]:
                return "", 0.0
            first = result[0]
            # Guard: handle both [[text, score]] and [text, score] nesting variants
            if isinstance(first[0], (list, tuple)):
                text, conf = first[0][0], first[0][1]
            else:
                text, conf = first[0], first[1]
            return str(text).strip(), float(conf)
        except Exception as exc:
            log.warning("[Stage 5] OCR rec failed for region %d: %s", det_id, exc)
            return "", 0.0


# ── Singleton accessor ────────────────────────────────────────────────────────
# The endpoint creates one instance per worker process and reuses it.
# (Do not instantiate at module import time — keep startup fast.)
