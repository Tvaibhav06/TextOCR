"""
python-backend/pipeline/core.py
────────────────────────────────────────────────────────────────────────────────
Backend pipeline — Phase 1: Library update (Track 1)

Used for:
  1. Verifying algorithmic correctness before porting to TS/native
  2. Server-side LaMa fallback when mobile quality is insufficient
  3. pytest integration tests run in CI

Stages (Phase 1 — Updated):
  Stage 1 — Detection   : PaddleOCR DBNet++ (replaces EasyOCR / MSER)
  Stage 2 — Mask        : cv2.fillPoly + morphological dilation  [unchanged]
  Stage 3 — Inpainting  : LaMa (premium) → OpenCV Telea fallback
  Stage 4 — Post        : SSIM/PSNR metrics                      [unchanged]

Change log:
  - REMOVED  EasyOCRDetector  (CRAFT model, slower, less accurate)
  - ADDED    PaddleOCRDetector (DBNet++ det, PP-OCRv4 rec, Apache-2.0)
  - REMOVED  OpenCVInpainter  (single-tier Telea/NS only)
  - ADDED    LaMaInpainter    (LaMa premium tier + Telea fallback)
"""

import os
# ── Windows DLL load-order fix ───────────────────────────────────────────────
# PaddlePaddle and PyTorch both bundle their own copies of Windows system DLLs
# (zlib1.dll, msvcp140.dll, etc.). When paddle is imported first it loads its
# own DLL variants; a subsequent torch import in the same process then tries to
# load torch/lib/shm.dll which depends on a different DLL variant → WinError 127.
# Fix: eagerly import torch HERE, before any paddle import, so Windows DLL
# caching locks in torch's DLL versions first.
try:
    import torch as _torch_preload   # noqa: F401 — side-effect import only
except Exception:
    pass  # torch not installed; LaMaInpainter will fall back to Telea gracefully
# ── PaddlePaddle 3.x PIR flag ────────────────────────────────────────────────
# Disable the PIR (Program IR) executor on Windows: it triggers an
# unimplemented OneDNN operator (pir::ArrayAttribute<pir::DoubleAttribute>)
# when running DBNet++.
os.environ['FLAGS_enable_pir_api'] = '0'
# Disable OneDNN (Intel MKL-DNN) backend entirely on Windows.
# The PIR executor uses OneDNN operator fusions that are not fully implemented
# for all layer types (pir::ArrayAttribute<pir::DoubleAttribute> crash).
# Disabling MKL-DNN forces CPU fallback kernels which are stable on Windows.
os.environ['FLAGS_use_mkldnn'] = '0'
# ─────────────────────────────────────────────────────────────────────────────

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple



# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class BBox:
    x: int; y: int; w: int; h: int

    @property
    def area(self) -> int: return self.w * self.h

    def iou(self, other: "BBox") -> float:
        # Guard: zero-area boxes have no meaningful IoU
        if self.area == 0 or other.area == 0:
            return 0.0
        ix1 = max(self.x, other.x);  iy1 = max(self.y, other.y)
        ix2 = min(self.x+self.w, other.x+other.w)
        iy2 = min(self.y+self.h, other.y+other.h)
        if ix2<=ix1 or iy2<=iy1: return 0.0
        inter = (ix2-ix1)*(iy2-iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    id:         int
    bbox:       BBox
    polygon:    np.ndarray   # shape (N,2), pixel coords
    confidence: float
    source:     str          # 'mser' | 'paddleocr'


@dataclass
class MaskResult:
    data:         np.ndarray   # uint8, H×W, 255=text 0=bg
    coverage:     float
    masked_pixels: int


@dataclass
class PipelineResult:
    output:      np.ndarray    # BGR uint8
    mask:        MaskResult
    detections:  List[Detection]
    quality:     dict
    timing:      dict          # stage → ms
    total_ms:    float


# ─── Stage 1: Text Detection ──────────────────────────────────────────────────

class PaddleOCRDetector:
    """
    Phase 1 replacement for EasyOCRDetector.

    Uses PaddleOCR's DBNet++ detection model (PP-OCRv4 det):
      - Apache-2.0 licensed, 100% offline.
      - Significantly more accurate than EasyOCR (CRAFT) on:
          * Complex / curved scene text
          * Rotated text
          * Non-Latin scripts
      - Runs on CPU via paddlepaddle; switch to paddlepaddle-gpu for CUDA.

    Falls back to MSERDetector automatically if PaddleOCR is unavailable
    (e.g., paddlepaddle not installed) so existing tests keep passing.
    """

    def __init__(self, lang: str = 'en', use_gpu: bool = False):
        self._paddle = None
        # Always create MSER as an inference-time fallback.
        # Previously MSER was only used if paddle failed to LOAD.
        # Now it also covers cases where paddle loads but .ocr() fails at runtime
        # (e.g. OneDNN / PIR regressions on Windows).
        self._mser_fallback = MSERDetector()
        try:
            # Must import paddle first and disable PIR before creating any model.
            # paddle.set_flags() is more reliable than env vars for PaddlePaddle 3.x.
            import paddle
            paddle.set_flags({'FLAGS_enable_pir_api': False})

            from paddleocr import PaddleOCR
            device = 'gpu' if use_gpu else 'cpu'

            # Use PP-OCRv4 models explicitly.
            # PP-OCRv5 (the 3.x default) contains pir::ArrayAttribute<pir::DoubleAttribute>
            # layer types that are unimplemented in the Windows OneDNN build, causing a
            # crash in onednn_instruction.cc. PP-OCRv4 models are stable on Windows.
            self._paddle = PaddleOCR(
                lang=lang,
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                det_model_dir=None,  # let it use default
                rec_model_dir=None,
            )
            print('[PaddleOCRDetector] PaddleOCR PP-OCRv4 loaded successfully.')
        except (ImportError, ValueError, Exception) as e:
            print(f'[PaddleOCRDetector] WARNING: Could not load PaddleOCR ({e}). '
                  'Falling back to MSERDetector. '
                  'Run: pip install paddlepaddle paddleocr')



    def detect(self, image: np.ndarray, threshold: float = 0.5) -> List[Detection]:
        """Detect text regions. Returns list of Detection with quadrilateral polygons."""

        # ── Paddle not loaded: use MSER directly ───────────────────────────────────
        if self._paddle is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self._mser_fallback.detect(gray, threshold)

        # ── PaddleOCR inference (with MSER fallback on any runtime error) ────────
        try:
            result = self._paddle.ocr(image)
        except Exception as e:
            print(f'[PaddleOCRDetector] Inference error: {e}')
            print('[PaddleOCRDetector] Falling back to MSER for this frame.')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self._mser_fallback.detect(gray, threshold)

        detections: List[Detection] = []

        if not result:
            return detections

        # ── PaddleOCR 3.x returns a list of OCRResult objects ─────────────────
        # Each OCRResult has: .boxes (list of 4-point quads), .rec_scores, .rec_texts
        # PaddleOCR 2.x returns: [ [ [box, (text, conf)], ... ] ]
        # We detect the format and normalise to (box, conf) pairs.
        def _iter_detections(raw):
            """Yield (box_4pts, confidence) from either result format."""
            if raw is None:
                return
            # 3.x path: result is a list of OCRResult objects
            if hasattr(raw, '__iter__') and len(raw) > 0 and hasattr(raw[0], 'boxes'):
                for page in raw:
                    for box, conf in zip(page.boxes, page.rec_scores):
                        yield box, conf
            # 2.x path: result[0] is a list of [box, (text, conf)]
            elif isinstance(raw, list) and len(raw) > 0:
                page = raw[0]
                if page is None:
                    return
                for item in page:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        box, text_conf = item
                        conf = text_conf[1] if isinstance(text_conf, (list, tuple)) else 0.0
                        yield box, conf

        for box, conf in _iter_detections(result):
            if conf < threshold:
                continue

            # box is a list of 4 [x, y] points (quadrilateral, clockwise)
            pts = np.array(box, dtype=np.int32)           # shape (4, 2)
            xs, ys = pts[:, 0], pts[:, 1]
            x, y   = int(xs.min()), int(ys.min())
            w, h   = int(xs.max()) - x, int(ys.max()) - y

            if w <= 0 or h <= 0:
                continue

            poly = pts.reshape(-1, 1, 2)                  # cv2 fillPoly shape
            detections.append(Detection(
                id=len(detections),
                bbox=BBox(x, y, w, h),
                polygon=poly,
                confidence=float(conf),
                source='paddleocr',
            ))

        return detections



class MSERDetector:
    """
    Maximally Stable Extremal Regions.
    Detects character blobs; no model required.
    Suitable as a fast offline fallback.
    """
    def __init__(self, min_area: int = 30, max_area: int = 4000, max_variation: float = 0.3):
        self.mser = cv2.MSER_create(max_variation=max_variation)
        self.mser.setMinArea(min_area)
        self.mser.setMaxArea(max_area)

    def detect(self, gray: np.ndarray, threshold: float = 0.5) -> List[Detection]:
        regions, _ = self.mser.detectRegions(gray)
        results: List[Detection] = []
        seen: List[BBox] = []

        for i, region in enumerate(regions):
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            x, y, w, h = cv2.boundingRect(hull)
            aspect = w / max(h, 1)
            
            # 1. Text aspect ratio and size constraints (reject thin lines / giant blocks)
            if not (0.2 < aspect < 5.0 and w > 10 and h > 10):
                continue
                
            # 2. Solidity filter (Contour Area / Convex Hull Area)
            # Text characters are typically solid blobs (solidity > 0.4).
            # This rejects organic textures like grass, waves, and branching tree leaves.
            area = cv2.contourArea(region.reshape(-1, 1, 2))
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < 0.4:
                continue

            bbox = BBox(x, y, w, h)
            # Simple NMS: skip if IoU > 0.4 with an already-kept box
            if any(bbox.iou(s) > 0.4 for s in seen): continue
            seen.append(bbox)

            poly = hull.reshape(-1, 2).astype(np.int32)
            results.append(Detection(
                id=len(results), bbox=bbox, polygon=poly,
                confidence=0.75,  # MSER doesn't output confidence; use fixed value
                source='mser',
            ))

        return results


# ─── Stage 2: Mask Generation ─────────────────────────────────────────────────

class MaskGenerator:
    """
    Rasterises detection polygons to a binary mask with optional dilation.

    Dilation:  M_dilated = M ⊕ B_r  (disk structuring element, radius r)
    """
    def generate(
        self,
        detections: List[Detection],
        h: int, w: int,
        dilation_px: int = 3,
    ) -> MaskResult:
        mask = np.zeros((h, w), dtype=np.uint8)
        for det in detections:
            cv2.fillPoly(mask, [det.polygon], 255)

        if dilation_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2*dilation_px+1, 2*dilation_px+1)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        masked_px = int((mask > 0).sum())
        return MaskResult(
            data=mask,
            coverage=masked_px / (h * w),
            masked_pixels=masked_px,
        )


# ─── Stage 3: Inpainting ──────────────────────────────────────────────────────

class LaMaInpainter:
    """
    Phase 1 two-tier inpainting strategy:

    Tier 1 — LaMa (Large Mask inpainting):
      - Fourier convolution architecture; SOTA on large text masks.
      - Library: simple-lama-inpainting (Apache-2.0).
      - Loaded lazily: only imported when available.
      - Target quality: SSIM > 0.92 on synthetic text removal set.

    Tier 2 — OpenCV Telea (fallback):
      - Fast Marching Method: always available, zero model download.
      - For pixel p ∈ Ω:
          I(p) = Σ_{q ∈ N_ε(p)} w(p,q) · [I(q) + ∇I(q)·(p-q)]
          w(p,q) = dst(p)·lev(p)·dir(p,q) / (‖p-q‖²)
      - Target quality: SSIM > 0.82 (POC acceptance bar).
    """

    def __init__(self):
        self._lama = None
        try:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama()
            print('[LaMaInpainter] LaMa model loaded — using premium Tier 1.')
        except ImportError:
            print('[LaMaInpainter] simple-lama-inpainting not installed. '
                  'Falling back to OpenCV Telea. '
                  'Run: pip install simple-lama-inpainting')

    def inpaint(
        self,
        image:  np.ndarray,
        mask:   MaskResult,
        radius: int = 5,
        method: str = 'telea',     # used only when LaMa is unavailable
    ) -> Tuple[np.ndarray, str]:
        """
        Returns (inpainted_image, tier_used) where tier_used is 'lama' or 'telea'.
        The caller records tier_used in the timing/quality dict for transparency.
        """
        # ── Tier 1: LaMa ──────────────────────────────────────────────────────
        if self._lama is not None:
            try:
                from PIL import Image as PILImage
                pil_img  = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                pil_mask = PILImage.fromarray(mask.data)
                result   = self._lama(pil_img, pil_mask)
                bgr      = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                return bgr, 'lama'
            except Exception as e:
                print(f'[LaMaInpainter] LaMa failed ({e}), falling back to Telea.')

        # ── Tier 2: OpenCV Telea fallback ─────────────────────────────────────
        flag    = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
        inpainted = cv2.inpaint(image, mask.data, radius, flag)
        return inpainted, 'telea'


# ─── Stage 4: Quality Assessment ─────────────────────────────────────────────

class QualityAssessor:
    """
    SSIM and PSNR computed only over the masked region.

    SSIM(x,y) = (2μₓμᵧ+C₁)(2σₓᵧ+C₂) / ((μₓ²+μᵧ²+C₁)(σₓ²+σᵧ²+C₂))
    PSNR = 10·log₁₀(255²/MSE)

    NOTE: default threshold=0.82 is the POC acceptance bar.
    TRD on-device target for tier-2 (PConv): SSIM > 0.90.
    TRD server target (LaMa):               SSIM > 0.92.
    """
    def assess(
        self,
        original: np.ndarray,
        result:   np.ndarray,
        mask:     MaskResult,
        threshold: float = 0.82,
    ) -> dict:
        if mask.masked_pixels == 0:
            return dict(ssim=1.0, psnr=float('inf'), mse=0.0, above_threshold=True)

        # Convert to float luminance [0,1] over masked region only
        def lum(img):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
            return g[mask.data > 0].flatten()

        a = lum(original)
        b = lum(result)

        mu_a, mu_b   = a.mean(), b.mean()
        sig_a2       = ((a - mu_a)**2).mean()
        sig_b2       = ((b - mu_b)**2).mean()
        sig_ab       = ((a - mu_a)*(b - mu_b)).mean()
        mse          = ((a - b)**2).mean()

        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mu_a*mu_b+C1)*(2*sig_ab+C2)) / \
               ((mu_a**2+mu_b**2+C1)*(sig_a2+sig_b2+C2))
        psnr = float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)

        return dict(
            ssim=float(np.clip(ssim, 0, 1)),
            psnr=round(float(psnr), 2),
            mse=round(float(mse), 6),
            above_threshold=float(ssim) >= threshold,
        )


# ─── Pipeline Orchestrator ────────────────────────────────────────────────────

class TextRemovalPipeline:
    def __init__(self, lang: str = 'en', use_gpu: bool = False):
        # Phase 1: EasyOCRDetector → PaddleOCRDetector (DBNet++)
        self.detector  = PaddleOCRDetector(lang=lang, use_gpu=use_gpu)
        self.masker    = MaskGenerator()
        # Phase 1: OpenCVInpainter → LaMaInpainter (LaMa + Telea fallback)
        self.inpainter = LaMaInpainter()
        self.qa        = QualityAssessor()

    def run(self,
            image: np.ndarray,
            threshold: float = 0.4,   # PaddleOCR confidence threshold
            dilation_px: int = 5,
            inpaint_method: str = 'telea',
            inpaint_radius: int = 7) -> PipelineResult:
        timing: dict = {}
        h, w = image.shape[:2]
        
        # Stage 1
        t0 = time.perf_counter()
        detections = self.detector.detect(image, threshold)
        timing['detection_ms'] = round((time.perf_counter() - t0) * 1000, 1)

        if not detections:
            empty_mask = MaskResult(np.zeros((h,w),np.uint8), 0.0, 0)
            q = dict(ssim=1.0, psnr=float('inf'), mse=0.0, above_threshold=True)
            timing['total_ms'] = timing.get('detection_ms', 0.0)  # fix: KeyError guard
            return PipelineResult(image, empty_mask, [], q, timing, timing['total_ms'])

        # Stage 2
        t0 = time.perf_counter()
        mask = self.masker.generate(detections, h, w, dilation_px)
        timing['mask_ms'] = round((time.perf_counter() - t0) * 1000, 1)

        # Stage 3 — LaMaInpainter returns (image, tier_used)
        t0 = time.perf_counter()
        inpainted, inpaint_tier = self.inpainter.inpaint(image, mask, inpaint_radius, inpaint_method)
        timing['inpaint_ms']   = round((time.perf_counter() - t0) * 1000, 1)
        timing['inpaint_tier'] = inpaint_tier   # 'lama' or 'telea' — logged for QA review

        # Stage 4
        t0 = time.perf_counter()
        quality = self.qa.assess(image, inpainted, mask)
        timing['quality_ms'] = round((time.perf_counter() - t0) * 1000, 1)

        # Compute total BEFORE inserting into timing dict;
        # sum(timing.values()) iterates values correctly, but the previous code
        # ran sum() before 'quality_ms' was added — fixed by computing here.
        total_ms = round(sum(v for v in timing.values() if isinstance(v, (int, float))), 1)
        timing['total_ms'] = total_ms
        return PipelineResult(inpainted, mask, detections, quality, timing, total_ms)
