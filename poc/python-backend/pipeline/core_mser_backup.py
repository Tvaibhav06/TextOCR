"""
python-backend/pipeline/core.py
────────────────────────────────────────────────────────────────────────────────
Backend pipeline that mirrors the mobile stages.
Used for:
  1. Verifying algorithmic correctness before porting to TS/native
  2. Server-side LaMa fallback when mobile quality is insufficient
  3. pytest integration tests run in CI

Stages match the mobile pipeline 1:1:
  Stage 1 — Detection   : OpenCV MSER (no model, instant) + PaddleOCR stub
  Stage 2 — Mask        : cv2.fillPoly + morphological dilation
  Stage 3 — Inpainting  : cv2.inpaint (Telea / NS)
  Stage 4 — Post        : Poisson blend stub + SSIM/PSNR metrics
"""

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

class OpenCVInpainter:
    """
    Telea inpainting — Fast Marching Method.

    For pixel p ∈ Ω:
      I(p) = Σ_{q ∈ N_ε(p)} w(p,q) · [I(q) + ∇I(q)·(p-q)]
      w(p,q) = dst(p)·lev(p)·dir(p,q) / (‖p-q‖²)
    """
    def inpaint(
        self,
        image: np.ndarray,
        mask:  MaskResult,
        radius: int = 5,
        method: str = 'telea',
    ) -> np.ndarray:
        flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
        return cv2.inpaint(image, mask.data, radius, flag)


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
    def __init__(self):
        self.detector  = MSERDetector()
        self.masker    = MaskGenerator()
        self.inpainter = OpenCVInpainter()
        self.qa        = QualityAssessor()

    def run(self,
            image: np.ndarray,
            threshold: float = 0.5,
            dilation_px: int = 5,
            inpaint_method: str = 'telea',
            inpaint_radius: int = 7) -> PipelineResult:
        timing: dict = {}
        h, w = image.shape[:2]
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Stage 1
        t0 = time.perf_counter()
        detections = self.detector.detect(gray, threshold)
        timing['detection_ms'] = round((time.perf_counter() - t0) * 1000, 1)

        if not detections:
            empty_mask = MaskResult(np.zeros((h,w),np.uint8), 0.0, 0)
            q = dict(ssim=1.0, psnr=float('inf'), mse=0.0, above_threshold=True)
            return PipelineResult(image, empty_mask, [], q, timing, 0.0)

        # Stage 2
        t0 = time.perf_counter()
        mask = self.masker.generate(detections, h, w, dilation_px)
        timing['mask_ms'] = round((time.perf_counter() - t0) * 1000, 1)

        # Stage 3
        t0 = time.perf_counter()
        inpainted = self.inpainter.inpaint(image, mask, inpaint_radius, inpaint_method)
        timing['inpaint_ms'] = round((time.perf_counter() - t0) * 1000, 1)

        # Stage 4
        t0 = time.perf_counter()
        quality = self.qa.assess(image, inpainted, mask)
        timing['quality_ms'] = round((time.perf_counter() - t0) * 1000, 1)

        # Compute total BEFORE inserting into timing dict;
        # sum(timing.values()) iterates values correctly, but the previous code
        # ran sum() before 'quality_ms' was added — fixed by computing here.
        total_ms = round(sum(timing.values()), 1)
        timing['total_ms'] = total_ms
        return PipelineResult(inpainted, mask, detections, quality, timing, total_ms)
