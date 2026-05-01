"""
poc_demo.py
-------------------------------------------------------------------------------
Proof-of-Concept demonstration for the Text Detection & Removal System (TDRS).

WHAT THIS SCRIPT DOES — end to end:
  1. Builds a synthetic test image: gradient background + 3 text overlays
     (simulates a watermarked/captioned photo without needing a real camera)

  2. Runs the full 4-stage Python pipeline (pipeline/core.py):
       Stage 1 — MSERDetector      : finds text-blob candidates (no model needed)
       Stage 2 — MaskGenerator     : rasterises polygons → binary mask + dilation
       Stage 3 — OpenCVInpainter   : Telea Fast-Marching fills the masked region
       Stage 4 — QualityAssessor   : SSIM + PSNR measured over the mask region

  3. Saves 5 images to ./poc_output/:
       01_original.png        — input image WITH text overlays
       02_mask.png            — the binary mask the pipeline produced
       03_result.png          — cleaned image AFTER inpainting
       04_comparison.png      — side-by-side: Original | Mask | Result
       05_clean_reference.png — same background WITHOUT any text (ground truth)

  4. Prints a full timing breakdown and quality measurement.

DOES THE MAIN POC CODE DO THE WORK?
  YES. poc_demo.py is just a runner / visualiser.
  All the real logic lives in:
    python-backend/pipeline/core.py
      - MSERDetector      -> Stage 1
      - MaskGenerator     -> Stage 2
      - OpenCVInpainter   -> Stage 3
      - QualityAssessor   -> Stage 4
      - TextRemovalPipeline -> orchestrates all stages

HOW TO RUN:
  python poc_demo.py            (Windows: no -X utf8 needed after this fix)

DEPENDENCIES (all standard, already installed):
  opencv-python >= 4.9
  numpy >= 1.26
"""

import os
import sys

# Make sure the python-backend package is importable without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-backend'))

import cv2
import numpy as np
import time

from pipeline.core import TextRemovalPipeline, QualityAssessor


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUT_DIR = os.path.join(os.path.dirname(__file__), 'poc_output')
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg=''):
    """Print to stdout, flushing immediately so terminals that buffer don't hide output."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Step 1 — Build a synthetic test image
# ---------------------------------------------------------------------------

log()
log("=" * 62)
log("   TEXT DETECTION & REMOVAL SYSTEM  --  POC DEMO")
log("=" * 62)
log()
log("[Step 1/4]  Building synthetic test image ...")

H, W = 480, 640

# Smooth gradient background — mimics a real photo background
img_clean = np.zeros((H, W, 3), dtype=np.uint8)
for y in range(H):
    img_clean[y, :, 0] = min(255, int(30  + y * 0.15))   # B
    img_clean[y, :, 1] = min(255, int(60  + y * 0.25))   # G
    img_clean[y, :, 2] = min(255, int(180 - y * 0.30))   # R

# Light noise adds texture so inpainting has something realistic to reconstruct
rng = np.random.default_rng(seed=42)   # fixed seed = reproducible
noise = rng.integers(-8, 9, img_clean.shape, dtype=np.int16)
img_clean = np.clip(img_clean.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Overlay 3 text regions — simulates: watermark, subtitle, branding
img_with_text = img_clean.copy()
cv2.putText(img_with_text, "SAMPLE WATERMARK",
            (40, 100),  cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
cv2.putText(img_with_text, "Confidential",
            (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20,  20,  230), 3)
cv2.putText(img_with_text, "www.example.com",
            (80,  400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 10),  2)

cv2.imwrite(os.path.join(OUT_DIR, '01_original.png'), img_with_text)
cv2.imwrite(os.path.join(OUT_DIR, '05_clean_reference.png'), img_clean)
log(f"   Image size : {W}x{H} px  |  3 text overlays added")
log(f"   Saved      : poc_output/01_original.png")
log(f"   Saved      : poc_output/05_clean_reference.png")


# ---------------------------------------------------------------------------
# Step 2 — Run the full pipeline  (all the actual work happens here)
# ---------------------------------------------------------------------------

log()
log("[Step 2/4]  Running pipeline (pipeline/core.py) ...")
log()
log("   The pipeline stages that are about to run:")
log("     Stage 1  MSERDetector    -- scans image for text-like blobs")
log("     Stage 2  MaskGenerator   -- fills blobs into a binary mask + dilates")
log("     Stage 3  OpenCVInpainter -- Telea Fast Marching fills the masked area")
log("     Stage 4  QualityAssessor -- computes SSIM + PSNR over mask region")
log()

pipeline = TextRemovalPipeline()

t_start = time.perf_counter()
result = pipeline.run(
    image          = img_with_text,
    threshold      = 0.5,       # confidence cutoff for MSER detections
    dilation_px    = 5,         # pixel radius to expand mask edges
    inpaint_method = 'telea',   # 'telea' = Fast Marching | 'ns' = Navier-Stokes
    inpaint_radius = 7,         # neighbourhood radius for inpainting
)
t_total_ms = (time.perf_counter() - t_start) * 1000

log("   Pipeline complete.")


# ---------------------------------------------------------------------------
# Step 3 — Save visual outputs
# ---------------------------------------------------------------------------

log()
log("[Step 3/4]  Saving visual outputs ...")

# Mask is single-channel (H x W) — convert to BGR for saving as PNG
mask_vis = cv2.cvtColor(result.mask.data, cv2.COLOR_GRAY2BGR)
cv2.imwrite(os.path.join(OUT_DIR, '02_mask.png'), mask_vis)

cv2.imwrite(os.path.join(OUT_DIR, '03_result.png'), result.output)

# Side-by-side: Original | Mask | Result   (easiest way to judge quality)
sep = np.full((H, 6, 3), 60, dtype=np.uint8)   # thin dark separator
comparison = np.hstack([img_with_text, sep, mask_vis, sep, result.output])
cv2.imwrite(os.path.join(OUT_DIR, '04_comparison.png'), comparison)

log("   Saved  poc_output/02_mask.png")
log("   Saved  poc_output/03_result.png")
log("   Saved  poc_output/04_comparison.png   <-- OPEN THIS to see everything")


# ---------------------------------------------------------------------------
# Step 4 — Print results
# ---------------------------------------------------------------------------

log()
log("[Step 4/4]  Results")
log("-" * 62)

log(f"  Text regions detected : {len(result.detections)}")
log(f"  Mask coverage         : {result.mask.coverage * 100:.1f}%  of image area")
log(f"  Masked pixels         : {result.mask.masked_pixels:,}")
log()

log("  Timing breakdown:")
stage_keys = [k for k in result.timing if k != 'total_ms']
for k in stage_keys:
    ms  = result.timing[k]
    bar = '|' * max(1, int(ms / 15))
    log(f"    {k:<18} {ms:>7.1f} ms  {bar}")
log(f"    {'TOTAL':<18} {result.timing['total_ms']:>7.1f} ms")
log()

# Quality is measured against the clean reference (ground truth)
# so we know how close inpainting got to reconstructing the original background.
qa = QualityAssessor()
quality = qa.assess(img_clean, result.output, result.mask, threshold=0.82)

log("  Quality  (result vs clean-reference ground truth):")
log(f"    SSIM   : {quality['ssim']:.4f}    target >= 0.82  -->  {'PASS' if quality['ssim'] >= 0.82 else 'FAIL'}")
log(f"    PSNR   : {quality['psnr']:.2f} dB  target >= 28 dB  -->  {'PASS' if quality['psnr'] >= 28 else 'FAIL'}")
log(f"    MSE    : {quality['mse']:.6f}")
log()
log(f"  Wall-clock total : {t_total_ms:.1f} ms  (target < 3000 ms on CPU)")
log("-" * 62)

# Final verdict
log()
if len(result.detections) == 0:
    log("  [!] WARNING: No text detected.")
    log("      MSER needs higher-contrast text or a lower min_area threshold.")
elif quality['above_threshold']:
    log("  [PASS]  Inpainting quality meets the POC threshold.")
    log("          Algorithm is validated. Ready for native sprint.")
else:
    log("  [NOTE]  Quality below 0.82 on this image.")
    log("          For complex masks the server LaMa tier (Tier 3) would be used.")

log()
log("  Open these files to visually inspect the output:")
log(f"    {os.path.join(OUT_DIR, '04_comparison.png')}")
log(f"    {os.path.join(OUT_DIR, '02_mask.png')}")
log(f"    {os.path.join(OUT_DIR, '03_result.png')}")
log()
