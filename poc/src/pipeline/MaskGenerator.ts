/**
 * Stage 2 — Mask Generation
 *
 * Takes detection polygons → renders binary mask via React Native Skia offscreen canvas.
 * Skia runs on the GPU, making fillPath fast even at 1080p.
 *
 * Algorithm:
 *  1. Create offscreen Skia surface at image resolution
 *  2. Fill black background
 *  3. For each detected polygon: draw filled white path
 *  4. Morphological dilation via custom kernel convolution on the Uint8 bitmap
 *  5. Return Uint8Array mask + SkImage for Skia-based compositing
 *
 * Dilation math:
 *   M_dilated = M ⊕ B_r  where B_r = disk of radius r
 *   Approximated with 3×3 or 5×5 box kernel (separable, fast on CPU)
 */

import {
  Skia,
  SkCanvas,
  SkSurface,
  PaintStyle,
  BlendMode,
  ColorType,
} from '@shopify/react-native-skia';
import type { DetectionResult } from './TextDetector';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface MaskConfig {
  /** Dilation radius in pixels */
  dilationPx: number;        // default 3
  /** Soft-edge feathering (Gaussian sigma). 0 = hard binary mask */
  featherSigma: number;      // default 0
}

export interface MaskResult {
  /** Raw binary mask: 255 = text region, 0 = background */
  data: Uint8Array;
  width: number;
  height: number;
  /** Number of masked pixels */
  maskedPixels: number;
  /** Fraction of image covered [0,1] */
  coverage: number;
}

const DEFAULT_MASK_CONFIG: MaskConfig = {
  dilationPx: 3,
  featherSigma: 0,
};

// ─── Skia-based Mask Renderer ─────────────────────────────────────────────────

export class MaskGenerator {
  /**
   * Render binary mask from detection polygons using Skia offscreen surface.
   */
  async generate(
    detections: DetectionResult[],
    imageWidth: number,
    imageHeight: number,
    config: Partial<MaskConfig> = {},
  ): Promise<MaskResult> {
    const cfg = { ...DEFAULT_MASK_CONFIG, ...config };

    // Create offscreen Skia surface (rendered on GPU thread)
    const surface = Skia.Surface.Make(imageWidth, imageHeight);
    if (!surface) throw new Error('[MaskGenerator] Could not create Skia surface');

    const canvas = surface.getCanvas();
    this.renderMask(canvas, detections, imageWidth, imageHeight);
    surface.flush();

    // Read back pixels as RGBA Uint8Array (4 bytes per pixel)
    const snapshot = surface.makeImageSnapshot();
    const pixelData = snapshot.readPixels(0, 0, {
      width: imageWidth,
      height: imageHeight,
      // ColorType.RGBA_8888 = 4 in Skia's SkColorType enum.
      // Using the named constant avoids magic-number confusion.
      colorType: ColorType.RGBA_8888,
      alphaType: 1,   // kOpaque
    });
    if (!pixelData) throw new Error('[MaskGenerator] Failed to read pixels from surface');

    // Extract single-channel mask from red channel (since we drew white = 255,255,255)
    const mask = this.extractSingleChannel(new Uint8Array(pixelData), imageWidth, imageHeight);

    // Morphological dilation
    const dilated = cfg.dilationPx > 0
      ? this.dilate(mask, imageWidth, imageHeight, cfg.dilationPx)
      : mask;

    // Feathering (Gaussian blur on mask edges)
    const final = cfg.featherSigma > 0
      ? this.gaussianBlur(dilated, imageWidth, imageHeight, cfg.featherSigma)
      : dilated;

    const maskedPixels = final.reduce((acc, v) => acc + (v > 0 ? 1 : 0), 0);

    return {
      data: final,
      width: imageWidth,
      height: imageHeight,
      maskedPixels,
      coverage: maskedPixels / (imageWidth * imageHeight),
    };
  }

  // ── Private: render polygons to Skia canvas ──

  private renderMask(
    canvas: SkCanvas,
    detections: DetectionResult[],
    width: number,
    height: number,
  ): void {
    // Black background
    const bgPaint = Skia.Paint();
    bgPaint.setColor(Skia.Color('black'));
    bgPaint.setStyle(PaintStyle.Fill);
    canvas.drawRect({ x: 0, y: 0, width, height }, bgPaint);

    // White fill for each text polygon
    const textPaint = Skia.Paint();
    textPaint.setColor(Skia.Color('white'));
    textPaint.setStyle(PaintStyle.Fill);
    textPaint.setBlendMode(BlendMode.Src);

    for (const det of detections) {
      const path = Skia.Path.Make();
      const pts  = det.polygon.points;
      if (pts.length < 3) continue;

      // De-normalise: multiply by image dimensions
      path.moveTo(pts[0].x * width, pts[0].y * height);
      for (let i = 1; i < pts.length; i++) {
        path.lineTo(pts[i].x * width, pts[i].y * height);
      }
      path.close();
      canvas.drawPath(path, textPaint);
    }
  }

  // ── Private: RGBA → single channel ──

  private extractSingleChannel(rgba: Uint8Array, w: number, h: number): Uint8Array {
    const out = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) {
      out[i] = rgba[i * 4]; // Red channel (same as G and B since we drew white)
    }
    return out;
  }

  // ── Private: morphological dilation (disk approximation) ──
  /**
   * Fast separable box dilation — approximates disk of radius r with
   * two 1D box kernels of size (2r+1).
   *
   * M_dilated(x,y) = 1  if  ∃(dx,dy): dx²+dy² ≤ r²  and  M(x+dx, y+dy) = 1
   *
   * The separable approximation runs in O(W·H·r) instead of O(W·H·r²).
   */
  private dilate(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
    const tmp = new Uint8Array(w * h);
    const out = new Uint8Array(w * h);

    // Horizontal pass
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let max = 0;
        for (let dx = -r; dx <= r; dx++) {
          const nx = x + dx;
          if (nx >= 0 && nx < w) max = Math.max(max, mask[y * w + nx]);
        }
        tmp[y * w + x] = max;
      }
    }

    // Vertical pass on tmp
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let max = 0;
        for (let dy = -r; dy <= r; dy++) {
          const ny = y + dy;
          if (ny >= 0 && ny < h) max = Math.max(max, tmp[ny * w + x]);
        }
        out[y * w + x] = max;
      }
    }

    return out;
  }

  // ── Private: Gaussian blur (1D separable) ──
  /**
   * Gaussian kernel: G(x) = exp(-x²/2σ²), kernel size = ceil(3σ)
   */
  private gaussianBlur(mask: Uint8Array, w: number, h: number, sigma: number): Uint8Array {
    const r      = Math.ceil(sigma * 3);
    const kernel = new Float32Array(2 * r + 1);
    let   ksum   = 0;
    for (let i = -r; i <= r; i++) {
      kernel[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma));
      ksum += kernel[i + r];
    }
    for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;

    const tmp = new Float32Array(w * h);
    const out = new Uint8Array(w * h);

    // Horizontal pass
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let acc = 0;
        for (let i = -r; i <= r; i++) {
          const nx = Math.min(Math.max(x + i, 0), w - 1);
          acc += mask[y * w + nx] * kernel[i + r];
        }
        tmp[y * w + x] = acc;
      }
    }

    // Vertical pass
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let acc = 0;
        for (let i = -r; i <= r; i++) {
          const ny = Math.min(Math.max(y + i, 0), h - 1);
          acc += tmp[ny * w + x] * kernel[i + r];
        }
        out[y * w + x] = Math.round(Math.min(255, Math.max(0, acc)));
      }
    }

    return out;
  }
}

// ─── Utility: mask stats for pipeline logging ──────────────────────────────

export function maskStats(mask: MaskResult) {
  return {
    coverage: (mask.coverage * 100).toFixed(2) + '%',
    maskedPx: mask.maskedPixels.toLocaleString(),
    totalPx:  (mask.width * mask.height).toLocaleString(),
    dims:     `${mask.width}×${mask.height}`,
  };
}

export const maskGenerator = new MaskGenerator();
