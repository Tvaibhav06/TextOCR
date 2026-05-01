/**
 * Stage 3 — Inpainting
 *
 * Two-tier strategy:
 *
 *  Tier 1 — OpenCV Telea (on-device, no model, always available)
 *    - Algorithm: Fast Marching Method along mask boundary
 *    - Mathematical foundation:
 *        For pixel p in mask Ω, reconstruct from neighbourhood N_ε(p):
 *        I(p) = Σ_{q ∈ N_ε(p)} w(p,q) · [I(q) + ∇I(q)·(p-q)]
 *        where w(p,q) = dst(p)·lev(p)·dir(p,q) / (|p-q|·|p-q|)
 *          dst = distance transform, lev = level-set, dir = gradient direction
 *    - Runs in O(N log N) via priority queue over masked pixels
 *
 *  Tier 2 — AOT-compiled PConv TFLite (~12 MB INT8 model)
 *    - Partial Convolution: renormalises output by valid pixel ratio
 *        x_out = (W^T · (X ⊙ M)) · (|K²| / sum(M)) + b
 *    - Better on textured backgrounds than Telea
 *    - ~3–8 s on mobile ARM, loaded on-demand
 */

import { NativeModules } from 'react-native';
import { loadTensorflowModel } from 'react-native-fast-tflite';
import type { MaskResult } from './MaskGenerator';

// ─── Types ────────────────────────────────────────────────────────────────────

export type InpaintMethod = 'telea' | 'ns' | 'pconv';

export interface InpaintConfig {
  method: InpaintMethod;       // default 'telea'
  inpaintRadius: number;       // Telea/NS neighbourhood radius, default 5
  /** Auto-upgrade to PConv when mask coverage > this threshold */
  pconvCoverageThreshold: number;  // default 0.15 (15%)
}

export interface InpaintResult {
  /** URI of the inpainted image written to temp storage */
  outputUri: string;
  method: InpaintMethod;
  processingMs: number;
}

const DEFAULT_INPAINT_CONFIG: InpaintConfig = {
  method: 'telea',
  inpaintRadius: 5,
  pconvCoverageThreshold: 0.15,
};

// ─── OpenCV Native Bridge ─────────────────────────────────────────────────────

/**
 * OpenCVInpainter bridges to the native OpenCV module.
 *
 * Native side (Android: JNI via CMake / iOS: Objective-C++ wrapper):
 *   Mat img   = imread(imagePath);
 *   Mat mask  = Scalar(0) then fill from maskData (Uint8Array over bridge)
 *   inpaint(img, mask, dst, radius, INPAINT_TELEA | INPAINT_NS)
 *   imwrite(outputPath, dst)
 *
 * Bridge passes:
 *   - imageUri: local file path
 *   - maskBase64: base64-encoded single-channel PNG of the mask
 *   - radius: inpaint neighbourhood radius
 *   - method: 'telea' | 'ns'
 */
class OpenCVInpainter {
  private module = NativeModules.OpenCVInpaint;

  async isAvailable(): Promise<boolean> {
    return typeof this.module?.inpaint === 'function';
  }

  async inpaint(
    imageUri: string,
    mask: MaskResult,
    config: InpaintConfig,
  ): Promise<string> {
    const maskBase64 = this.maskToBase64PNG(mask);

    const outputUri: string = await this.module.inpaint({
      imageUri,
      maskBase64,
      radius: config.inpaintRadius,
      method: config.method === 'ns' ? 'ns' : 'telea',
    });

    return outputUri;
  }

  /**
   * Encode single-channel Uint8Array mask as a base64 string.
   *
   * ⚠️ POC NOTE: Uses btoa() which is available in React Native's Hermes runtime.
   * `Buffer` is a Node.js global — it does NOT exist in Hermes and will crash
   * at runtime. btoa() on a binary string is the Hermes-safe equivalent.
   *
   * Production upgrade path: write mask to a temp file via expo-file-system and
   * read it back as base64 — avoids the full RGBA buffer allocation over the bridge.
   */
  private maskToBase64PNG(mask: MaskResult): string {
    const rgba = new Uint8Array(mask.width * mask.height * 4);
    for (let i = 0; i < mask.width * mask.height; i++) {
      const v = mask.data[i];
      rgba[i * 4]     = v;
      rgba[i * 4 + 1] = v;
      rgba[i * 4 + 2] = v;
      rgba[i * 4 + 3] = 255;
    }
    // Hermes-safe base64 encoding — no Buffer, no external lib
    let bin = '';
    for (let i = 0; i < rgba.length; i++) bin += String.fromCharCode(rgba[i]);
    return btoa(bin);
  }
}

// ─── PConv TFLite Inpainter ───────────────────────────────────────────────────

/**
 * PConvInpainter runs the AOT-compiled Partial Convolution model.
 *
 * Input tensor:  [1, H, W, 4]  — RGBA with alpha = mask (1=valid, 0=masked)
 * Output tensor: [1, H, W, 3]  — reconstructed RGB
 *
 * Model processes 512×512 tiles; result stitched with linear blend in overlaps.
 */
class PConvInpainter {
  private model: any = null;
  private readonly TILE_SIZE = 512;
  private readonly OVERLAP   = 64;

  async load(): Promise<void> {
    this.model = await loadTensorflowModel(
      require('../../assets/models/pconv_inpainter_int8.tflite'),
    );
    console.log('[PConvInpainter] Model loaded');
  }

  isLoaded() { return this.model !== null; }

  async inpaint(
    imageRGBA: Uint8Array,
    mask: MaskResult,
    width: number,
    height: number,
  ): Promise<Uint8Array> {
    if (!this.model) throw new Error('PConv model not loaded');

    const output = new Uint8Array(width * height * 4);

    // Tile-based processing with overlap blending
    for (let ty = 0; ty < height; ty += this.TILE_SIZE - this.OVERLAP) {
      for (let tx = 0; tx < width; tx += this.TILE_SIZE - this.OVERLAP) {
        const tw = Math.min(this.TILE_SIZE, width  - tx);
        const th = Math.min(this.TILE_SIZE, height - ty);

        // Extract tile — RGBA + mask channel → [1, 512, 512, 4] float32
        const inputTensor = this.extractTile(imageRGBA, mask.data, tx, ty, tw, th, width);

        // Run inference: [1, th, tw, 3] float32
        const [reconstructed] = await this.model.run([inputTensor]);

        // Write tile to output buffer with linear blend in overlap zone
        this.blendTile(output, reconstructed as Float32Array, tx, ty, tw, th, width, height);
      }
    }

    return output;
  }

  private extractTile(
    rgba: Uint8Array,
    mask: Uint8Array,
    tx: number, ty: number,
    tw: number, th: number,
    fullW: number,
  ): Float32Array {
    const tensor = new Float32Array(this.TILE_SIZE * this.TILE_SIZE * 4);
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        const srcIdx = ((ty + y) * fullW + (tx + x)) * 4;
        const dstIdx = (y * this.TILE_SIZE + x) * 4;
        tensor[dstIdx]     = rgba[srcIdx]     / 255;  // R
        tensor[dstIdx + 1] = rgba[srcIdx + 1] / 255;  // G
        tensor[dstIdx + 2] = rgba[srcIdx + 2] / 255;  // B
        // alpha = valid pixel indicator (inverse of mask)
        tensor[dstIdx + 3] = mask[(ty + y) * fullW + (tx + x)] > 0 ? 0 : 1;
      }
    }
    return tensor;
  }

  private blendTile(
    output: Uint8Array,
    reconstructed: Float32Array,
    tx: number, ty: number,
    tw: number, th: number,
    fullW: number, fullH: number,
  ): void {
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        const oy = ty + y;
        const ox = tx + x;
        if (oy >= fullH || ox >= fullW) continue;

        // Linear blend weight in overlap zone
        const wx = x < this.OVERLAP ? x / this.OVERLAP : (tw - x) < this.OVERLAP ? (tw - x) / this.OVERLAP : 1;
        const wy = y < this.OVERLAP ? y / this.OVERLAP : (th - y) < this.OVERLAP ? (th - y) / this.OVERLAP : 1;
        const w  = wx * wy;

        const srcIdx = (y * tw + x) * 3;
        const dstIdx = (oy * fullW + ox) * 4;

        output[dstIdx]     = Math.round(Math.min(255, output[dstIdx]     * (1 - w) + reconstructed[srcIdx]     * 255 * w));
        output[dstIdx + 1] = Math.round(Math.min(255, output[dstIdx + 1] * (1 - w) + reconstructed[srcIdx + 1] * 255 * w));
        output[dstIdx + 2] = Math.round(Math.min(255, output[dstIdx + 2] * (1 - w) + reconstructed[srcIdx + 2] * 255 * w));
        output[dstIdx + 3] = 255;
      }
    }
  }
}

// ─── Unified Inpainting Facade ────────────────────────────────────────────────

export class Inpainter {
  private opencv  = new OpenCVInpainter();
  private pconv   = new PConvInpainter();
  private pconvLoaded = false;

  async warmUp(): Promise<void> {
    try {
      await this.pconv.load();
      this.pconvLoaded = true;
    } catch (err) {
      console.warn('[Inpainter] PConv load failed, Telea-only mode:', err);
    }
  }

  async inpaint(
    imageUri: string,
    mask: MaskResult,
    config: Partial<InpaintConfig> = {},
  ): Promise<InpaintResult> {
    const cfg = { ...DEFAULT_INPAINT_CONFIG, ...config };
    const t0  = performance.now();

    // Auto-upgrade method for high-coverage masks
    const effectiveMethod: InpaintMethod =
      cfg.method === 'telea' && mask.coverage > cfg.pconvCoverageThreshold && this.pconvLoaded
        ? 'pconv'
        : cfg.method;

    console.log(`[Inpainter] coverage=${(mask.coverage * 100).toFixed(1)}% → method=${effectiveMethod}`);

    let outputUri: string;

    if (effectiveMethod === 'pconv' && this.pconvLoaded) {
      // PConv path: read image pixels → run model → write output
      outputUri = await this.runPConv(imageUri, mask, cfg);
    } else {
      // Telea / NS path: delegate to native OpenCV
      outputUri = await this.opencv.inpaint(imageUri, mask, { ...cfg, method: effectiveMethod as 'telea' | 'ns' });
    }

    return {
      outputUri,
      method: effectiveMethod,
      processingMs: Math.round(performance.now() - t0),
    };
  }

  private async runPConv(imageUri: string, mask: MaskResult, cfg: InpaintConfig): Promise<string> {
    // ⚠️ POC STUB: PConv file I/O is not yet implemented.
    //
    // What's needed for full implementation:
    //   1. Read imageUri pixels into Uint8Array via expo-file-system readAsStringAsync (base64)
    //   2. Pass pixels + mask to this.pconv.inpaint(imageRGBA, mask, width, height)
    //   3. Write the output Uint8Array to a temp file & return its URI
    //
    // For now: fall back to OpenCV Telea so the pipeline remains runnable
    // even when mask.coverage falls in the PConv range (5–25%).
    // Q2 decision: no status flag in InpaintResult for this POC iteration.
    console.warn(
      '[Inpainter] PConv I/O not implemented in POC — falling back to Telea.',
      `coverage=${(mask.coverage * 100).toFixed(1)}%`,
    );
    return this.opencv.inpaint(imageUri, mask, {
      method: 'telea',
      inpaintRadius: cfg.inpaintRadius,
      pconvCoverageThreshold: cfg.pconvCoverageThreshold,
    });
  }
}

export const inpainter = new Inpainter();
