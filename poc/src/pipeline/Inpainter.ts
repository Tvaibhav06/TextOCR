/**
 * Stage 3 — Inpainting
 *
 * Active strategy:
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
 *  Tier 2 — OpenCV NS (Navier-Stokes, on-device, no model)
 *    - Higher quality on smooth gradients, slower than Telea
 *
 * Future upgrade path (not yet implemented):
 *  - TFLite PConv — requires real INT8 model + pixel I/O (see M-2 in verification_report)
 */

import { NativeModules } from 'react-native';
import type { MaskResult } from './MaskGenerator';

// ─── Hermes-safe base64 encoder ───────────────────────────────────────────────
// btoa() is a browser Web API — it does NOT exist in Hermes on React Native
// 0.73.x. This pure-JS implementation works on any JS engine.
const BASE64_CHARS =
  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

function toBase64(bytes: Uint8Array): string {
  let result = '';
  const len = bytes.length;
  for (let i = 0; i < len; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < len ? bytes[i + 1] : 0;
    const b2 = i + 2 < len ? bytes[i + 2] : 0;
    result += BASE64_CHARS[b0 >> 2];
    result += BASE64_CHARS[((b0 & 3) << 4) | (b1 >> 4)];
    result += i + 1 < len ? BASE64_CHARS[((b1 & 0xf) << 2) | (b2 >> 6)] : '=';
    result += i + 2 < len ? BASE64_CHARS[b2 & 0x3f] : '=';
  }
  return result;
}

// ─── Types ────────────────────────────────────────────────────────────────────

export type InpaintMethod = 'telea' | 'ns';

export interface InpaintConfig {
  method: InpaintMethod;       // default 'telea'
  inpaintRadius: number;       // Telea/NS neighbourhood radius, default 10
}

export interface InpaintResult {
  /** URI of the inpainted image written to temp storage */
  outputUri: string;
  method: InpaintMethod;
  processingMs: number;
}

const DEFAULT_INPAINT_CONFIG: InpaintConfig = {
  method: 'telea',
  inpaintRadius: 10,
};

// ─── OpenCV Native Bridge ─────────────────────────────────────────────────────

/**
 * OpenCVInpainter bridges to the native OpenCVProcessor TurboModule.
 *
 * Track 3: Uses NativeModules.OpenCVProcessor.processImage() with
 * method: 'inpaint_telea' or 'inpaint_ns'.
 * This consolidates all OpenCV operations into a single native module.
 *
 * C++ side (OpenCVProcessor.cpp):
 *   cv::Mat img  = cv::imread(imagePath)
 *   cv::Mat mask = decodeMask(maskBase64, img.cols, img.rows)
 *   cv::inpaint(img, mask, dst, radius, INPAINT_TELEA | INPAINT_NS)
 *   cv::imwrite(tempPath, dst)
 *   resolves Promise with "file://" + tempPath
 */
class OpenCVInpainter {
  private module = NativeModules.OpenCVProcessor;

  isAvailable(): boolean {
    return typeof this.module?.processImage === 'function';
  }

  async inpaint(
    imageUri: string,
    mask: MaskResult,
    config: InpaintConfig,
  ): Promise<string> {
    if (!this.isAvailable()) {
      throw new Error('[OpenCVInpainter] NativeModules.OpenCVProcessor not available');
    }

    const maskBase64 = this.maskToBase64(mask);
    const method     = config.method === 'ns' ? 'inpaint_ns' : 'inpaint_telea';

    const outputUri: string = await this.module.processImage(
      imageUri,
      maskBase64,
      {
        method,
        radius: config.inpaintRadius,
      },
    );

    return outputUri;
  }

  /**
   * Encode single-channel Uint8Array mask as base64.
   * Expands 1-channel mask to RGBA so decodeMask() on the C++ side
   * can extract the R channel uniformly (same encoding as PostProcessor).
   *
   * Uses the module-level toBase64() helper instead of btoa() because
   * btoa is NOT available in Hermes on React Native 0.73.x.
   */
  private maskToBase64(mask: MaskResult): string {
    const rgba = new Uint8Array(mask.width * mask.height * 4);
    for (let i = 0; i < mask.width * mask.height; i++) {
      const v = mask.data[i];
      rgba[i * 4]     = v;
      rgba[i * 4 + 1] = v;
      rgba[i * 4 + 2] = v;
      rgba[i * 4 + 3] = 255;
    }
    return toBase64(rgba);
  }
}

// ─── Unified Inpainting Facade ────────────────────────────────────────────────

export class Inpainter {
  private opencv = new OpenCVInpainter();

  /**
   * No-op: OpenCV is always available via the native module — no model to load.
   * Kept for API compatibility with Pipeline.warmUp().
   */
  async warmUp(): Promise<void> {
    // OpenCV Telea/NS are native — no pre-loading required.
    console.log('[Inpainter] Warm-up complete (OpenCV native — no model load required)');
  }

  async inpaint(
    imageUri: string,
    mask: MaskResult,
    config: Partial<InpaintConfig> = {},
  ): Promise<InpaintResult> {
    const cfg = { ...DEFAULT_INPAINT_CONFIG, ...config };
    const t0  = performance.now();

    console.log(`[Inpainter] coverage=${(mask.coverage * 100).toFixed(1)}% → method=${cfg.method}`);

    if (!this.opencv.isAvailable()) {
      throw new Error('[Inpainter] NativeModules.OpenCVProcessor is not available');
    }

    const outputUri = await this.opencv.inpaint(imageUri, mask, cfg);

    return {
      outputUri,
      method: cfg.method,
      processingMs: Math.round(performance.now() - t0),
    };
  }
}

export const inpainter = new Inpainter();
