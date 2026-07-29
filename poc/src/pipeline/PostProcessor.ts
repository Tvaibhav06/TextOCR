/**
 * Stage 4 — Post-Processing
 *
 * Two operations:
 *
 *  A. Native Edge Blending  (OpenCV C++ via JSI — NativeModules.OpenCVProcessor)
 *     Smooths hard colour discontinuities at the mask boundary.
 *     Replaces the old expo-gl / WebGL Jacobi Poisson solver.
 *
 *     On the native side (Android JNI / iOS Obj-C++):
 *       cv::seamlessClone()  or  cv::inpaint() second-pass edge feather
 *     Returns the blended image URI written to temp storage.
 *
 *     For the Track 2 POC the native module returns the input URI unchanged
 *     (stub). Wired to real cv::seamlessClone in Track 3.
 *
 *  B. Quality Assessment  (pure TypeScript — no native dependency)
 *     Computes SSIM and PSNR on the masked region to decide
 *     whether to offer the user a "request higher quality" button
 *     that triggers the server-side LaMa pipeline.
 *
 *     SSIM(x,y) = (2μₓμᵧ+C₁)(2σₓᵧ+C₂) / ((μₓ²+μᵧ²+C₁)(σₓ²+σᵧ²+C₂))
 *     PSNR      = 10·log₁₀(255²/MSE)
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

export interface PostProcessConfig {
  enableBlend: boolean;       // default true — run native edge blend
  qualityThreshold: number;   // SSIM below this → prompt server upgrade, default 0.82
}

export interface QualityMetrics {
  ssim:      number;
  psnr:      number;
  mse:       number;
  aboveThreshold: boolean;
}

export interface PostProcessResult {
  outputUri:    string;
  quality:      QualityMetrics;
  blendApplied: boolean;
  processingMs: number;
}

const DEFAULT_POST_CONFIG: PostProcessConfig = {
  enableBlend: true,
  qualityThreshold: 0.82,
};

// ─── Native Image Processor Bridge ───────────────────────────────────────────

/**
 * NativeImageProcessor bridges to the OpenCVProcessor TurboModule (C++ JSI).
 *
 * Native module interface (Android: JNI / iOS: Obj-C++):
 *   processImage(imageUri, maskBase64, { method, radius }) → Promise<string>
 *     - method: 'blend' | 'dilate' | 'inpaint_telea' | 'inpaint_ns'
 *     - Returns output file URI (written to app temp dir)
 *
 * For Track 2 (stub): the native module returns the input URI unchanged.
 * Real OpenCV operations will be wired in Track 3 (pipeline wiring sprint).
 *
 * Docs:
 *   Android: src/native/OpenCVProcessor.cpp + CMakeLists.txt
 *   iOS:     src/native/ios/OpenCVProcessor.mm
 */
export class NativeImageProcessor {
  /** NativeModules.OpenCVProcessor — registered by the C++ TurboModule. */
  private module = NativeModules.OpenCVProcessor;

  /** True if the native module is installed and callable. */
  isAvailable(): boolean {
    return typeof this.module?.processImage === 'function';
  }

  /**
   * Edge-blend the inpainted image against the original at mask boundaries.
   *
   * Native path: cv::seamlessClone (NORMAL_CLONE mode) over the mask region.
   * JS fallback: returns inpaintedUri unchanged (no-op blend).
   *
   * @param originalUri  - Local file URI of the original image
   * @param inpaintedUri - Local file URI of the inpainted image
   * @param maskBase64   - Base64-encoded single-channel PNG mask (255=text region)
   * @returns            - Local file URI of the blended output
   */
  async blend(
    originalUri: string,
    inpaintedUri: string,
    maskBase64: string,
  ): Promise<string> {
    if (!this.isAvailable()) {
      console.warn('[NativeImageProcessor] OpenCVProcessor not available — skipping blend');
      return inpaintedUri;
    }

    try {
      const outputUri: string = await this.module.processImage(
        inpaintedUri,
        maskBase64,
        {
          method: 'blend',
          originalUri,
        },
      );
      return outputUri;
    } catch (err) {
      console.warn('[NativeImageProcessor] blend failed, returning inpainted URI:', err);
      return inpaintedUri;
    }
  }

  /**
   * Encode single-channel mask as base64 for native bridge.
   * Uses the module-level toBase64() helper — btoa() is NOT available
   * in Hermes on React Native 0.73.x.
   */
  maskToBase64(mask: MaskResult): string {
    // Expand 1-channel mask to RGBA for PNG encoding compatibility
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

// ─── Quality Metrics (pure JS) ────────────────────────────────────────────────

export class QualityAssessor {
  /**
   * Compute SSIM and PSNR between original and result
   * only over the masked region to avoid rewarding unchanged pixels.
   *
   * SSIM uses 11×11 Gaussian window; simplified to global stats for mobile perf.
   */
  assess(
    original: Uint8Array,
    result:   Uint8Array,
    mask:     MaskResult,
    threshold: number,
  ): QualityMetrics {
    const N = mask.width * mask.height;

    let sumA = 0, sumB = 0, sumA2 = 0, sumB2 = 0, sumAB = 0, mse = 0;
    let count = 0;

    for (let i = 0; i < N; i++) {
      if (mask.data[i] === 0) continue; // only assess masked region
      count++;

      // Use luminance: Y = 0.299R + 0.587G + 0.114B
      const lumA = (original[i*4] * 0.299 + original[i*4+1] * 0.587 + original[i*4+2] * 0.114) / 255;
      const lumB = (result[i*4]   * 0.299 + result[i*4+1]   * 0.587 + result[i*4+2]   * 0.114) / 255;

      sumA  += lumA;  sumB  += lumB;
      sumA2 += lumA * lumA;  sumB2 += lumB * lumB;
      sumAB += lumA * lumB;
      mse   += (lumA - lumB) ** 2;
    }

    if (count === 0) return { ssim: 1, psnr: Infinity, mse: 0, aboveThreshold: true };

    const muA   = sumA  / count;
    const muB   = sumB  / count;
    const sigA2 = sumA2 / count - muA * muA;
    const sigB2 = sumB2 / count - muB * muB;
    const sigAB = sumAB / count - muA * muB;
    const mseV  = mse / count;

    // SSIM constants: C1=(0.01)², C2=(0.03)²
    const C1 = 0.0001, C2 = 0.0009;
    const ssim = ((2 * muA * muB + C1) * (2 * sigAB + C2))
               / ((muA**2 + muB**2 + C1) * (sigA2 + sigB2 + C2));

    const psnr = mseV === 0 ? Infinity : 10 * Math.log10(1 / mseV);

    return {
      ssim:  Math.max(0, Math.min(1, ssim)),
      psnr:  Math.round(psnr * 100) / 100,
      mse:   Math.round(mseV * 1e6) / 1e6,
      aboveThreshold: ssim >= threshold,
    };
  }
}

// ─── Singletons ───────────────────────────────────────────────────────────────

/**
 * @deprecated PoissonBlender removed — WebGL expo-gl dependency eliminated.
 *             Use nativeImageProcessor.blend() instead.
 */
export const poissonBlender = {
  blend: () => {
    throw new Error(
      '[PostProcessor] PoissonBlender removed. Use nativeImageProcessor.blend() via NativeModules.OpenCVProcessor.',
    );
  },
  setGL: () => {},
};

export const nativeImageProcessor = new NativeImageProcessor();
export const qualityAssessor       = new QualityAssessor();

// ─── Pixel Extraction (Skia) ──────────────────────────────────────────────────

import { Skia, ColorType, AlphaType, PaintStyle } from '@shopify/react-native-skia';

/**
 * Extracts raw RGBA pixels from a local image URI using an offscreen Skia surface.
 * This solves the POC limitation and allows real SSIM math.
 */
export async function getImagePixels(uri: string, width: number, height: number): Promise<Uint8Array | null> {
  try {
    const data = await Skia.Data.fromURI(uri);
    const img = Skia.Image.MakeImageFromEncoded(data);
    if (!img) {
      console.warn('[PostProcessor] Skia failed to decode image data');
      return null;
    }

    const surface = Skia.Surface.Make(width, height);
    if (!surface) {
      console.warn('[PostProcessor] Skia failed to create offscreen surface');
      return null;
    }

    const canvas = surface.getCanvas();
    const paint = Skia.Paint();
    paint.setStyle(PaintStyle.Fill);

    // Draw the image scaled to the target dimensions
    canvas.drawImageRect(
      img,
      { x: 0, y: 0, width: img.width(), height: img.height() },
      { x: 0, y: 0, width, height },
      paint
    );
    surface.flush();

    const snapshot = surface.makeImageSnapshot();
    const pixels = snapshot.readPixels(0, 0, {
      width,
      height,
      colorType: ColorType.RGBA_8888,
      alphaType: 1, // Opaque
    });

    return pixels ? new Uint8Array(pixels) : null;
  } catch (err) {
    console.warn('[PostProcessor] getImagePixels failed:', err);
    return null;
  }
}

