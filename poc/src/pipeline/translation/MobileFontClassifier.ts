/**
 * src/pipeline/translation/MobileFontClassifier.ts
 * ────────────────────────────────────────────────────────────────────────────
 * Phase 3 — Stage 7 (Mobile Path): Font Family Classification
 *
 * Uses react-native-fast-tflite (already installed) to run a lightweight
 * TFLite font classifier. Falls back to SANS_SERIF if the model is not
 * loaded or inference fails.
 *
 * Per approved plan:
 *  - Coarse 5-class prediction only (family).
 *  - Weight defaults to REGULAR on mobile.
 *  - Italic detection not attempted on mobile (renders upright).
 *  - Model: font_classifier.tflite bundled in android/app/src/main/assets/
 *
 * Usage:
 *   import { classifyFont } from './MobileFontClassifier';
 *   const { fontClass, confidence } = await classifyFont(cropUri);
 */

import { useTensorflowModel } from 'react-native-fast-tflite';
import type { TensorflowModel } from 'react-native-fast-tflite';

// ── Types ─────────────────────────────────────────────────────────────────────

export type FontClass =
  | 'SANS_SERIF'
  | 'SERIF'
  | 'MONOSPACE'
  | 'DISPLAY'
  | 'HANDWRITTEN';

export type FontWeight = 'THIN' | 'REGULAR' | 'BOLD';

export interface FontClassResult {
  fontClass:  FontClass;
  confidence: number;   // 0.0 – 1.0
}

// ── Constants ─────────────────────────────────────────────────────────────────

/** Class labels in the order they appear in the TFLite model output. */
const CLASS_LABELS: FontClass[] = [
  'SANS_SERIF',
  'SERIF',
  'MONOSPACE',
  'DISPLAY',
  'HANDWRITTEN',
];

/** Input image size expected by the model (H × W × 3). */
const MODEL_INPUT_SIZE = 64;

/** Minimum confidence to trust the model prediction. Below this → SANS_SERIF. */
const MIN_CONFIDENCE = 0.55;

// ── Fallback result ───────────────────────────────────────────────────────────

const FALLBACK: FontClassResult = { fontClass: 'SANS_SERIF', confidence: 0.0 };

// ── Font Classifier ───────────────────────────────────────────────────────────

/**
 * Classifies the font family of a text region crop.
 *
 * @param model  TFLite model instance from useTensorflowModel().
 *               Pass null/undefined to skip inference and use fallback.
 * @param pixels Float32Array of shape [1, 64, 64, 3] — normalised [0,1].
 *               Pre-process with cropAndResizeToFloat32() below.
 */
export function classifyFontFromPixels(
  model:  TensorflowModel | null | undefined,
  pixels: Float32Array,
): FontClassResult {
  if (!model) return FALLBACK;

  try {
    const outputs = model.runSync([pixels]);
    const scores  = outputs[0] as Float32Array;

    if (!scores || scores.length !== CLASS_LABELS.length) {
      return FALLBACK;
    }

    // Argmax
    let maxIdx = 0;
    let maxVal = scores[0];
    for (let i = 1; i < scores.length; i++) {
      if (scores[i] > maxVal) { maxVal = scores[i]; maxIdx = i; }
    }

    if (maxVal < MIN_CONFIDENCE) {
      return { fontClass: 'SANS_SERIF', confidence: maxVal };
    }

    return { fontClass: CLASS_LABELS[maxIdx], confidence: maxVal };
  } catch {
    return FALLBACK;
  }
}

/**
 * Pre-process a raw pixel buffer (Uint8 RGBA, H×W×4) into the Float32Array
 * [1, 64, 64, 3] that the TFLite model expects.
 *
 * If the source buffer is from a different size crop, the caller must resize
 * first (e.g. via Skia makeImageFromEncoded + resize).
 */
export function cropToModelInput(
  rgbaPixels: Uint8Array,
  srcWidth:   number,
  srcHeight:  number,
): Float32Array {
  const S   = MODEL_INPUT_SIZE;
  const out = new Float32Array(S * S * 3);

  const scaleX = srcWidth  / S;
  const scaleY = srcHeight / S;

  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const srcX  = Math.min(srcWidth  - 1, Math.floor(x * scaleX));
      const srcY  = Math.min(srcHeight - 1, Math.floor(y * scaleY));
      const srcIdx = (srcY * srcWidth + srcX) * 4;
      const dstIdx = (y * S + x) * 3;
      out[dstIdx    ] = rgbaPixels[srcIdx    ] / 255.0;  // R
      out[dstIdx + 1] = rgbaPixels[srcIdx + 1] / 255.0;  // G
      out[dstIdx + 2] = rgbaPixels[srcIdx + 2] / 255.0;  // B
    }
  }

  return out;
}
