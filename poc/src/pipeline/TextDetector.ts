/**
 * Stage 1 — Text Detection
 *
 * Strategy:
 *  1. Google ML Kit  — zero model download, 50+ scripts, ~150–300 ms @ 720p
 *
 * Future upgrade paths (not yet implemented):
 *  - TFLite EAST  — requires a real INT8 model + real pixel extraction (see M-1 in
 *    verification_report). The EAST path was intentionally NOT added to this file
 *    because the only available model file is a 6-byte placeholder stub. When a real
 *    INT8 EAST model is bundled, also implement readAndResizeImage() to replace the
 *    zero-tensor — providing a zero-filled input to a real model produces wrong results.
 *    (F-5: keep this comment until M-1 is fully resolved.)
 *  - MediaPipe    — experimental
 *
 * Returns an array of DetectionResult with bounding polygon + confidence.
 */

import TextRecognition from '@react-native-ml-kit/text-recognition';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface BoundingPolygon {
  /** Clockwise ordered corner points, normalised [0,1] relative to image dims */
  points: Array<{ x: number; y: number }>;
}

export interface DetectionResult {
  id: number;
  polygon: BoundingPolygon;
  /** Bounding box derived from polygon — convenience field */
  bbox: { x: number; y: number; width: number; height: number };
  confidence: number;
  /** Which detector produced this result */
  source: 'mlkit' | 'mediapipe' | 'ensemble';
}

export interface DetectorConfig {
  confidenceThreshold: number;   // default 0.50
  nmsIouThreshold: number;       // default 0.50
  maxDetections: number;         // default 200
}

const DEFAULT_CONFIG: DetectorConfig = {
  confidenceThreshold: 0.50,
  nmsIouThreshold: 0.50,
  maxDetections: 200,
};

// ─── ML Kit Bridge ────────────────────────────────────────────────────────────

/**
 * MLKitTextDetector wraps the native ML Kit on-device text recognition.
 * The native module returns bounding boxes as { x, y, width, height } in
 * pixel coords; we convert to normalised polygon format here.
 */
class MLKitTextDetector {
  async isAvailable(): Promise<boolean> {
    return true; // The library is always available natively
  }

  async detect(
    imageUri: string,
    imageWidth: number,
    imageHeight: number,
    threshold: number,
  ): Promise<DetectionResult[]> {
    const result = await TextRecognition.recognize(imageUri);

    return result.blocks.map((b: any, idx: number) => {
      // Use cornerPoints if available, else derive from frame
      const pts = b.cornerPoints && b.cornerPoints.length === 4
        ? b.cornerPoints.map((p: any) => ({
            x: p.x / imageWidth,
            y: p.y / imageHeight,
          }))
        : [
            { x: b.frame.left / imageWidth, y: b.frame.top / imageHeight },
            { x: (b.frame.left + b.frame.width) / imageWidth, y: b.frame.top / imageHeight },
            { x: (b.frame.left + b.frame.width) / imageWidth, y: (b.frame.top + b.frame.height) / imageHeight },
            { x: b.frame.left / imageWidth, y: (b.frame.top + b.frame.height) / imageHeight },
          ];

      return {
        id: idx,
        polygon: { points: pts },
        bbox: {
          x: b.frame.left / imageWidth,
          y: b.frame.top / imageHeight,
          width: b.frame.width / imageWidth,
          height: b.frame.height / imageHeight,
        },
        confidence: 0.9, // TextRecognition doesn't always expose confidence per block
        source: 'mlkit' as const,
      };
    });
  }
}

// ─── Text Detector ────────────────────────────────────────────────────────────

/**
 * TextDetector runs ML Kit for on-device text detection.
 *
 * The warmUp() method is intentionally a no-op — ML Kit requires no model
 * pre-loading. It is kept for API compatibility with Pipeline.warmUp().
 */
export class TextDetector {
  private mlkit = new MLKitTextDetector();

  /** No-op: ML Kit needs no pre-loading. Kept for Pipeline.warmUp() compatibility. */
  async warmUp(_config: Partial<DetectorConfig> = {}): Promise<void> {
    // ML Kit is always available natively — nothing to warm up.
    console.log('[TextDetector] Warm-up complete (ML Kit — no model load required)');
  }

  async detect(
    imageUri: string,
    imageWidth: number,
    imageHeight: number,
    config: Partial<DetectorConfig> = {},
  ): Promise<DetectionResult[]> {
    const cfg = { ...DEFAULT_CONFIG, ...config };

    // ── ML Kit detection ──
    let results: DetectionResult[] = [];
    try {
      results = await this.mlkit.detect(
        imageUri, imageWidth, imageHeight, cfg.confidenceThreshold,
      );
      console.log(`[TextDetector] ML Kit: ${results.length} detections`);
    } catch (err) {
      console.warn('[TextDetector] ML Kit failed:', err);
    }

    // Filter by confidence + cap at maxDetections
    return results
      .filter(r => r.confidence >= cfg.confidenceThreshold)
      .slice(0, cfg.maxDetections)
      .map((r, i) => ({ ...r, id: i }));
  }
}

// Keep the legacy alias so EnsembleDetector references in tests continue to compile.
/** @deprecated Use TextDetector directly. Alias kept for test compatibility. */
export { TextDetector as EnsembleDetector };

export const textDetector = new TextDetector();
