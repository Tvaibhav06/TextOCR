/**
 * Stage 1 — Text Detection
 *
 * Strategy (in priority order):
 *  1. Google ML Kit  — zero model download, 50+ scripts, ~150–300 ms @ 720p
 *  2. TFLite EAST    — 6 MB model, polygon-level boxes, fallback if ML Kit misses
 *  3. MediaPipe      — experimental, future upgrade path
 *
 * Returns an array of DetectionResult with bounding polygon + confidence.
 */

import { NativeModules, Platform } from 'react-native';
import type { TensorflowModel } from 'react-native-fast-tflite';
import { loadTensorflowModel } from 'react-native-fast-tflite';

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
  source: 'mlkit' | 'east' | 'mediapipe' | 'ensemble';
}

export interface DetectorConfig {
  confidenceThreshold: number;   // default 0.50
  enableEastFallback: boolean;   // default true
  nmsIouThreshold: number;       // default 0.50
  maxDetections: number;         // default 200
}

const DEFAULT_CONFIG: DetectorConfig = {
  confidenceThreshold: 0.50,
  enableEastFallback: true,
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
  private module = NativeModules.MLKitTextDetection;

  async isAvailable(): Promise<boolean> {
    try {
      return await this.module.isModelAvailable();
    } catch {
      return false;
    }
  }

  async detect(
    imageUri: string,
    imageWidth: number,
    imageHeight: number,
    threshold: number,
  ): Promise<DetectionResult[]> {
    // Native call — returns raw JSON array from ML Kit TextRecognition API
    const raw: Array<{
      boundingBox: { left: number; top: number; right: number; bottom: number };
      confidence: number;
      cornerPoints?: Array<{ x: number; y: number }>;
    }> = await this.module.detectText(imageUri, threshold);

    return raw.map((r, idx) => {
      // Prefer cornerPoints (polygon) if present; fall back to bbox rect
      const pts = r.cornerPoints && r.cornerPoints.length === 4
        ? r.cornerPoints.map(p => ({
            x: p.x / imageWidth,
            y: p.y / imageHeight,
          }))
        : [
            { x: r.boundingBox.left  / imageWidth, y: r.boundingBox.top    / imageHeight },
            { x: r.boundingBox.right / imageWidth, y: r.boundingBox.top    / imageHeight },
            { x: r.boundingBox.right / imageWidth, y: r.boundingBox.bottom / imageHeight },
            { x: r.boundingBox.left  / imageWidth, y: r.boundingBox.bottom / imageHeight },
          ];

      return {
        id: idx,
        polygon: { points: pts },
        bbox: {
          x: r.boundingBox.left / imageWidth,
          y: r.boundingBox.top  / imageHeight,
          width:  (r.boundingBox.right  - r.boundingBox.left) / imageWidth,
          height: (r.boundingBox.bottom - r.boundingBox.top)  / imageHeight,
        },
        confidence: r.confidence ?? 0.9,  // ML Kit doesn't always return confidence
        source: 'mlkit' as const,
      };
    });
  }
}

// ─── TFLite EAST Detector ─────────────────────────────────────────────────────

/**
 * EASTDetector loads a quantised INT8 EAST model via react-native-fast-tflite.
 * Model: east_text_detector_int8.tflite (~6 MB)
 * Input: 320×320 float32 RGB, normalised [0,1]
 * Output:
 *   [0]: score_map  — [1, 80, 80, 1]  — text probability per cell
 *   [1]: geo_map    — [1, 80, 80, 5]  — (dx1,dy1,dx2,dy2,angle) per cell
 */
class EASTDetector {
  private model: TensorflowModel | null = null;
  private readonly MODEL_INPUT_SIZE = 320;
  private readonly SCORE_THRESHOLD = 0.8;
  private readonly NMS_THRESHOLD   = 0.2;

  async load(): Promise<void> {
    this.model = await loadTensorflowModel(
      require('../../assets/models/east_text_detector_int8.tflite'),
    );
  }

  isLoaded(): boolean {
    return this.model !== null;
  }

  async detect(
    pixelData: Float32Array,   // 320×320×3 pre-normalised
    origWidth: number,
    origHeight: number,
    threshold: number,
  ): Promise<DetectionResult[]> {
    if (!this.model) throw new Error('EAST model not loaded');

    const [scores, geometry] = await this.model.run([pixelData]) as [Float32Array, Float32Array];

    const scaleX = origWidth  / this.MODEL_INPUT_SIZE;
    const scaleY = origHeight / this.MODEL_INPUT_SIZE;
    const cellSize = this.MODEL_INPUT_SIZE / 80;   // 80×80 output grid

    const candidates: DetectionResult[] = [];
    let id = 0;

    for (let row = 0; row < 80; row++) {
      for (let col = 0; col < 80; col++) {
        const scoreIdx = row * 80 + col;
        const score = scores[scoreIdx];
        if (score < Math.max(threshold, this.SCORE_THRESHOLD)) continue;

        // Decode geometry: each cell encodes distances to 4 sides + rotation angle
        const geoBase = scoreIdx * 5;
        const top   = geometry[geoBase];
        const right = geometry[geoBase + 1];
        const bot   = geometry[geoBase + 2];
        const left  = geometry[geoBase + 3];
        const angle = geometry[geoBase + 4];

        // Offset vector from cell anchor
        const ox = (col + 0.5) * cellSize;
        const oy = (row + 0.5) * cellSize;

        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);

        // Rotated rectangle corners (pixel space at MODEL_INPUT_SIZE)
        const x1 = ox - left * cosA - top * sinA;
        const y1 = oy + left * sinA - top * cosA;
        const x2 = x1 + right * cosA + bot * sinA;
        const y2 = y1 - right * sinA + bot * cosA;

        // Normalise to [0,1] using original image dimensions
        const nx1 = (x1 * scaleX) / origWidth;
        const ny1 = (y1 * scaleY) / origHeight;
        const nx2 = (x2 * scaleX) / origWidth;
        const ny2 = (y2 * scaleY) / origHeight;

        candidates.push({
          id: id++,
          polygon: {
            points: [
              { x: nx1, y: ny1 },
              { x: nx2, y: ny1 },
              { x: nx2, y: ny2 },
              { x: nx1, y: ny2 },
            ],
          },
          bbox: {
            x: nx1,
            y: ny1,
            width:  nx2 - nx1,
            height: ny2 - ny1,
          },
          confidence: score,
          source: 'east' as const,
        });
      }
    }

    return this.nms(candidates, this.NMS_THRESHOLD);
  }

  /** Non-Maximum Suppression — removes overlapping candidates */
  private nms(dets: DetectionResult[], iouThreshold: number): DetectionResult[] {
    const sorted = [...dets].sort((a, b) => b.confidence - a.confidence);
    const kept: DetectionResult[] = [];

    for (const det of sorted) {
      const overlap = kept.some(k => this.iou(det.bbox, k.bbox) > iouThreshold);
      if (!overlap) kept.push(det);
    }
    return kept;
  }

  private iou(
    a: { x: number; y: number; width: number; height: number },
    b: { x: number; y: number; width: number; height: number },
  ): number {
    const ix1 = Math.max(a.x, b.x);
    const iy1 = Math.max(a.y, b.y);
    const ix2 = Math.min(a.x + a.width, b.x + b.width);
    const iy2 = Math.min(a.y + a.height, b.y + b.height);
    if (ix2 <= ix1 || iy2 <= iy1) return 0;
    const inter = (ix2 - ix1) * (iy2 - iy1);
    const union  = a.width * a.height + b.width * b.height - inter;
    return union <= 0 ? 0 : inter / union;
  }
}

// ─── Ensemble Detector ────────────────────────────────────────────────────────

/**
 * EnsembleDetector runs ML Kit first (always), then optionally merges EAST
 * results when enableEastFallback is true. Duplicate regions are removed via IoU NMS.
 */
export class EnsembleDetector {
  private mlkit = new MLKitTextDetector();
  private east  = new EASTDetector();
  private eastLoaded = false;

  async warmUp(config: Partial<DetectorConfig> = {}): Promise<void> {
    const cfg = { ...DEFAULT_CONFIG, ...config };
    if (cfg.enableEastFallback) {
      try {
        await this.east.load();
        this.eastLoaded = true;
        console.log('[TextDetector] EAST model loaded');
      } catch (err) {
        console.warn('[TextDetector] EAST load failed, skipping:', err);
      }
    }
  }

  async detect(
    imageUri: string,
    imageWidth: number,
    imageHeight: number,
    config: Partial<DetectorConfig> = {},
  ): Promise<DetectionResult[]> {
    const cfg = { ...DEFAULT_CONFIG, ...config };

    // ── Primary: ML Kit ──
    let results: DetectionResult[] = [];
    try {
      const mlkitResults = await this.mlkit.detect(
        imageUri, imageWidth, imageHeight, cfg.confidenceThreshold,
      );
      results = mlkitResults;
      console.log(`[TextDetector] ML Kit: ${results.length} detections`);
    } catch (err) {
      console.warn('[TextDetector] ML Kit failed:', err);
    }

    // ── Fallback / supplement: EAST ──
    if (this.eastLoaded && cfg.enableEastFallback) {
      try {
        // ⚠️ POC STUB — EAST receives a zero-filled tensor here.
        //
        // A zero tensor causes EAST to produce zero detections (score map all below
        // threshold), so the EAST path is currently a no-op. This is intentional
        // for the POC: the ML Kit primary path alone is exercised end-to-end.
        //
        // What a real implementation needs (Q1 decision: implement in native sprint):
        //   1. Read image pixels → expo-file-system / Skia readPixels → Uint8Array RGBA
        //   2. Resize to 320×320 via a JS bicubic or Skia resize
        //   3. Normalise to float32 [0, 1]: tensor[i] = rgba[i*4+ch] / 255.0
        //   4. Pass the resulting Float32Array(320*320*3) to east.detect()
        const eastPixels = new Float32Array(320 * 320 * 3); // stub: replace with real pixels
        const eastResults = await this.east.detect(
          eastPixels,
          imageWidth,
          imageHeight,
          cfg.confidenceThreshold,
        );
        console.log(`[TextDetector] EAST: ${eastResults.length} additional candidates`);
        results = this.mergeWithNMS([...results, ...eastResults], cfg.nmsIouThreshold);
      } catch (err) {
        console.warn('[TextDetector] EAST detection failed:', err);
      }
    }

    // Filter by confidence + cap at maxDetections
    return results
      .filter(r => r.confidence >= cfg.confidenceThreshold)
      .slice(0, cfg.maxDetections)
      .map((r, i) => ({ ...r, id: i }));
  }

  private mergeWithNMS(dets: DetectionResult[], iouThreshold: number): DetectionResult[] {
    const sorted = [...dets].sort((a, b) => b.confidence - a.confidence);
    const kept: DetectionResult[] = [];
    for (const det of sorted) {
      const iouFn = (
        a: { x: number; y: number; width: number; height: number },
        b: { x: number; y: number; width: number; height: number },
      ) => {
        const ix1 = Math.max(a.x, b.x), iy1 = Math.max(a.y, b.y);
        const ix2 = Math.min(a.x + a.width, b.x + b.width);
        const iy2 = Math.min(a.y + a.height, b.y + b.height);
        if (ix2 <= ix1 || iy2 <= iy1) return 0;
        const inter = (ix2 - ix1) * (iy2 - iy1);
        const union  = a.width * a.height + b.width * b.height - inter;
        return union <= 0 ? 0 : inter / union;
      };
      if (!kept.some(k => iouFn(det.bbox, k.bbox) > iouThreshold)) kept.push(det);
    }
    return kept;
  }
}

export const textDetector = new EnsembleDetector();
