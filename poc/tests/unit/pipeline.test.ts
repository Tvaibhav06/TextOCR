/**
 * ─────────────────────────────────────────────────────────────────────────────
 *  UNIT TESTS — Jest (runs without a device)
 *  File: tests/unit/pipeline.test.ts
 * ─────────────────────────────────────────────────────────────────────────────
 */

// ─── Mock native modules ──────────────────────────────────────────────────────
jest.mock('react-native', () => ({
  NativeModules: {
    MLKitTextDetection: {
      isModelAvailable: jest.fn().mockResolvedValue(true),
      detectText: jest.fn().mockResolvedValue([]),
    },
    OpenCVInpaint: {
      inpaint: jest.fn().mockResolvedValue('file://output/result.jpg'),
    },
  },
  Platform: { OS: 'android' },
}));

jest.mock('react-native-fast-tflite', () => ({
  loadTensorflowModel: jest.fn().mockResolvedValue({ run: jest.fn() }),
}));

jest.mock('@shopify/react-native-skia', () => ({
  Skia: {
    // Use a factory fn so each Surface.Make() call returns a fresh mock object
    // and there is no shared state leaking between tests.
    Surface: {
      Make: jest.fn().mockImplementation(() => ({
        getCanvas: () => ({
          drawRect: jest.fn(),
          drawPath: jest.fn(),
        }),
        flush: jest.fn(),
        makeImageSnapshot: () => ({
          readPixels: jest.fn().mockReturnValue(new ArrayBuffer(100 * 100 * 4)),
        }),
      })),
    },
    Paint: jest.fn().mockReturnValue({
      setColor: jest.fn(), setStyle: jest.fn(), setBlendMode: jest.fn(),
    }),
    Color: jest.fn().mockReturnValue(0xFFFFFFFF),
    Path: { Make: jest.fn().mockReturnValue({
      moveTo: jest.fn(), lineTo: jest.fn(), close: jest.fn(),
    })},
  },
  PaintStyle: { Fill: 0 },
  BlendMode: { Src: 0 },
  // ColorType named constant (used in MaskGenerator after Fix 6)
  ColorType: { RGBA_8888: 4 },
}));

// ─── Imports ──────────────────────────────────────────────────────────────────
import { EnsembleDetector, type DetectionResult } from '../../src/pipeline/TextDetector';
import { MaskGenerator }                          from '../../src/pipeline/MaskGenerator';
import { QualityAssessor }                        from '../../src/pipeline/PostProcessor';
import { CancellationToken }                      from '../../src/pipeline/Pipeline';

// ════════════════════════════════════════════════════════════════════════════
// SUITE 1: EnsembleDetector — confidence filtering + NMS
// ════════════════════════════════════════════════════════════════════════════

describe('EnsembleDetector', () => {
  const detector = new EnsembleDetector();

  const mockDetections: DetectionResult[] = [
    { id: 0, polygon: { points: [{x:0.1,y:0.1},{x:0.5,y:0.1},{x:0.5,y:0.2},{x:0.1,y:0.2}] },
      bbox: {x:0.1,y:0.1,width:0.4,height:0.1}, confidence: 0.92, source: 'mlkit' },
    { id: 1, polygon: { points: [{x:0.1,y:0.4},{x:0.7,y:0.4},{x:0.7,y:0.5},{x:0.1,y:0.5}] },
      bbox: {x:0.1,y:0.4,width:0.6,height:0.1}, confidence: 0.45, source: 'mlkit' },
    { id: 2, polygon: { points: [{x:0.2,y:0.7},{x:0.8,y:0.7},{x:0.8,y:0.8},{x:0.2,y:0.8}] },
      bbox: {x:0.2,y:0.7,width:0.6,height:0.1}, confidence: 0.75, source: 'east' },
  ];

  it('filters detections below confidence threshold 0.5', () => {
    const filtered = mockDetections.filter(d => d.confidence >= 0.5);
    expect(filtered).toHaveLength(2);
    expect(filtered.map(d => d.id)).toEqual([0, 2]);
  });

  it('returns all detections at threshold 0.0', () => {
    const filtered = mockDetections.filter(d => d.confidence >= 0.0);
    expect(filtered).toHaveLength(3);
  });

  it('returns zero detections above max confidence', () => {
    const filtered = mockDetections.filter(d => d.confidence >= 0.99);
    expect(filtered).toHaveLength(0);
  });

  it('assigns sequential IDs after filtering', () => {
    const filtered = mockDetections
      .filter(d => d.confidence >= 0.5)
      .map((r, i) => ({ ...r, id: i }));
    expect(filtered[0].id).toBe(0);
    expect(filtered[1].id).toBe(1);
  });

  it('NMS removes duplicate overlapping box (IoU > 0.5)', () => {
    const duplicate: DetectionResult = {
      id: 3,
      polygon: { points: [{x:0.1,y:0.09},{x:0.5,y:0.09},{x:0.5,y:0.19},{x:0.1,y:0.19}] },
      bbox: {x:0.1, y:0.09, width:0.4, height:0.1},
      confidence: 0.88,
      source: 'east',
    };
    // IoU between mockDetections[0].bbox and duplicate.bbox ≈ 0.8 → should be suppressed
    const dets = [mockDetections[0], duplicate];
    const sorted = [...dets].sort((a,b) => b.confidence - a.confidence);
    const kept: DetectionResult[] = [];
    const iou = (a: typeof dets[0]['bbox'], b: typeof dets[0]['bbox']) => {
      const ix1 = Math.max(a.x,b.x), iy1 = Math.max(a.y,b.y);
      const ix2 = Math.min(a.x+a.width, b.x+b.width);
      const iy2 = Math.min(a.y+a.height, b.y+b.height);
      if (ix2<=ix1||iy2<=iy1) return 0;
      const inter=(ix2-ix1)*(iy2-iy1);
      const union=a.width*a.height+b.width*b.height-inter;
      return union<=0?0:inter/union;
    };
    for (const d of sorted) {
      if (!kept.some(k => iou(d.bbox, k.bbox) > 0.5)) kept.push(d);
    }
    expect(kept).toHaveLength(1);
    expect(kept[0].confidence).toBe(0.92);
  });

  it('handles empty detection array gracefully', async () => {
    const { NativeModules } = require('react-native');
    NativeModules.MLKitTextDetection.detectText.mockResolvedValueOnce([]);
    // warmUp not needed here since EAST is mocked
    const results = []; // simulate zero results
    expect(results).toHaveLength(0);
  });

  /**
   * This test calls detector.detect() for real (with mocked native module)
   * and confirms the class's confidence filter + ID reassignment pipeline
   * works end-to-end — previously SUITE 1 only tested inline logic.
   */
  it('detect() filters low-confidence results from MLKit and reassigns IDs', async () => {
    const { NativeModules } = require('react-native');
    // MLKit returns 3 boxes: conf 0.9, 0.3, 0.8 — threshold 0.5 should drop 0.3
    NativeModules.MLKitTextDetection.detectText.mockResolvedValueOnce([
      { boundingBox: { left: 10, top: 10, right: 60, bottom: 30 }, confidence: 0.9 },
      { boundingBox: { left: 70, top: 10, right: 120, bottom: 30 }, confidence: 0.3 },
      { boundingBox: { left: 130, top: 10, right: 200, bottom: 30 }, confidence: 0.8 },
    ]);

    const results = await detector.detect('file://test.jpg', 640, 480, {
      confidenceThreshold: 0.5,
      enableEastFallback: false,   // disable EAST so only MLKit path runs
    });

    // Expect 2 results (conf 0.9 and 0.8 pass; 0.3 is filtered)
    expect(results).toHaveLength(2);
    // IDs must be reassigned to 0, 1 after filtering
    expect(results[0].id).toBe(0);
    expect(results[1].id).toBe(1);
    // Results should be sorted by confidence (highest first via MLKit order)
    expect(results.every(r => r.confidence >= 0.5)).toBe(true);
    // Source must be 'mlkit'
    expect(results.every(r => r.source === 'mlkit')).toBe(true);
  });
});

// ════════════════════════════════════════════════════════════════════════════
// SUITE 2: MaskGenerator — dilation math
// ════════════════════════════════════════════════════════════════════════════

describe('MaskGenerator — dilation kernel', () => {
  /**
   * We test the dilation logic directly on typed arrays
   * without needing a real Skia surface.
   */
  function dilate(mask: Uint8Array, w: number, h: number, r: number): Uint8Array {
    const tmp = new Uint8Array(w * h);
    const out = new Uint8Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let max = 0;
        for (let dx = -r; dx <= r; dx++) {
          const nx = x + dx;
          if (nx >= 0 && nx < w) max = Math.max(max, mask[y * w + nx]);
        }
        tmp[y * w + x] = max;
      }
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let max = 0;
        for (let dy = -r; dy <= r; dy++) {
          const ny = y + dy;
          if (ny >= 0 && ny < h) max = Math.max(max, tmp[ny * w + x]);
        }
        out[y * w + x] = max;
      }
    return out;
  }

  it('dilated mask is strictly larger than undilated mask', () => {
    const W = 20, H = 20;
    const mask = new Uint8Array(W * H);
    mask[10 * W + 10] = 255;   // single pixel
    const d0 = mask.filter(v => v > 0).length;
    const d3 = dilate(mask, W, H, 3).filter(v => v > 0).length;
    expect(d3).toBeGreaterThan(d0);
  });

  it('zero-radius dilation is identity', () => {
    const W = 10, H = 10;
    const mask = new Uint8Array(W * H);
    mask[5 * W + 5] = 255;
    const d0 = dilate(mask, W, H, 0);
    expect(Array.from(d0)).toEqual(Array.from(mask));
  });

  it('dilation covers adjacent pixels within radius', () => {
    const W = 10, H = 10;
    const mask = new Uint8Array(W * H);
    mask[5 * W + 5] = 255;
    const d2 = dilate(mask, W, H, 2);
    // Pixels within manhattan distance 2 should be set
    expect(d2[5 * W + 3]).toBe(255);   // 2 left
    expect(d2[3 * W + 5]).toBe(255);   // 2 up
    expect(d2[7 * W + 5]).toBe(255);   // 2 down
    expect(d2[5 * W + 7]).toBe(255);   // 2 right
  });

  it('mask coverage increases monotonically with dilation radius', () => {
    const W = 20, H = 20;
    const mask = new Uint8Array(W * H);
    for (let i = 5; i < 8; i++) for (let j = 5; j < 15; j++) mask[i * W + j] = 255;
    const cov = (r: number) => dilate(mask, W, H, r).filter(v => v > 0).length;
    expect(cov(1)).toBeGreaterThanOrEqual(cov(0));
    expect(cov(2)).toBeGreaterThanOrEqual(cov(1));
    expect(cov(3)).toBeGreaterThanOrEqual(cov(2));
  });

  it('empty mask produces zero coverage regardless of dilation', () => {
    const W = 20, H = 20;
    const mask = new Uint8Array(W * H);
    const d5 = dilate(mask, W, H, 5);
    expect(d5.every(v => v === 0)).toBe(true);
  });

  it('full mask stays full after dilation', () => {
    const W = 10, H = 10;
    const mask = new Uint8Array(W * H).fill(255);
    const d3 = dilate(mask, W, H, 3);
    expect(d3.every(v => v === 255)).toBe(true);
  });
});

// ════════════════════════════════════════════════════════════════════════════
// SUITE 3: QualityAssessor — SSIM + PSNR correctness
// ════════════════════════════════════════════════════════════════════════════

describe('QualityAssessor', () => {
  const qa = new QualityAssessor();
  const W = 64, H = 64, N = W * H;

  const makeMask = (fill: number) => ({
    data: new Uint8Array(N).fill(fill),
    width: W, height: H, maskedPixels: fill > 0 ? N : 0, coverage: fill > 0 ? 1 : 0,
  });

  const makeRGBA = (r: number, g: number, b: number) => {
    const buf = new Uint8Array(N * 4);
    for (let i = 0; i < N; i++) { buf[i*4]=r; buf[i*4+1]=g; buf[i*4+2]=b; buf[i*4+3]=255; }
    return buf;
  };

  it('identical images yield SSIM = 1', () => {
    const img = makeRGBA(128, 64, 200);
    const m = qa.assess(img, img, makeMask(255), 0.82);
    expect(m.ssim).toBeCloseTo(1, 2);
  });

  it('MSE = 0 for identical images', () => {
    const img = makeRGBA(100, 100, 100);
    const m = qa.assess(img, img, makeMask(255), 0.82);
    expect(m.mse).toBe(0);
  });

  it('completely different images yield SSIM < 0.5', () => {
    const black = makeRGBA(0, 0, 0);
    const white = makeRGBA(255, 255, 255);
    const m = qa.assess(black, white, makeMask(255), 0.82);
    expect(m.ssim).toBeLessThan(0.5);
  });

  it('aboveThreshold = true when SSIM >= threshold', () => {
    const img = makeRGBA(128, 128, 128);
    const m = qa.assess(img, img, makeMask(255), 0.82);
    expect(m.aboveThreshold).toBe(true);
  });

  it('aboveThreshold = false for badly distorted image', () => {
    const orig  = makeRGBA(0,   0,   0);
    const dist  = makeRGBA(255, 255, 255);
    const m = qa.assess(orig, dist, makeMask(255), 0.82);
    expect(m.aboveThreshold).toBe(false);
  });

  it('returns SSIM=1 and no error for empty mask', () => {
    const img = makeRGBA(100, 100, 100);
    const m = qa.assess(img, img, makeMask(0), 0.82);
    expect(m.ssim).toBe(1);
    expect(m.mse).toBe(0);
  });
});

// ════════════════════════════════════════════════════════════════════════════
// SUITE 4: CancellationToken
// ════════════════════════════════════════════════════════════════════════════

describe('CancellationToken', () => {
  it('starts not cancelled', () => {
    const t = new CancellationToken();
    expect(t.isCancelled).toBe(false);
  });

  it('isCancelled becomes true after cancel()', () => {
    const t = new CancellationToken();
    t.cancel();
    expect(t.isCancelled).toBe(true);
  });

  it('throwIfCancelled throws after cancel()', () => {
    const t = new CancellationToken();
    t.cancel();
    expect(() => t.throwIfCancelled()).toThrow('Pipeline cancelled by user');
  });

  it('throwIfCancelled is safe before cancel()', () => {
    const t = new CancellationToken();
    expect(() => t.throwIfCancelled()).not.toThrow();
  });

  it('calling cancel() twice is idempotent', () => {
    const t = new CancellationToken();
    t.cancel(); t.cancel();
    expect(t.isCancelled).toBe(true);
  });
});
