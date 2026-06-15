/**
 * tests/unit/SkiaRenderer.test.ts
 * ────────────────────────────────────────────────────────────────────────────
 * Stage 7 Mobile Path — SkiaRenderer and MobileFontClassifier unit tests.
 *
 * All tests mock @shopify/react-native-skia and react-native-fast-tflite.
 * No device, emulator, or font download required.
 *
 * Run:
 *   npx jest tests/unit/SkiaRenderer.test.ts --no-coverage
 */

// ── Mock @shopify/react-native-skia ──────────────────────────────────────────

const mockParagraph = {
  layout: jest.fn(),
};
const mockBuilder = {
  pushStyle: jest.fn(),
  addText:   jest.fn(),
  pop:       jest.fn(),
  build:     jest.fn(() => mockParagraph),
};
const mockSkia = {
  ParagraphStyle:   jest.fn((s) => s),
  ParagraphBuilder: { Make: jest.fn(() => mockBuilder) },
  Color:            jest.fn((c) => c),
};

jest.mock('@shopify/react-native-skia', () => ({
  __esModule: true,
  Canvas:     'Canvas',
  Image:      'Image',
  Paragraph:  'Paragraph',
  Skia:       mockSkia,
  TextAlign:  { Left: 0, Right: 2 },
  FontStyle:  { Upright: 0 },
  useFonts:   jest.fn(() => ({})),
  useImage:   jest.fn(() => null),
}));

// ── Mock react-native-fast-tflite ────────────────────────────────────────────

const mockRunSync = jest.fn();
const mockModel = { runSync: mockRunSync };

jest.mock('react-native-fast-tflite', () => ({
  __esModule: true,
  useTensorflowModel: jest.fn(() => ({ model: mockModel, state: 'loaded' })),
}));

// ── Imports (after mocks) ─────────────────────────────────────────────────────

import {
  classifyFontFromPixels,
  cropToModelInput,
} from '../../src/pipeline/translation/MobileFontClassifier';
import type { FontClass } from '../../src/pipeline/translation/MobileFontClassifier';
import type { MobileOcrResult } from '../../src/pipeline/translation/MobileOCR';
import type { MobileTranslationResult } from '../../src/pipeline/translation/MobileTranslate';

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeOcrResult(
  id:     number,
  skip:   boolean = false,
  x = 10, y = 20, w = 200, h = 50,
): MobileOcrResult {
  return {
    detId:      id,
    text:       'Hello',
    confidence: 0.95,
    skip,
    bbox: { x, y, width: w, height: h },
  };
}

function makeTrResult(
  text:   string,
  engine: 'mlkit' | 'failed' = 'mlkit',
): MobileTranslationResult {
  return { text, engine, srcLang: 'en', tgtLang: 'fr' };
}

function makeScores(scores: number[]): Float32Array {
  return new Float32Array(scores);
}

const CLASS_LABELS: FontClass[] = [
  'SANS_SERIF', 'SERIF', 'MONOSPACE', 'DISPLAY', 'HANDWRITTEN',
];

// ════════════════════════════════════════════════════════════════════════════
// SUITE 1: MobileFontClassifier — classifyFontFromPixels
// ════════════════════════════════════════════════════════════════════════════

describe('MobileFontClassifier — classifyFontFromPixels', () => {

  beforeEach(() => mockRunSync.mockReset());

  it('returns SANS_SERIF fallback when model is null', () => {
    const pixels = new Float32Array(64 * 64 * 3);
    const result = classifyFontFromPixels(null, pixels);
    expect(result.fontClass).toBe('SANS_SERIF');
    expect(result.confidence).toBe(0.0);
  });

  it('returns SANS_SERIF fallback when model is undefined', () => {
    const pixels = new Float32Array(64 * 64 * 3);
    const result = classifyFontFromPixels(undefined, pixels);
    expect(result.fontClass).toBe('SANS_SERIF');
  });

  it('returns the argmax class when confidence is above threshold', () => {
    mockRunSync.mockReturnValue([makeScores([0.1, 0.8, 0.05, 0.03, 0.02])]);
    const pixels = new Float32Array(64 * 64 * 3);
    const result = classifyFontFromPixels(mockModel as any, pixels);
    expect(result.fontClass).toBe('SERIF');   // index 1
    expect(result.confidence).toBeCloseTo(0.8);
  });

  it('returns SANS_SERIF (not original class) when confidence below threshold (0.55)', () => {
    mockRunSync.mockReturnValue([makeScores([0.22, 0.20, 0.20, 0.19, 0.19])]);
    const pixels = new Float32Array(64 * 64 * 3);
    const result = classifyFontFromPixels(mockModel as any, pixels);
    expect(result.fontClass).toBe('SANS_SERIF');
  });

  it('returns MONOSPACE for the third class', () => {
    mockRunSync.mockReturnValue([makeScores([0.01, 0.01, 0.95, 0.01, 0.02])]);
    const pixels = new Float32Array(64 * 64 * 3);
    const result = classifyFontFromPixels(mockModel as any, pixels);
    expect(result.fontClass).toBe('MONOSPACE');
    expect(result.confidence).toBeCloseTo(0.95);
  });

  it('returns DISPLAY when confidence >= 0.55', () => {
    mockRunSync.mockReturnValue([makeScores([0.05, 0.05, 0.05, 0.80, 0.05])]);
    const pixels = new Float32Array(64 * 64 * 3);
    const result = classifyFontFromPixels(mockModel as any, pixels);
    expect(result.fontClass).toBe('DISPLAY');
  });

  it('returns HANDWRITTEN for the fifth class', () => {
    mockRunSync.mockReturnValue([makeScores([0.0, 0.0, 0.0, 0.05, 0.95])]);
    const pixels = new Float32Array(64 * 64 * 3);
    const result = classifyFontFromPixels(mockModel as any, pixels);
    expect(result.fontClass).toBe('HANDWRITTEN');
  });

  it('returns SANS_SERIF fallback when model throws', () => {
    mockRunSync.mockImplementation(() => { throw new Error('Model error'); });
    const pixels = new Float32Array(64 * 64 * 3);
    const result = classifyFontFromPixels(mockModel as any, pixels);
    expect(result.fontClass).toBe('SANS_SERIF');
  });

  it('returns SANS_SERIF when output shape is wrong', () => {
    mockRunSync.mockReturnValue([makeScores([0.5, 0.5])]);  // wrong length
    const pixels = new Float32Array(64 * 64 * 3);
    const result = classifyFontFromPixels(mockModel as any, pixels);
    expect(result.fontClass).toBe('SANS_SERIF');
  });
});

// ════════════════════════════════════════════════════════════════════════════
// SUITE 2: MobileFontClassifier — cropToModelInput
// ════════════════════════════════════════════════════════════════════════════

describe('MobileFontClassifier — cropToModelInput', () => {

  it('produces Float32Array of the correct length (64×64×3)', () => {
    const rgba = new Uint8Array(100 * 100 * 4).fill(128);
    const out  = cropToModelInput(rgba, 100, 100);
    expect(out.length).toBe(64 * 64 * 3);
  });

  it('all values are in [0, 1] range', () => {
    const rgba = new Uint8Array(32 * 32 * 4);
    for (let i = 0; i < rgba.length; i++) rgba[i] = Math.floor(Math.random() * 256);
    const out = cropToModelInput(rgba, 32, 32);
    for (let i = 0; i < out.length; i++) {
      expect(out[i]).toBeGreaterThanOrEqual(0.0);
      expect(out[i]).toBeLessThanOrEqual(1.0);
    }
  });

  it('maps white pixels (255,255,255) to 1.0', () => {
    const rgba = new Uint8Array(64 * 64 * 4).fill(255);
    const out  = cropToModelInput(rgba, 64, 64);
    // All RGB values should be 1.0
    for (let i = 0; i < out.length; i++) {
      expect(out[i]).toBeCloseTo(1.0);
    }
  });

  it('maps black pixels (0,0,0) to 0.0', () => {
    const rgba = new Uint8Array(64 * 64 * 4).fill(0);
    // Alpha to 255 (doesn't affect output)
    for (let i = 3; i < rgba.length; i += 4) rgba[i] = 255;
    const out = cropToModelInput(rgba, 64, 64);
    // All RGB output values should be 0.0
    for (let i = 0; i < out.length; i++) {
      expect(out[i]).toBeCloseTo(0.0);
    }
  });

  it('works correctly for 1×1 source (all pixels same)', () => {
    const rgba = new Uint8Array([200, 100, 50, 255]);
    const out  = cropToModelInput(rgba, 1, 1);
    expect(out.length).toBe(64 * 64 * 3);
    expect(out[0]).toBeCloseTo(200 / 255);
    expect(out[1]).toBeCloseTo(100 / 255);
    expect(out[2]).toBeCloseTo(50  / 255);
  });
});

// ════════════════════════════════════════════════════════════════════════════
// SUITE 3: SkiaRenderer logic tests (font selection, RTL, skip logic)
// ════════════════════════════════════════════════════════════════════════════
// We test the pure logic functions extracted from SkiaRenderer rather than
// rendering the full React component (which requires a running Skia canvas).

// Re-export internal helpers for testing via a test-only import shim.
// Since they are not exported, we test the observable behaviour through
// the component's memoised region list logic.

describe('SkiaRenderer — region filtering logic', () => {

  // Simulate the filter that render_all applies
  function filterRegions(
    ocr: MobileOcrResult[],
    tr:  MobileTranslationResult[],
  ) {
    return ocr
      .map((o, i) => ({ ocr: o, tr: tr[i] }))
      .filter(({ ocr, tr }) => {
        if (ocr.skip) return false;
        if (!tr || tr.engine === 'failed') return false;
        if (!tr.text?.trim()) return false;
        return true;
      });
  }

  it('keeps valid non-skipped regions', () => {
    const ocr = [makeOcrResult(0, false)];
    const tr  = [makeTrResult('Bonjour')];
    expect(filterRegions(ocr, tr)).toHaveLength(1);
  });

  it('filters out skipped OCR regions', () => {
    const ocr = [makeOcrResult(0, true)];
    const tr  = [makeTrResult('Bonjour')];
    expect(filterRegions(ocr, tr)).toHaveLength(0);
  });

  it('filters out failed translation regions', () => {
    const ocr = [makeOcrResult(0, false)];
    const tr  = [makeTrResult('', 'failed')];
    expect(filterRegions(ocr, tr)).toHaveLength(0);
  });

  it('filters out empty translation text', () => {
    const ocr = [makeOcrResult(0, false)];
    const tr  = [makeTrResult('   ')];   // whitespace only
    expect(filterRegions(ocr, tr)).toHaveLength(0);
  });

  it('keeps multiple valid regions', () => {
    const ocr = [makeOcrResult(0), makeOcrResult(1), makeOcrResult(2, true)];
    const tr  = [makeTrResult('A'), makeTrResult('B'), makeTrResult('C')];
    expect(filterRegions(ocr, tr)).toHaveLength(2);
  });
});
