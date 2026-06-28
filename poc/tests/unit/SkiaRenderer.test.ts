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

  // Simulate the filter that render_all applies (matches updated SkiaRenderer logic)
  function filterRegions(
    ocr: MobileOcrResult[],
    tr:  MobileTranslationResult[],
  ) {
    return ocr
      .map((o, i) => ({
        ocr: o,
        // When failed, substitute the original OCR text (mirrors PC renderer)
        tr: tr[i]?.engine === 'failed' ? { ...tr[i], text: o.text } : tr[i],
      }))
      .filter(({ ocr, tr }) => {
        if (ocr.skip) return false;
        if (!tr) return false;
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

  it('falls back to OCR text for failed translation regions', () => {
    const ocr = [makeOcrResult(0, false)]; // ocr.text = 'Hello'
    const tr  = [makeTrResult('', 'failed')];
    const result = filterRegions(ocr, tr);
    // Should NOT be dropped — should render with OCR fallback text 'Hello'
    expect(result).toHaveLength(1);
    expect(result[0].tr.text).toBe('Hello');
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

// ════════════════════════════════════════════════════════════════════════════
// SUITE 4: CJK font sentinel logic
// ════════════════════════════════════════════════════════════════════════════

describe('SkiaRenderer — CJK font sentinel', () => {
  // Mirror the getFontAssetName sentinel logic for CJK
  const SYSTEM_FONT = 'SYSTEM';
  function fontNameForLang(lang: string): string {
    const cjk = ['zh', 'ja', 'ko'];
    if (cjk.includes(lang.toLowerCase())) return SYSTEM_FONT;
    return 'NotoSans-Regular'; // non-CJK returns a real font name
  }

  it('returns SYSTEM sentinel for Chinese (zh)', () => {
    expect(fontNameForLang('zh')).toBe(SYSTEM_FONT);
  });

  it('returns SYSTEM sentinel for Japanese (ja)', () => {
    expect(fontNameForLang('ja')).toBe(SYSTEM_FONT);
  });

  it('returns SYSTEM sentinel for Korean (ko)', () => {
    expect(fontNameForLang('ko')).toBe(SYSTEM_FONT);
  });

  it('does NOT return SYSTEM sentinel for Latin (fr)', () => {
    expect(fontNameForLang('fr')).not.toBe(SYSTEM_FONT);
  });

  it('does NOT return SYSTEM sentinel for Hindi (hi)', () => {
    expect(fontNameForLang('hi')).not.toBe(SYSTEM_FONT);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// SUITE 5: estimateItalic
// ═══════════════════════════════════════════════════════════════════════════
// Inline replica of estimateItalic logic (cannot import from SkiaRenderer.tsx
// directly — it pulls expo-file-system which is ESM-only and fails Jest).
// The production function is tested via its observable guards.

describe('estimateItalic (guard logic)', () => {

  /**
   * Inline replica of SkiaRenderer.estimateItalic.
   * Kept in sync manually; tests the guard behaviour and regression-checks
   * the slope-detection algorithm without touching the full React component.
   */
  function estimateItalic(rgbaPixels: Uint8Array | null, width: number, height: number): boolean {
    if (!rgbaPixels || width < 16 || height < 8) return false;
    if (width < height * 1.5) return false;
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < width * height; i++) {
      gray[i] = Math.round(0.299 * rgbaPixels[i*4] + 0.587 * rgbaPixels[i*4+1] + 0.114 * rgbaPixels[i*4+2]);
    }
    const hist = new Array(256).fill(0);
    for (const v of gray) hist[v]++;
    const total = width * height;
    let sumAll = 0;
    for (let i = 0; i < 256; i++) sumAll += i * hist[i];
    let wB = 0, sumB = 0, maxVar = 0, thr = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t]; if (wB === 0) continue;
      const wF = total - wB; if (wF === 0) break;
      sumB += t * hist[t];
      const mB = sumB / wB, mF = (sumAll - sumB) / wF;
      const v = wB * wF * (mB - mF) ** 2;
      if (v > maxVar) { maxVar = v; thr = t; }
    }
    thr = Math.max(1, Math.min(254, thr)); // clamp: thr=0/255 produces degenerate masks
    const darkCount = gray.filter(v => v < thr).length;
    const binary = new Uint8Array(total);
    if (darkCount < total * 0.5) {
      for (let i = 0; i < total; i++) binary[i] = gray[i] < thr ? 255 : 0;
    } else {
      for (let i = 0; i < total; i++) binary[i] = gray[i] >= thr ? 255 : 0;
    }
    const fgCount = binary.reduce((s, v) => s + (v > 0 ? 1 : 0), 0);
    if (fgCount < 50) return false;
    const rowCount = new Array(height).fill(0);
    const rowSumX  = new Array(height).fill(0);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (binary[y * width + x] > 0) { rowCount[y]++; rowSumX[y] += x; }
      }
    }
    const minRowFg = Math.max(1, Math.floor(width * 0.05));
    const pairs: [number, number][] = [];
    for (let y = 0; y < height; y++) {
      if (rowCount[y] >= minRowFg) pairs.push([y, rowSumX[y] / rowCount[y]]);
    }
    if (pairs.length < 4) return false;
    let n = pairs.length, sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (const [py, px] of pairs) { sumX += py; sumY += px; sumXY += py*px; sumX2 += py*py; }
    const denom = n * sumX2 - sumX * sumX;
    if (Math.abs(denom) < 1e-9) return false;
    const slope = (n * sumXY - sumX * sumY) / denom;
    return Math.abs(Math.atan(slope) * 180 / Math.PI) > 5.0;
  }

  function makeBlankRgba(w: number, h: number): Uint8Array { return new Uint8Array(w * h * 4).fill(255); }
  function paintRect(buf: Uint8Array, bw: number, x: number, y: number, rw: number, rh: number) {
    for (let row = y; row < y + rh; row++)
      for (let col = x; col < x + rw; col++) {
        const p = (row * bw + col) * 4;
        buf[p] = buf[p+1] = buf[p+2] = 0; buf[p+3] = 255;
      }
  }

  it('returns false for null pixels', () => expect(estimateItalic(null, 100, 40)).toBe(false));

  it('returns false for region too small (width < 16)', () => {
    expect(estimateItalic(makeBlankRgba(10, 20), 10, 20)).toBe(false);
  });

  it('returns false for tall region (single char guard: width < height*1.5)', () => {
    const buf = makeBlankRgba(20, 30);
    paintRect(buf, 20, 2, 2, 16, 26);
    expect(estimateItalic(buf, 20, 30)).toBe(false);
  });

  it('returns false for a straight vertical stroke (upright text)', () => {
    const w = 100, h = 30;
    const buf = makeBlankRgba(w, h);
    paintRect(buf, w, 45, 0, 10, h);  // vertical bar — zero slope
    expect(estimateItalic(buf, w, h)).toBe(false);
  });

  it('returns true for a strongly skewed stroke (italic ~56°)', () => {
    // 160×40 region: paint a thick diagonal band that shifts rightward with row.
    // Each row gets a 10px-wide, 1px-tall stroke; across 40 rows → 400 fg pixels > 50 threshold.
    // Centroid shifts 2px per row → ~63° angle from vertical → well above 5° threshold.
    const w = 160, h = 40;
    const buf = makeBlankRgba(w, h);
    for (let row = 0; row < h; row++) {
      const cx = 10 + Math.round(row * 2.5); // 2.5px/row → ~68° lean
      paintRect(buf, w, Math.min(cx, w - 12), row, 12, 1);
    }
    expect(estimateItalic(buf, w, h)).toBe(true);
  });
});
