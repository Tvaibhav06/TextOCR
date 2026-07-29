/**
 * tests/unit/MobileOCR.test.ts
 * ────────────────────────────────────────────────────────────────────────────
 * Stage 5 Mobile Path — MobileOCR unit tests.
 *
 * All tests mock @react-native-ml-kit/text-recognition so no device or
 * network access is required. Tests validate:
 *   - IoU matching between ML Kit blocks and Stage 1 detection boxes
 *   - Confidence gate (skip=true when conf < 0.60 or text < 3 chars)
 *   - Empty-detection fast path
 *   - Multiple detections matched to correct blocks
 *   - Text aggregation from lines/elements within a block
 *
 * Run:
 *   npx jest tests/unit/MobileOCR.test.ts --no-coverage
 */

import { extractTextFromRegions } from '../../src/pipeline/translation/MobileOCR';
import type { DetectionResult } from '../../src/pipeline/TextDetector';

// ── Mock ML Kit ───────────────────────────────────────────────────────────────

const mockRecognize = jest.fn();

jest.mock('@react-native-ml-kit/text-recognition', () => ({
  __esModule: true,
  default: { recognize: (...args: unknown[]) => mockRecognize(...args) },
}));

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeDetection(
  id: number,
  x: number, y: number, w: number, h: number,
  normalised = true,
  imgW = 1000, imgH = 1000,
): DetectionResult {
  // If normalised=true, x/y/w/h are fractions; convert for polygon too.
  const px = normalised ? x : x / imgW;
  const py = normalised ? y : y / imgH;
  const pw = normalised ? w : w / imgW;
  const ph = normalised ? h : h / imgH;
  return {
    id,
    confidence: 0.9,
    source: 'mlkit',
    bbox: { x: px, y: py, width: pw, height: ph },
    polygon: {
      points: [
        { x: px,       y: py       },
        { x: px + pw,  y: py       },
        { x: px + pw,  y: py + ph  },
        { x: px,       y: py + ph  },
      ],
    },
  };
}

function makeMlKitBlock(
  left: number, top: number, width: number, height: number,
  text: string,
  hasLang = true,
) {
  return {
    frame: { left, top, width, height },
    recognizedLanguages: hasLang ? [{ languageCode: 'en' }] : [],
    lines: [
      {
        elements: text.split(' ').map(word => ({ text: word })),
      },
    ],
  };
}

const IMAGE_DIMS = { width: 1000, height: 1000 };

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('MobileOCR — extractTextFromRegions', () => {

  afterEach(() => {
    mockRecognize.mockReset();
  });

  // ── Empty fast path ───────────────────────────────────────────────────────

  it('returns empty array when detections array is empty', async () => {
    const result = await extractTextFromRegions('file://image.jpg', [], IMAGE_DIMS);
    expect(result).toEqual([]);
    expect(mockRecognize).not.toHaveBeenCalled();
  });

  // ── Happy path ────────────────────────────────────────────────────────────

  it('extracts text from a perfectly overlapping block', async () => {
    // Detection at (0.1, 0.1, 0.4, 0.1) in normalised coords
    // → pixel rect (100, 100, 400, 100)
    const det = makeDetection(0, 0.1, 0.1, 0.4, 0.1);
    mockRecognize.mockResolvedValue({
      blocks: [
        makeMlKitBlock(100, 100, 400, 100, 'Hello World'),
      ],
    });

    const results = await extractTextFromRegions('file://img.jpg', [det], IMAGE_DIMS);

    expect(results).toHaveLength(1);
    expect(results[0].detId).toBe(0);
    expect(results[0].text).toBe('Hello World');
    expect(results[0].confidence).toBe(1.0);
    expect(results[0].skip).toBe(false);
  });

  it('maps detId correctly to the detection id', async () => {
    const det = makeDetection(42, 0.2, 0.2, 0.3, 0.1);
    mockRecognize.mockResolvedValue({
      blocks: [makeMlKitBlock(200, 200, 300, 100, 'SALE')],
    });

    const results = await extractTextFromRegions('file://img.jpg', [det], IMAGE_DIMS);
    expect(results[0].detId).toBe(42);
  });

  it('returns pixel bbox in result', async () => {
    const det = makeDetection(0, 0.1, 0.2, 0.3, 0.05);
    mockRecognize.mockResolvedValue({
      blocks: [makeMlKitBlock(100, 200, 300, 50, 'TEXT')],
    });

    const results = await extractTextFromRegions('file://img.jpg', [det], IMAGE_DIMS);
    expect(results[0].bbox.x).toBeCloseTo(100);
    expect(results[0].bbox.y).toBeCloseTo(200);
    expect(results[0].bbox.width).toBeCloseTo(300);
    expect(results[0].bbox.height).toBeCloseTo(50);
  });

  // ── Skip conditions ───────────────────────────────────────────────────────

  it('sets skip=true when no overlapping ML Kit block is found', async () => {
    // Detection at (0.1, 0.1, 0.2, 0.1)
    // ML Kit block is far away → no overlap
    const det = makeDetection(0, 0.1, 0.1, 0.2, 0.1);
    mockRecognize.mockResolvedValue({
      blocks: [makeMlKitBlock(800, 800, 100, 50, 'DISTANT')],
    });

    const results = await extractTextFromRegions('file://img.jpg', [det], IMAGE_DIMS);
    expect(results[0].skip).toBe(true);
    expect(results[0].text).toBe('');
  });

  it('sets skip=true when text is shorter than 3 chars', async () => {
    const det = makeDetection(0, 0.1, 0.1, 0.3, 0.1);
    mockRecognize.mockResolvedValue({
      blocks: [makeMlKitBlock(100, 100, 300, 100, 'Hi')],
    });

    const results = await extractTextFromRegions('file://img.jpg', [det], IMAGE_DIMS);
    expect(results[0].skip).toBe(true);
  });

  it('sets skip=true when ML Kit returns no recognised language (low conf)', async () => {
    const det = makeDetection(0, 0.1, 0.1, 0.3, 0.1);
    mockRecognize.mockResolvedValue({
      blocks: [makeMlKitBlock(100, 100, 300, 100, 'Hello World', false)],
    });

    const results = await extractTextFromRegions('file://img.jpg', [det], IMAGE_DIMS);
    // conf=0.5 < MIN_CONFIDENCE=0.60 → skip
    expect(results[0].skip).toBe(true);
    expect(results[0].confidence).toBe(0.5);
  });

  it('sets skip=false for text of exactly 3 chars with high conf', async () => {
    const det = makeDetection(0, 0.1, 0.1, 0.3, 0.1);
    mockRecognize.mockResolvedValue({
      blocks: [makeMlKitBlock(100, 100, 300, 100, 'BUY', true)],
    });

    const results = await extractTextFromRegions('file://img.jpg', [det], IMAGE_DIMS);
    expect(results[0].skip).toBe(false);
    expect(results[0].text).toBe('BUY');
  });

  // ── Multiple detections ───────────────────────────────────────────────────

  it('handles multiple detections and matches each to the correct block', async () => {
    const det0 = makeDetection(0, 0.0, 0.0, 0.2, 0.1);
    const det1 = makeDetection(1, 0.8, 0.8, 0.2, 0.1);

    mockRecognize.mockResolvedValue({
      blocks: [
        makeMlKitBlock(0, 0, 200, 100, 'First Text'),
        makeMlKitBlock(800, 800, 200, 100, 'Second Text'),
      ],
    });

    const results = await extractTextFromRegions('file://img.jpg', [det0, det1], IMAGE_DIMS);
    expect(results).toHaveLength(2);
    expect(results[0].text).toBe('First Text');
    expect(results[1].text).toBe('Second Text');
  });

  it('handles ML Kit returning empty blocks array', async () => {
    const det = makeDetection(0, 0.1, 0.1, 0.3, 0.1);
    mockRecognize.mockResolvedValue({ blocks: [] });

    const results = await extractTextFromRegions('file://img.jpg', [det], IMAGE_DIMS);
    expect(results[0].skip).toBe(true);
    expect(results[0].text).toBe('');
  });

  it('picks the best-IoU block when multiple blocks overlap the same detection', async () => {
    const det = makeDetection(0, 0.1, 0.1, 0.4, 0.1);
    mockRecognize.mockResolvedValue({
      blocks: [
        makeMlKitBlock(50, 50, 200, 100, 'Partial'),    // lower IoU
        makeMlKitBlock(100, 100, 400, 100, 'Best Match'), // perfect overlap, higher IoU
      ],
    });

    const results = await extractTextFromRegions('file://img.jpg', [det], IMAGE_DIMS);
    expect(results[0].text).toBe('Best Match');
  });

  // ── Text aggregation ──────────────────────────────────────────────────────

  it('aggregates elements from multiple lines in a block', async () => {
    const det = makeDetection(0, 0.0, 0.0, 0.5, 0.2);
    mockRecognize.mockResolvedValue({
      blocks: [{
        frame: { left: 0, top: 0, width: 500, height: 200 },
        recognizedLanguages: [{ languageCode: 'en' }],
        lines: [
          { elements: [{ text: 'Line' }, { text: 'One' }] },
          { elements: [{ text: 'Line' }, { text: 'Two' }] },
        ],
      }],
    });

    const results = await extractTextFromRegions('file://img.jpg', [det], IMAGE_DIMS);
    expect(results[0].text).toBe('Line One Line Two');
  });
});
