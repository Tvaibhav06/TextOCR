/**
 * tests/unit/MobileTranslate.test.ts
 * ────────────────────────────────────────────────────────────────────────────
 * Stage 6 Mobile Path — MobileTranslate unit tests.
 *
 * All tests mock the native module (NativeModules.RNMLKitTranslateText) so
 * no device, emulator, or network access is needed. Tests validate:
 *   - Happy-path translation
 *   - Same-language passthrough
 *   - Empty text fast path
 *   - Download progress callback fires correctly
 *   - Download deduplication (concurrent calls don't double-download)
 *   - Model caching (isModelDownloaded=true skips re-download)
 *   - Failure handling (native error → engine='failed', original text preserved)
 *   - Batch translateAll skips flagged regions
 *   - prefetchModels triggers download for each language
 *
 * Run:
 *   npx jest tests/unit/MobileTranslate.test.ts --no-coverage
 */

import { NativeModules } from 'react-native';

// ── Set up native module mock BEFORE importing the module under test ──────────
// The module reads NativeModules at call time (via getNative()), so we can
// provide a mock object and replace methods per-test.

jest.mock('react-native-mlkit-translate-text', () => {
  const mockNative = {
    downloadModel:      jest.fn(),
    isModelDownloaded:  jest.fn(),
    translate:          jest.fn(),
    identifyLanguage:   jest.fn(),
  };
  return {
    __esModule: true,
    MLKitTranslator: mockNative,
    default: { MLKitTranslator: mockNative },
    __mockNative: mockNative, // export it so tests can access it
  };
});

// Import AFTER mock is installed
import { MLKitTranslator } from 'react-native-mlkit-translate-text';
const mockNative = (MLKitTranslator as any).__mockNative || MLKitTranslator;


// Import AFTER mock is installed
import { mobileTranslator } from '../../src/pipeline/translation/MobileTranslate';
import type { MobileOcrResult } from '../../src/pipeline/translation/MobileOCR';

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeOcrResult(
  id: number,
  text: string,
  skip = false,
  conf = 0.95,
): MobileOcrResult {
  return {
    detId:      id,
    text,
    confidence: conf,
    skip,
    bbox: { x: 0, y: 0, width: 100, height: 30 },
  };
}

// Reset state before each test
beforeEach(() => {
  jest.clearAllMocks();
  mobileTranslator._resetCacheForTest();
  // Default: model not downloaded, download succeeds, translate succeeds
  mockNative.isModelDownloaded.mockResolvedValue(false);
  mockNative.downloadModel.mockResolvedValue(undefined);
  mockNative.translate.mockResolvedValue('Bonjour');
  mockNative.identifyLanguage.mockResolvedValue('en');
});

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('MobileTranslate — translate()', () => {

  // ── Happy path ────────────────────────────────────────────────────────────

  it('translates text and returns engine="mlkit"', async () => {
    mockNative.translate.mockResolvedValue('Bonjour');
    const result = await mobileTranslator.translate('Hello', 'en', 'fr');
    expect(result.engine).toBe('mlkit');
    expect(result.text).toBe('Bonjour');
    expect(result.srcLang).toBe('en');
    expect(result.tgtLang).toBe('fr');
    expect(result.error).toBeUndefined();
  });

  it('calls native translate with correct arguments', async () => {
    await mobileTranslator.translate('Good morning', 'en', 'hi');
    expect(mockNative.translate).toHaveBeenCalledWith('Good morning', 'ENGLISH', 'HINDI');
  });

  // ── Fast paths ────────────────────────────────────────────────────────────

  it('returns empty text without calling native when input is empty', async () => {
    const result = await mobileTranslator.translate('', 'en', 'fr');
    expect(result.text).toBe('');
    expect(mockNative.translate).not.toHaveBeenCalled();
  });

  it('returns original text when source and target language are the same', async () => {
    const result = await mobileTranslator.translate('Hello', 'en', 'en');
    expect(result.text).toBe('Hello');
    expect(result.engine).toBe('mlkit');
    expect(mockNative.translate).not.toHaveBeenCalled();
  });

  it('trims whitespace from input before translating', async () => {
    await mobileTranslator.translate('  Hello  ', 'en', 'fr');
    expect(mockNative.translate).toHaveBeenCalledWith('Hello', 'ENGLISH', 'FRENCH');
  });

  // ── Download behaviour ────────────────────────────────────────────────────

  it('downloads src and tgt models before translating', async () => {
    await mobileTranslator.translate('Hello', 'en', 'fr');
    expect(mockNative.downloadModel).toHaveBeenCalledWith('ENGLISH');
    expect(mockNative.downloadModel).toHaveBeenCalledWith('FRENCH');
  });

  it('fires progress callback at 0% start and 100% done', async () => {
    const events: number[] = [];
    const onProgress = jest.fn(p => events.push(p.progress));
    await mobileTranslator.translate('Hello', 'en', 'fr', onProgress);
    expect(events).toContain(0.0);
    expect(events).toContain(1.0);
  });

  it('does not re-download a model that is already cached in-session', async () => {
    // First call downloads
    await mobileTranslator.translate('Hello', 'en', 'fr');
    expect(mockNative.downloadModel).toHaveBeenCalledTimes(2); // en + fr

    jest.clearAllMocks();
    mockNative.translate.mockResolvedValue('Bonjour encore');

    // Second call — same languages — should not re-download
    await mobileTranslator.translate('Hello again', 'en', 'fr');
    expect(mockNative.downloadModel).not.toHaveBeenCalled();
  });

  it('skips download when isModelDownloaded returns true', async () => {
    mockNative.isModelDownloaded.mockResolvedValue(true);
    await mobileTranslator.translate('Hello', 'en', 'fr');
    expect(mockNative.downloadModel).not.toHaveBeenCalled();
  });

  it('does not fire progress callback when model is already cached', async () => {
    mockNative.isModelDownloaded.mockResolvedValue(true);
    const onProgress = jest.fn();
    await mobileTranslator.translate('Hello', 'en', 'fr', onProgress);
    expect(onProgress).not.toHaveBeenCalled();
  });

  it('deduplicates concurrent downloads for the same language', async () => {
    // Create a deferred promise that we control externally
    let resolve!: () => void;
    const gate = new Promise<void>(r => { resolve = r; });
    mockNative.downloadModel.mockImplementation(() => gate);

    // Fire two translate calls concurrently for the same language pair
    const p1 = mobileTranslator.translate('Hello', 'en', 'fr');
    const p2 = mobileTranslator.translate('World', 'en', 'fr');

    // Let event loop run so both calls reach the download step
    await Promise.resolve();

    resolve(); // unblock the shared download
    await Promise.all([p1, p2]);

    // Should only have called downloadModel once for 'ENGLISH' (then cached)
    const enCalls = mockNative.downloadModel.mock.calls.filter(c => c[0] === 'ENGLISH');
    expect(enCalls.length).toBe(1);
  });

  // ── Error handling ────────────────────────────────────────────────────────

  it('returns engine="failed" and preserves original text when translate throws', async () => {
    mockNative.translate.mockRejectedValue(new Error('Native crash'));
    const result = await mobileTranslator.translate('Hello', 'en', 'fr');
    expect(result.engine).toBe('failed');
    expect(result.text).toBe('Hello');
    expect(result.error).toBe('Native crash');
  });

  it('returns engine="failed" when download fails', async () => {
    mockNative.downloadModel.mockRejectedValue(new Error('No internet'));
    const result = await mobileTranslator.translate('Hello', 'en', 'fr');
    expect(result.engine).toBe('failed');
    expect(result.text).toBe('Hello');
  });

  it('trims whitespace from native translate output', async () => {
    mockNative.translate.mockResolvedValue('  Bonjour  ');
    const result = await mobileTranslator.translate('Hello', 'en', 'fr');
    expect(result.text).toBe('Bonjour');
  });
});


describe('MobileTranslate — translateAll()', () => {

  it('translates all non-skipped regions', async () => {
    mockNative.translate
      .mockResolvedValueOnce('नमस्ते')
      .mockResolvedValueOnce('दुनिया');

    const ocr = [
      makeOcrResult(0, 'Hello', false),
      makeOcrResult(1, 'World', false),
    ];
    const results = await mobileTranslator.translateAll(ocr, 'en', 'hi');
    expect(results).toHaveLength(2);
    expect(results[0].text).toBe('नमस्ते');
    expect(results[1].text).toBe('दुनिया');
  });

  it('passes through skipped regions without translating', async () => {
    const ocr = [
      makeOcrResult(0, 'Hello', false),
      makeOcrResult(1, '',      true),   // skipped region
    ];
    mockNative.translate.mockResolvedValue('नमस्ते');

    const results = await mobileTranslator.translateAll(ocr, 'en', 'hi');
    expect(results[1].text).toBe('');
    // translate should only be called once (for region 0)
    expect(mockNative.translate).toHaveBeenCalledTimes(1);
  });

  it('fires progress callback during model download phase', async () => {
    const events: boolean[] = [];
    const onProgress = jest.fn(p => events.push(p.done));
    const ocr = [makeOcrResult(0, 'Hello', false)];
    await mobileTranslator.translateAll(ocr, 'en', 'fr', onProgress);
    expect(onProgress).toHaveBeenCalled();
  });

  it('returns empty array when ocr results is empty', async () => {
    const results = await mobileTranslator.translateAll([], 'en', 'hi');
    expect(results).toEqual([]);
    expect(mockNative.translate).not.toHaveBeenCalled();
  });

  it('preserves original text for failed translations within a batch', async () => {
    mockNative.translate
      .mockResolvedValueOnce('नमस्ते')
      .mockRejectedValueOnce(new Error('Region 1 failed'));

    const ocr = [
      makeOcrResult(0, 'Hello', false),
      makeOcrResult(1, 'World', false),
    ];
    const results = await mobileTranslator.translateAll(ocr, 'en', 'hi');
    expect(results[0].text).toBe('नमस्ते');
    expect(results[1].engine).toBe('failed');
    expect(results[1].text).toBe('World');  // original preserved
  });
});


describe('MobileTranslate — identifyLanguage()', () => {

  it('returns language code from native module', async () => {
    mockNative.identifyLanguage.mockResolvedValue('hi');
    const lang = await mobileTranslator.identifyLanguage('नमस्ते');
    expect(lang).toBe('hi');
  });

  it('returns "und" on native error', async () => {
    mockNative.identifyLanguage.mockRejectedValue(new Error('fail'));
    const lang = await mobileTranslator.identifyLanguage('???');
    expect(lang).toBe('und');
  });
});


describe('MobileTranslate — prefetchModels()', () => {

  it('downloads each language in the list', async () => {
    await mobileTranslator.prefetchModels(['hi', 'fr', 'de']);
    const codes = mockNative.downloadModel.mock.calls.map(c => c[0]);
    expect(codes).toContain('HINDI');
    expect(codes).toContain('FRENCH');
    expect(codes).toContain('GERMAN');
  });

  it('skips already-cached languages', async () => {
    mockNative.isModelDownloaded.mockResolvedValue(true);
    await mobileTranslator.prefetchModels(['hi', 'fr']);
    expect(mockNative.downloadModel).not.toHaveBeenCalled();
  });
});
