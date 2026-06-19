/**
 * src/pipeline/translation/MobileTranslate.ts
 * ────────────────────────────────────────────────────────────────────────────
 * Phase 3 — Stage 6 (Mobile Path): On-Device Translation
 *
 * Uses react-native-mlkit-translate-text (wrapper around Google ML Kit
 * Translation API) for fully offline, on-device translation.
 *
 * Key behaviours:
 *  - Language model packs are downloaded once and cached permanently.
 *  - Progress callbacks fire during the initial download of each new language.
 *  - After first download, translation is instant with no progress events.
 *  - A single concurrent download guard prevents duplicate downloads.
 *  - If ML Kit translation fails, the result is flagged engine='failed' and
 *    the original text is preserved — Stage 7 will skip that region.
 *  - This module is ADDITIVE. It does NOT change any Stage 1–4 code.
 *
 * Supported language codes:
 *  ML Kit supports ~50 languages. Codes are BCP-47 (e.g. "en", "hi", "fr").
 *  Full list: https://developers.google.com/ml-kit/language/translation/translation-language-support
 *
 * Usage:
 *   import { mobileTranslator } from './translation/MobileTranslate';
 *   const result = await mobileTranslator.translate('Hello', 'en', 'hi', onProgress);
 */

import { MLKitTranslator } from 'react-native-mlkit-translate-text';
import type { MobileOcrResult } from './MobileOCR';

// MLKit wrapper uses uppercase strings for languages, so we map BCP-47 codes to them.
// Only languages actually supported by ML Kit Translation are listed here.
// If a language is NOT in this map, canTranslate() returns false and the
// TranslationRouter automatically falls back to the PC server (IndicTrans2 + Argos).
//
// Notable absences and why:
//   ml (Malayalam)  — NOT supported by ML Kit Translation. Falls back to PC (IndicTrans2).
//   Malay is ms (not ml); added correctly below.
const BCP_TO_MLKIT: Record<string, any> = {
  'en': 'ENGLISH',
  'hi': 'HINDI',
  'mr': 'MARATHI',
  'bn': 'BENGALI',
  'ta': 'TAMIL',
  'te': 'TELUGU',
  'kn': 'KANNADA',
  // 'ml' intentionally removed — Malayalam is NOT a ML Kit Translation language.
  //      It was incorrectly mapped to MALAY before. PC fallback handles it.
  'ms': 'MALAY',   // Bahasa Malaysia / Malay (BCP-47: ms)
  'gu': 'GUJARATI',
  'ar': 'ARABIC',
  'ur': 'URDU',
  'zh': 'CHINESE',
  'ja': 'JAPANESE',
  'ko': 'KOREAN',
  'fr': 'FRENCH',
  'de': 'GERMAN',
  'es': 'SPANISH',
  'pt': 'PORTUGUESE',
  'it': 'ITALIAN',
  'ru': 'RUSSIAN',
  'th': 'THAI',
  'vi': 'VIETNAMESE',
  'tr': 'TURKISH',
};

// ── Language picker list ──────────────────────────────────────────────────────
// This is the SOURCE OF TRUTH for the UI language picker.
// It includes ALL languages the system supports — whether via ML Kit (on-device)
// or via the PC server (IndicTrans2 + Argos). Languages not in BCP_TO_MLKIT
// will automatically fall back to the PC path via TranslationRouter.canTranslate().
export const AVAILABLE_LANGUAGES: { code: string; name: string; pcOnly?: boolean }[] = [
  // ── On-device ML Kit languages ──────────────────────────────────────────
  ...Object.keys(BCP_TO_MLKIT).map(code => ({
    code,
    name: BCP_TO_MLKIT[code].charAt(0) + BCP_TO_MLKIT[code].slice(1).toLowerCase(),
  })),
  // ── PC-server-only languages (not supported by ML Kit Translation) ───────
  // These are valid selections; TranslationRouter will skip mobile and use PC.
  { code: 'ml', name: 'Malayalam', pcOnly: true },  // IndicTrans2 on PC server
  { code: 'pa', name: 'Punjabi',   pcOnly: true },  // Gurmukhi — IndicTrans2
  { code: 'or', name: 'Odia',      pcOnly: true },  // IndicTrans2
  { code: 'as', name: 'Assamese',  pcOnly: true },  // IndicTrans2
];

function getMlkitLang(bcp47: string): any {
  return BCP_TO_MLKIT[bcp47.toLowerCase()] || 'ENGLISH';
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DownloadProgress {
  language:   string;
  /** 0.0 – 1.0 */
  progress:   number;
  done:       boolean;
}

export interface MobileTranslationResult {
  text:     string;
  engine:   'mlkit' | 'failed';
  srcLang:  string;
  tgtLang:  string;
  error?:   string;
}

export type ProgressCallback = (p: DownloadProgress) => void;

// ── Constants ─────────────────────────────────────────────────────────────────

/** Languages whose packs have been confirmed downloaded this session. */
const _downloadedCache = new Set<string>();

/** Active download promises — prevents parallel downloads of the same lang. */
const _pendingDownloads = new Map<string, Promise<void>>();



// ── Download helpers ──────────────────────────────────────────────────────────

/**
 * Ensure the ML Kit model for `lang` is downloaded.
 * - If already cached this session → returns immediately.
 * - If currently downloading → waits for the in-progress download.
 * - Otherwise → starts download and fires onProgress callbacks.
 *
 * Progress is synthetic: ML Kit's native layer doesn't expose byte-level
 * progress, so we fire 0% at start and 100% on completion.
 */
async function ensureModelDownloaded(
  lang: string,
  onProgress?: ProgressCallback,
): Promise<void> {
  if (_downloadedCache.has(lang)) return;

  // Check native cache (survives app restarts)
  try {
    const already = await MLKitTranslator.isModelDownloaded(getMlkitLang(lang));
    if (already) {
      _downloadedCache.add(lang);
      return;
    }
  } catch {
    // If check fails, attempt download anyway
  }

  // Deduplicate concurrent downloads for the same language
  if (_pendingDownloads.has(lang)) {
    await _pendingDownloads.get(lang);
    return;
  }

  const downloadPromise = (async () => {
    onProgress?.({ language: lang, progress: 0.0, done: false });
    if (!MLKitTranslator || typeof MLKitTranslator.downloadModel !== 'function') {
      console.warn('MLKit native module is not linked or unavailable. Skipping mobile translation.');
      throw new Error('MLKit native module is not available');
    }
    console.log(`[MLKit] Attempting to download model for: ${getMlkitLang(lang)} (${lang})`);
    await MLKitTranslator.downloadModel(getMlkitLang(lang));
    console.log(`[MLKit] Download SUCCESS for: ${getMlkitLang(lang)}`);
    _downloadedCache.add(lang);
    onProgress?.({ language: lang, progress: 1.0, done: true });
  })();

  _pendingDownloads.set(lang, downloadPromise);
  try {
    await downloadPromise;
  } finally {
    _pendingDownloads.delete(lang);
  }
}

// ── Translator class ──────────────────────────────────────────────────────────

class MobileTranslator {
  /**
   * Translate a single string using on-device ML Kit.
   *
   * @param text        Text to translate (from Stage 5 OCR).
   * @param src         Source language BCP-47 code (e.g. "en").
   * @param tgt         Target language BCP-47 code (e.g. "hi").
   * @param onProgress  Optional callback for model download progress.
   *                    Only fires on the FIRST use of a new target language.
   */
  async translate(
    text:       string,
    src:        string,
    tgt:        string,
    onProgress?: ProgressCallback,
  ): Promise<MobileTranslationResult> {
    const trimmed = text.trim();

    // Fast paths
    if (!trimmed) {
      return { text: '', engine: 'mlkit', srcLang: src, tgtLang: tgt };
    }
    if (src.toLowerCase() === tgt.toLowerCase()) {
      return { text: trimmed, engine: 'mlkit', srcLang: src, tgtLang: tgt };
    }

    try {
      // Ensure both language packs are present (source needed for verify, tgt for output)
      await ensureModelDownloaded(src, onProgress);
      await ensureModelDownloaded(tgt, onProgress);

      if (!MLKitTranslator || typeof MLKitTranslator.translateText !== 'function') {
      throw new Error('MLKit native module is not available');
    }
    console.log(`[MLKit] Translating text length ${trimmed.length} from ${getMlkitLang(src)} to ${getMlkitLang(tgt)}`);
    const translated = await MLKitTranslator.translateText(trimmed, getMlkitLang(src), getMlkitLang(tgt));
    const translatedStr = String(translated);
    console.log(`[MLKit] Translation SUCCESS: ${translatedStr.substring(0, 20)}...`);

      return {
        text:    typeof translated === 'string' ? translated.trim() : trimmed,
        engine:  'mlkit',
        srcLang: src,
        tgtLang: tgt,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return {
        text:    trimmed,   // preserve original so Stage 7 can skip this region
        engine:  'failed',
        srcLang: src,
        tgtLang: tgt,
        error:   message,
      };
    }
  }

  /**
   * Batch-translate all non-skipped OCR results.
   *
   * Downloads target language model once (with progress), then translates
   * all regions in sequence. Skipped regions pass through untouched.
   *
   * @param ocrResults  From Stage 5 MobileOCR.extractTextFromRegions()
   * @param src         Source language code
   * @param tgt         Target language code
   * @param onProgress  Progress callback for the model download phase
   */
  async translateAll(
    ocrResults:  MobileOcrResult[],
    src:         string,
    tgt:         string,
    onProgress?: ProgressCallback,
  ): Promise<MobileTranslationResult[]> {
    // Pre-download models once before processing any region
    let downloadFired = false;
    const progressOnce: ProgressCallback = (p) => {
      if (!downloadFired || p.done) {
        downloadFired = true;
        onProgress?.(p);
      }
    };

    // Only pre-download target model. Source model is downloaded lazily if src="auto"
    await ensureModelDownloaded(tgt, progressOnce);
    if (src !== 'auto') {
      await ensureModelDownloaded(src, progressOnce);
    }

    // Translate regions sequentially (ML Kit is single-threaded on device)
    return Promise.all(
      ocrResults.map(async (ocr): Promise<MobileTranslationResult> => {
        if (ocr.skip) {
          // Pass through skipped regions — Stage 7 won't render them
          return { text: ocr.text, engine: 'mlkit', srcLang: src, tgtLang: tgt };
        }

        let detectedSrc = src;
        if (src === 'auto') {
          // 1. ML Kit detector
          detectedSrc = await this.identifyLanguage(ocr.text);

          // 2. Fallback if uncertain
          if (detectedSrc === 'und') {
            detectedSrc = this.identifyScriptHint(ocr.text);
          }

          // 3. Last resort fallback for short pure Latin
          if (detectedSrc === 'und' && /^[a-zA-Z\s]+$/.test(ocr.text.trim()) && ocr.text.trim().length > 0) {
            detectedSrc = 'en';
          }

          if (detectedSrc !== 'und') {
            await ensureModelDownloaded(detectedSrc, progressOnce);
          }
        }

        // If it's completely undetermined, pass it through
        if (detectedSrc === 'und') {
          return { text: ocr.text, engine: 'mlkit', srcLang: 'und', tgtLang: tgt };
        }

        return this.translate(ocr.text, detectedSrc, tgt);
      }),
    );
  }

  /**
   * Identify the language of a text sample using ML Kit Language ID.
   * Returns BCP-47 code (e.g. "hi") or "und" if undetermined.
   */
  async identifyLanguage(text: string): Promise<string> {
    try {
      const lang = await MLKitTranslator.identifyLanguage(text.trim());
      return typeof lang === 'string' ? lang : 'und';
    } catch {
      return 'und';
    }
  }

  /**
   * Fallback: Identify language based on Unicode block hints.
   * Returns BCP-47 code or 'und' if unknown.
   */
  identifyScriptHint(text: string): string {
    if (/[\u0900-\u097F]/.test(text)) return 'hi'; // Devanagari (Hindi)
    if (/[\u0980-\u09FF]/.test(text)) return 'bn'; // Bengali
    if (/[\u0B80-\u0BFF]/.test(text)) return 'ta'; // Tamil
    if (/[\u0C00-\u0C7F]/.test(text)) return 'te'; // Telugu
    if (/[\u0600-\u06FF]/.test(text)) return 'ur'; // Arabic/Urdu
    if (/[\u4E00-\u9FFF]/.test(text)) return 'zh'; // Chinese/Kanji
    if (/[\u0400-\u04FF]/.test(text)) return 'ru'; // Cyrillic
    return 'und';
  }

  /**
   * Pre-flight check: can mobile translation proceed for this target language?
   *
   * Checks if the ML Kit native module is available and the target model can
   * be ensured within a timeout window. Returns false immediately (no throw)
   * if anything fails — the router will fall back to PC in that case.
   *
   * @param tgt        Target language BCP-47 code.
   * @param timeoutMs  Max ms to wait for a model download. Default 20s.
   */
  async canTranslate(tgt: string, timeoutMs = 20_000): Promise<boolean> {
    if (!MLKitTranslator || typeof MLKitTranslator.translateText !== 'function') {
      console.log('[MLKit] canTranslate: native module not available');
      return false;
    }
    if (!BCP_TO_MLKIT[tgt.toLowerCase()]) {
      console.log(`[MLKit] canTranslate: language ${tgt} not supported`);
      return false;
    }
    try {
      await Promise.race([
        ensureModelDownloaded('en'),  // source model (English is almost always cached)
        new Promise<never>((_, rej) => setTimeout(() => rej(new Error('timeout')), timeoutMs)),
      ]);
      await Promise.race([
        ensureModelDownloaded(tgt),
        new Promise<never>((_, rej) => setTimeout(() => rej(new Error('timeout')), timeoutMs)),
      ]);
      return true;
    } catch (e) {
      console.log('[MLKit] canTranslate: model unavailable —', e instanceof Error ? e.message : e);
      return false;
    }
  }

  /**
   * Pre-warm a set of language packs.  Call this on app startup or when
   * the user opens the language picker, so downloads happen before they tap.
   */
  async prefetchModels(
    langs: string[],
    onProgress?: ProgressCallback,
  ): Promise<void> {
    await Promise.all(langs.map(l => ensureModelDownloaded(l, onProgress)));
  }

  /** Exposed for testing: clears the in-memory download cache. */
  _resetCacheForTest(): void {
    _downloadedCache.clear();
    _pendingDownloads.clear();
  }
}

// ── Singleton export ──────────────────────────────────────────────────────────

export const mobileTranslator = new MobileTranslator();
