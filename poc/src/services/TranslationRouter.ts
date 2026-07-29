/**
 * src/services/TranslationRouter.ts
 * ────────────────────────────────────────────────────────────────────────────
 * Phase 3 — Translation Router: Mobile-First, PC-Fallback
 *
 * Mirrors the same hybrid decision pattern as the existing Stage 1–4 pipeline:
 *   Stage 1–4 erase:  Mobile → PC fallback (based on SSIM / coverage)
 *   Stage 5–7 translate: Mobile → PC fallback (based on translation quality gates)
 *
 * ARCHITECTURE:
 *   Each path is complete and self-contained:
 *     Mobile path: MobileOCR (Stage 5) → MobileTranslate (Stage 6) → SkiaRenderer (Stage 7 in UI)
 *     PC path:     POST /api/process-translate (runs Stages 1–7 fully on server)
 *
 * QUALITY GATES (mobile path must pass ALL to avoid PC fallback):
 *   1. canTranslate()           — ML Kit module available + model downloadable
 *   2. OCR confidence           — avg confidence ≥ 0.60 across non-skipped regions
 *   3. Translation availability — ≥ 50% of regions translate successfully
 *   (Stage 7 render coverage is evaluated separately by SkiaRenderer via onRenderComplete)
 *
 * IMPORTANT: Stages 1–4 (/api/process) are NOT touched by this router.
 */

import { extractTextFromRegions, type MobileOcrResult } from '../pipeline/translation/MobileOCR';
import { mobileTranslator, type MobileTranslationResult } from '../pipeline/translation/MobileTranslate';
import { backendService } from './BackendService';
import type { PipelineResult } from '../pipeline/Pipeline';

// ── Quality gate thresholds ────────────────────────────────────────────────────

const THRESHOLDS = {
  /** Minimum average OCR confidence to proceed on mobile path. */
  MIN_OCR_CONFIDENCE:          0.60,
  /** Minimum fraction of regions that must translate successfully. */
  MIN_TRANSLATION_AVAILABILITY: 0.50,
  /**
   * Minimum fraction of eligible regions that must render successfully.
   * Evaluated by SkiaRenderer via onRenderComplete — not in this file.
   */
  MIN_RENDER_COVERAGE:          0.50,
} as const;

// ── Types ─────────────────────────────────────────────────────────────────────

/** Which path produced the final result. */
export type TranslationPath = 'mobile' | 'pc' | 'failed';

/** Reason why the mobile path was skipped or failed. */
export type FallbackReason =
  | 'mlkit_unavailable'        // canTranslate() returned false
  | 'low_ocr_confidence'       // avg OCR confidence below threshold
  | 'all_regions_skipped'      // no OCR text found at all
  | 'low_translation_rate'     // too many ML Kit failures
  | 'mobile_translate_error'   // unexpected exception in mobile path
  | 'pc_unavailable'           // no server reachable
  | 'pc_translate_error';      // server returned error

export interface TranslationRouterResult {
  /** Which path produced this result. */
  path:            TranslationPath;
  /** Human-readable reason for PC fallback (null when mobile succeeded). */
  fallbackReason?: FallbackReason;
  fallbackDetail?: string;         // extra context string for the banner

  /** Final output image URI.
   *  - Mobile path: this is the INPAINTED uri (SkiaRenderer draws text on top in UI).
   *  - PC path: this is the FULLY RENDERED translated JPEG (no further rendering needed).
   */
  outputUri:   string;
  /** Original image URI (used to build SkiaRenderer on mobile path). */
  originalUri: string;
  targetLang:  string;
  /**
   * Source language filter active for this result.
   * 'auto' = all eligible regions were translated (original behavior).
   * BCP-47 code = only matching regions were translated; rest have engine='restored'.
   */
  sourceLang:  string;

  /** Populated on mobile path only. Null on PC path. */
  mobileData: {
    ocrResults:         MobileOcrResult[];
    translationResults: MobileTranslationResult[];
    /** @deprecated SkiaRenderer classifies fonts per-region; this field is unused. */
    fontClasses?:       string[];
    imageWidth:         number;
    imageHeight:        number;
  } | null;

  /** Performance metrics */
  totalMs:         number;
  translatedCount: number;
  detectionCount:  number;
  
  /** Populated on PC path only. List of translated strings for the testing popup. */
  pcTranslatedTexts?: string[];
  pcSourceLangs?: string[];
}

// ── Router ────────────────────────────────────────────────────────────────────

export class TranslationRouter {
  /**
   * Run the full Stage 5–7 translation pipeline, mobile-first.
   *
   * @param result      Completed PipelineResult from Stages 1–4 (already has outputUri + detections).
   * @param targetLang  BCP-47 target language code (e.g. "hi", "de").
   * @param onProgress  Optional progress callback for UI status updates.
   */
  async run(
    result:     PipelineResult,
    targetLang: string,
    onProgress?: (stage: string, detail?: string) => void,
    preferredPath: 'mobile' | 'pc' = 'mobile',
    existingOcrResults?: MobileOcrResult[],
    sourceLang: string = 'auto',
  ): Promise<TranslationRouterResult> {
    const t0 = Date.now();
    const originalUri = result.originalUri!;
    const detections  = result.detections ?? [];

    // Short-circuit: nothing to translate
    if (detections.length === 0) {
      return {
        path:            'mobile',
        outputUri:       result.outputUri,
        originalUri,
        targetLang,
        sourceLang,
        mobileData:      null,
        totalMs:         Date.now() - t0,
        translatedCount: 0,
        detectionCount:  0,
      };
    }

    if (preferredPath === 'mobile') {
      // ── Preferred Mobile (default) ──────────────────────────────────────────
      onProgress?.('mobile_start', `Translating on device…`);

      const mobileResult = await this._tryMobilePath(
        result, targetLang, originalUri, detections, t0, onProgress, sourceLang,
      );

      // Return mobile result if successful
      if (mobileResult) return mobileResult;

      // Mobile failed, fallback to PC
      onProgress?.('pc_start', 'Device translation failed. Falling back to PC…');
      return this._tryPcPath(result, targetLang, originalUri, t0, onProgress, existingOcrResults, sourceLang);

    } else {
      // ── Preferred PC ────────────────────────────────────────────────────────
      onProgress?.('pc_start', 'Translating on PC server…');
      const pcResult = await this._tryPcPath(result, targetLang, originalUri, t0, onProgress, existingOcrResults, sourceLang);

      // Return PC result if successful
      if (pcResult.path === 'pc') return pcResult;

      // PC failed, fallback to Mobile
      onProgress?.('mobile_start', 'PC translation failed. Falling back to device…');
      const mobileFallback = await this._tryMobilePath(
        result, targetLang, originalUri, detections, t0, onProgress, sourceLang,
      );

      // Return mobile result if successful, otherwise return the original PC failure
      if (mobileFallback) return mobileFallback;
      return pcResult;
    }
  }

  // ── Private: mobile path ──────────────────────────────────────────────────

  private async _tryMobilePath(
    result:      PipelineResult,
    targetLang:  string,
    originalUri: string,
    detections:  PipelineResult['detections'],
    t0:          number,
    onProgress?: (stage: string, detail?: string) => void,
    sourceLang:  string = 'auto',
  ): Promise<TranslationRouterResult | null> {
    const imgW = result.width  ?? 1;
    const imgH = result.height ?? 1;

    // ── Gate 1: ML Kit availability ──────────────────────────────────────────
    onProgress?.('mobile_check', 'Checking on-device translation availability…');
    const canTranslate = await mobileTranslator.canTranslate(targetLang);
    if (!canTranslate) {
      console.log('[TranslationRouter] Mobile gate 1 FAIL: ML Kit unavailable for', targetLang);
      this._lastFallbackReason = 'mlkit_unavailable';
      this._lastFallbackDetail = `ML Kit model for "${targetLang}" not available`;
      return null;
    }

    // ── Stage 5: OCR ─────────────────────────────────────────────────────────
    onProgress?.('mobile_ocr', 'Reading text from image…');
    let ocrResults: MobileOcrResult[];
    try {
      ocrResults = await extractTextFromRegions(
        originalUri, detections, { width: imgW, height: imgH },
      );
    } catch (e: any) {
      console.warn('[TranslationRouter] Mobile OCR exception:', e);
      // F-3: Use distinct error code for OCR failures vs. translation failures
      this._lastFallbackReason = 'mlkit_error'; // Use valid FallbackReason
      this._lastFallbackDetail = `OCR failed: ${e?.message ?? e}`;
      return null;
    }

    // ── Gate 2: OCR confidence (combined: MLKit × textQuality) ──────────────
    const nonSkipped = ocrResults.filter(r => !r.skip);
    if (nonSkipped.length === 0) {
      console.log('[TranslationRouter] Mobile gate 2 FAIL: all regions skipped by OCR');
      this._lastFallbackReason = 'all_regions_skipped';
      this._lastFallbackDetail = 'No readable text detected on device';
      return null;
    }

    const avgConf    = nonSkipped.reduce((s, r) => s + r.confidence, 0) / nonSkipped.length;
    const avgQuality = nonSkipped.reduce((s, r) => s + (r.textQuality ?? r.confidence), 0) / nonSkipped.length;
    console.log(`[TranslationRouter] Mobile gate 2: avg combined conf=${avgConf.toFixed(2)}, avg text quality=${avgQuality.toFixed(2)}, ${nonSkipped.length} non-skipped regions`);

    if (avgConf < THRESHOLDS.MIN_OCR_CONFIDENCE) {
      console.log(`[TranslationRouter] Mobile gate 2 FAIL: OCR confidence ${avgConf.toFixed(2)} < ${THRESHOLDS.MIN_OCR_CONFIDENCE}`);
      this._lastFallbackReason = 'low_ocr_confidence';
      this._lastFallbackDetail = `OCR confidence ${(avgConf * 100).toFixed(0)}% below threshold`;
      return null;
    }

    // ── Stage 6: Translate ───────────────────────────────────────────────────
    onProgress?.('mobile_translate', 'Translating text on device…');
    let translationResults: MobileTranslationResult[];
    try {
      translationResults = await mobileTranslator.translateAll(ocrResults, 'auto', targetLang, undefined, sourceLang);
    } catch (e: any) {
      console.warn('[TranslationRouter] Mobile translation exception:', e);
      // F-3: Use distinct error code for translation failures vs. OCR failures
      this._lastFallbackReason = 'mobile_translate_error'; // Use valid FallbackReason
      this._lastFallbackDetail = `Translation failed: ${e?.message ?? e}`;
      return null;
    }

    // ── Gate 3: translation availability ─────────────────────────────────────
    // 'restored' regions are a feature, not a failure — exclude them from the rate check.
    const successCount = translationResults.filter(
      r => r.engine !== 'failed' && r.engine !== 'restored' && r.text?.trim(),
    ).length;
    const restoredCount = translationResults.filter(r => r.engine === 'restored').length;
    const eligibleCount = ocrResults.filter(r => !r.skip).length - restoredCount;
    const rate = eligibleCount > 0 ? successCount / eligibleCount : 1; // if all restored, rate=1 (not a failure)

    if (rate < THRESHOLDS.MIN_TRANSLATION_AVAILABILITY) {
      console.log(`[TranslationRouter] Mobile gate 3 FAIL: translation rate ${rate.toFixed(2)} < ${THRESHOLDS.MIN_TRANSLATION_AVAILABILITY}`);
      this._lastFallbackReason = 'low_translation_rate';
      this._lastFallbackDetail = `Only ${Math.round(rate * 100)}% of regions translated`;
      return null;
    }

    // ── Mobile path SUCCESS ──────────────────────────────────────────────────
    // Stage 7 (SkiaRenderer) runs in the UI — we return the data it needs.
    // fontClasses is intentionally omitted: SkiaRenderer classifies fonts
    // per-region itself; a hardcoded 'SANS_SERIF' array adds no value.

    console.log(`[TranslationRouter] Mobile path SUCCESS: ${successCount}/${eligibleCount} regions translated, ${restoredCount} restored, in ${Date.now() - t0}ms`);
    onProgress?.('mobile_done', `Translated ${successCount} region${successCount !== 1 ? 's' : ''} on device`);

    return {
      path:            'mobile',
      outputUri:       result.outputUri,   // inpainted image — SkiaRenderer overlays text
      originalUri,
      targetLang,
      sourceLang,
      mobileData: {
        ocrResults,
        translationResults,
        imageWidth:  imgW,
        imageHeight: imgH,
      },
      totalMs:         Date.now() - t0,
      translatedCount: successCount,
      detectionCount:  detections.length,
    };
  }

  // ── Private: PC path ──────────────────────────────────────────────────────

  private async _tryPcPath(
    result:      PipelineResult,
    targetLang:  string,
    originalUri: string,
    t0:          number,
    onProgress?: (stage: string, detail?: string) => void,
    existingOcrResults?: MobileOcrResult[],
    sourceLang:  string = 'auto',
  ): Promise<TranslationRouterResult> {
    const fallbackReason = this._lastFallbackReason ?? 'mobile_translate_error';
    const fallbackDetail = this._lastFallbackDetail;

    // Resolve server URL
    onProgress?.('pc_resolve', 'Finding PC server…');
    const baseUrl = await backendService.resolveUrl();

    if (!baseUrl) {
      console.warn('[TranslationRouter] PC path: no server found');
      return {
        path:            'failed',
        fallbackReason:  'pc_unavailable',
        fallbackDetail:  'No PC server found. Connect to the same network.',
        outputUri:       result.outputUri,   // show inpainted (no text) as best effort
        originalUri,
        targetLang,
        sourceLang,
        mobileData:      null,
        totalMs:         Date.now() - t0,
        translatedCount: 0,
        detectionCount:  result.detections?.length ?? 0,
      };
    }

    // Call /api/process-translate (Stages 1–7 on server)
    onProgress?.('pc_upload', 'Uploading to PC server for translation…');
    const pcResult = await backendService.processTranslate(
      originalUri,
      targetLang,
      baseUrl,
      result.width ?? 1,
      result.height ?? 1,
      result.detections,
      existingOcrResults,
      sourceLang,
    );

    if (!pcResult) {
      console.warn('[TranslationRouter] PC path: processTranslate returned null');
      return {
        path:            'failed',
        fallbackReason:  'pc_translate_error',
        fallbackDetail:  'PC server translation failed.',
        outputUri:       result.outputUri,
        originalUri,
        targetLang,
        sourceLang,
        mobileData:      null,
        totalMs:         Date.now() - t0,
        translatedCount: 0,
        detectionCount:  result.detections?.length ?? 0,
      };
    }

    console.log(`[TranslationRouter] PC path SUCCESS: ${pcResult.translations}/${pcResult.detections} regions in ${pcResult.totalMs}ms`);
    onProgress?.('pc_done', `Server translated ${pcResult.translations} region${pcResult.translations !== 1 ? 's' : ''}`);

    return {
      path:            'pc',
      fallbackReason,
      fallbackDetail,
      // PC result is a fully rendered JPEG — no SkiaRenderer needed.
      outputUri:       pcResult.resultUri,
      originalUri,
      targetLang,
      sourceLang,
      mobileData:      null,   // PC handled rendering; no overlay data needed.
      totalMs:         Date.now() - t0,
      translatedCount: pcResult.translations,
      detectionCount:  pcResult.detections,
      pcTranslatedTexts: pcResult.pcTranslatedTexts,
      pcSourceLangs: pcResult.pcSourceLangs,
    };
  }

  // ── Shared state between _tryMobilePath and _tryPcPath ───────────────────

  private _lastFallbackReason?: FallbackReason;
  private _lastFallbackDetail?: string;
}

// ── Singleton export ──────────────────────────────────────────────────────────

export const translationRouter = new TranslationRouter();
