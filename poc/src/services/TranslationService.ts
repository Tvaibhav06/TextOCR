/**
 * src/services/TranslationService.ts
 * ────────────────────────────────────────────────────────────────────────────
 * Thin orchestration wrapper for the Phase 3 Translation Stage (Stages 5–7).
 *
 * Delegates all routing logic to TranslationRouter.
 * Exports TranslationSession for use in App.tsx and ResultBottomSheet.
 */

import { translationRouter, type TranslationRouterResult, type TranslationPath, type FallbackReason } from './TranslationRouter';
import type { PipelineResult } from '../pipeline/Pipeline';
import type { MobileOcrResult } from '../pipeline/translation/MobileOCR';
import type { MobileTranslationResult } from '../pipeline/translation/MobileTranslate';

// ── Types ─────────────────────────────────────────────────────────────────────

/** The result of a completed translation session — stored in App state. */
export interface TranslationSession {
  /** Which path produced this result. */
  path: TranslationPath;
  /** Human-readable reason the mobile path was skipped (null if mobile succeeded). */
  fallbackReason?: FallbackReason;
  fallbackDetail?: string;

  /**
   * Final output URI.
   * - mobile path: the INPAINTED image URI (SkiaRenderer overlays text on top in UI).
   * - pc path: the FULLY RENDERED translated JPEG (display as a static image).
   */
  outputUri:  string;
  originalUri: string;
  targetLang: string;

  /**
   * Mobile rendering data — non-null only when path === 'mobile'.
   * Passed to <TranslatedOverlay> to render Stage 7 in the UI.
   */
  mobileData: {
    ocrResults:         MobileOcrResult[];
    translationResults: MobileTranslationResult[];
    fontClasses:        string[];
    imageWidth:         number;
    imageHeight:        number;
  } | null;

  totalMs:         number;
  translatedCount: number;
  detectionCount:  number;
  
  /** PC path only: text returned from the server for the testing popup */
  pcTranslatedTexts?: string[];
}

// ── Service ───────────────────────────────────────────────────────────────────

export class TranslationService {
  /**
   * Run the full translation pipeline (Stages 5–7), mobile-first.
   *
   * Tries the mobile path (ML Kit OCR → ML Kit translate → data for SkiaRenderer).
   * Falls back to PC server (/api/process-translate) if any quality gate fails.
   *
   * @param result      Completed PipelineResult from Stages 1–4.
   * @param targetLang  BCP-47 target language code.
   * @param onProgress  Optional UI progress callback.
   */
  static async run(
    result:     PipelineResult,
    targetLang: string,
    onProgress?: (stage: string, detail?: string) => void,
    forceMode?: 'mobile' | 'pc',
    existingOcrResults?: any[],
  ): Promise<TranslationSession> {
    const routerResult: TranslationRouterResult = await translationRouter.run(
      result, targetLang, onProgress, forceMode, existingOcrResults
    );

    // Map router result directly to TranslationSession (same shape)
    return {
      path:            routerResult.path,
      fallbackReason:  routerResult.fallbackReason,
      fallbackDetail:  routerResult.fallbackDetail,
      outputUri:       routerResult.outputUri,
      originalUri:     routerResult.originalUri,
      targetLang:      routerResult.targetLang,
      mobileData:      routerResult.mobileData,
      totalMs:         routerResult.totalMs,
      translatedCount: routerResult.translatedCount,
      detectionCount:  routerResult.detectionCount,
      pcTranslatedTexts: routerResult.pcTranslatedTexts,
    };
  }
}
