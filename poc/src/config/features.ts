/**
 * src/config/features.ts
 * ────────────────────────────────────────────────────────────────────────────
 * Feature flags for the application.
 */

export const FEATURES = {
  /**
   * Phase 3: Translation Stage (Stages 5–7).
   * Enables the "Translate this" button on the ResultBottomSheet.
   */
  TRANSLATION_ENABLED: true,

  /**
   * Stage 7 (Mobile Path): Live Skia text overlay.
   * When true, translated text is drawn on the inpainted image via SkiaRenderer.
   * When false, the inpainted image is shown as-is (text regions erased but not re-rendered).
   * Set to false to quickly disable Stage 7 mobile without touching rendering code.
   */
  STAGE7_MOBILE_ENABLED: true,
};
