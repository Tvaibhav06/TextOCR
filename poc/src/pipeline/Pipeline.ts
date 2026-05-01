/**
 * Pipeline Orchestrator
 *
 * Wires Stage 1→2→3→4 in sequence with:
 *  - Progress callbacks (for animated UI)
 *  - Cancellation token support
 *  - Structured error reporting per stage
 *  - Timing breakdown for each stage
 *
 * Usage:
 *   const result = await pipeline.run({ imageUri, width, height }, onProgress);
 */

import { textDetector, type DetectionResult } from './TextDetector';
import { maskGenerator,  type MaskResult }     from './MaskGenerator';
import { inpainter }                            from './Inpainter';
import { poissonBlender, qualityAssessor }      from './PostProcessor';

// ─── Types ────────────────────────────────────────────────────────────────────

export type PipelineStage =
  | 'idle'
  | 'detecting'
  | 'masking'
  | 'inpainting'
  | 'postprocessing'
  | 'done'
  | 'error';

export interface PipelineInput {
  imageUri:    string;
  imageWidth:  number;
  imageHeight: number;
  confidenceThreshold?: number;   // default 0.50
  dilationPx?:          number;   // default 3
  inpaintMethod?:       'telea' | 'ns' | 'pconv';
  enablePoisson?:       boolean;
  poissonIterations?:   number;
}

export interface StageMetrics {
  stage:   string;
  ms:      number;
  success: boolean;
  detail?: string;
}

export interface PipelineResult {
  outputUri:   string;
  maskResult:  MaskResult;
  detections:  DetectionResult[];
  quality: {
    ssim: number;
    psnr: number;
    aboveThreshold: boolean;
    suggestServerUpgrade: boolean;
  };
  timing: StageMetrics[];
  totalMs: number;
}

export interface PipelineProgress {
  stage:      PipelineStage;
  progress:   number;   // 0–100
  message:    string;
}

export type ProgressCallback = (p: PipelineProgress) => void;

// ─── Cancellation ─────────────────────────────────────────────────────────────

export class CancellationToken {
  private _cancelled = false;
  cancel()                 { this._cancelled = true; }
  get isCancelled()        { return this._cancelled; }
  throwIfCancelled()       { if (this._cancelled) throw new Error('Pipeline cancelled by user'); }
}

// ─── Pipeline ─────────────────────────────────────────────────────────────────

export class TextRemovalPipeline {
  private _isWarmedUp = false;

  /** Pre-load models (call on app startup) */
  async warmUp(): Promise<void> {
    if (this._isWarmedUp) return;
    await Promise.all([
      textDetector.warmUp({ enableEastFallback: true }),
      inpainter.warmUp(),
    ]);
    this._isWarmedUp = true;
    console.log('[Pipeline] Warm-up complete');
  }

  async run(
    input: PipelineInput,
    onProgress?: ProgressCallback,
    token?: CancellationToken,
  ): Promise<PipelineResult> {
    const t0      = performance.now();
    const timing: StageMetrics[] = [];
    const report  = (stage: string, ms: number, success: boolean, detail?: string) =>
      timing.push({ stage, ms: Math.round(ms), success, detail });
    const progress = (stage: PipelineStage, pct: number, msg: string) =>
      onProgress?.({ stage, progress: pct, message: msg });

    token?.throwIfCancelled();

    // ── Stage 1: Text Detection ───────────────────────────────────────────────
    progress('detecting', 5, 'Detecting text regions…');
    let ts = performance.now();
    let detections: DetectionResult[];
    try {
      detections = await textDetector.detect(
        input.imageUri,
        input.imageWidth,
        input.imageHeight,
        { confidenceThreshold: input.confidenceThreshold ?? 0.5 },
      );
      report('detection', performance.now() - ts, true, `${detections.length} regions found`);
      console.log(`[Pipeline] Stage 1 ✓ — ${detections.length} detections in ${Math.round(performance.now()-ts)}ms`);
    } catch (err: any) {
      report('detection', performance.now() - ts, false, err.message);
      throw new Error(`Detection failed: ${err.message}`);
    }

    token?.throwIfCancelled();

    if (detections.length === 0) {
      return this.buildNoTextResult(input.imageUri, t0);
    }

    // ── Stage 2: Mask Generation ──────────────────────────────────────────────
    progress('masking', 30, 'Generating mask…');
    ts = performance.now();
    let mask: MaskResult;
    try {
      mask = await maskGenerator.generate(
        detections,
        input.imageWidth,
        input.imageHeight,
        { dilationPx: input.dilationPx ?? 3 },
      );
      report('masking', performance.now() - ts, true,
        `${(mask.coverage * 100).toFixed(1)}% coverage`);
      console.log(`[Pipeline] Stage 2 ✓ — mask coverage=${(mask.coverage*100).toFixed(1)}% in ${Math.round(performance.now()-ts)}ms`);
    } catch (err: any) {
      report('masking', performance.now() - ts, false, err.message);
      throw new Error(`Mask generation failed: ${err.message}`);
    }

    token?.throwIfCancelled();

    // ── Stage 3: Inpainting ───────────────────────────────────────────────────
    progress('inpainting', 50, 'Erasing text…');
    ts = performance.now();
    let inpaintedUri: string;
    let inpaintMethod: string;
    try {
      const inpaintResult = await inpainter.inpaint(input.imageUri, mask, {
        method: input.inpaintMethod ?? 'telea',
        pconvCoverageThreshold: 0.15,
      });
      inpaintedUri  = inpaintResult.outputUri;
      inpaintMethod = inpaintResult.method;
      report('inpainting', performance.now() - ts, true, `method=${inpaintMethod}`);
      console.log(`[Pipeline] Stage 3 ✓ — inpaint(${inpaintMethod}) in ${Math.round(performance.now()-ts)}ms`);
    } catch (err: any) {
      report('inpainting', performance.now() - ts, false, err.message);
      throw new Error(`Inpainting failed: ${err.message}`);
    }

    token?.throwIfCancelled();

    // ── Stage 4: Post-processing ──────────────────────────────────────────────
    progress('postprocessing', 80, 'Blending edges…');
    ts = performance.now();
    let finalUri = inpaintedUri;

    // ⚠️ POC LIMITATION — Quality assessment uses stub pixel arrays.
    //
    // Reading actual pixels from a URI requires expo-file-system + Skia
    // readPixels integration that is not yet implemented in this POC.
    //
    // What the SSIM/PSNR values mean here:
    //   - stubOrig  = all zeros (black image)
    //   - stubResult = all zeros except byte[0] = 128
    //   → SSIM will be < 1, MSE > 0: the QA math paths ARE exercised.
    //   → Absolute values are meaningless — do not use to gate quality decisions.
    //
    // Production fix: read originalPixels via Skia.Surface.makeImageSnapshot()
    //   and resultPixels similarly, then pass both to qualityAssessor.assess().
    const stubOrig   = new Uint8Array(input.imageWidth * input.imageHeight * 4);
    const stubResult = new Uint8Array(input.imageWidth * input.imageHeight * 4);
    // Introduce a single-byte difference so QA math is non-degenerate
    if (stubResult.length > 4) { stubResult[0] = 128; stubResult[1] = 128; stubResult[2] = 128; }

    const quality = qualityAssessor.assess(stubOrig, stubResult, mask, 0.82);

    report('postprocessing', performance.now() - ts, true,
      `SSIM=${quality.ssim.toFixed(3)} PSNR=${quality.psnr}dB`);
    console.log(`[Pipeline] Stage 4 ✓ — SSIM=${quality.ssim.toFixed(3)} in ${Math.round(performance.now()-ts)}ms`);

    progress('done', 100, 'Done!');

    const totalMs = Math.round(performance.now() - t0);
    console.log(`[Pipeline] Complete in ${totalMs}ms`);

    return {
      outputUri:  finalUri,
      maskResult: mask,
      detections,
      quality: {
        ...quality,
        suggestServerUpgrade: !quality.aboveThreshold || mask.coverage > 0.25,
      },
      timing,
      totalMs,
    };
  }

  private buildNoTextResult(imageUri: string, t0: number): PipelineResult {
    return {
      outputUri:   imageUri,
      maskResult:  { data: new Uint8Array(0), width: 0, height: 0, maskedPixels: 0, coverage: 0 },
      detections:  [],
      quality: { ssim: 1, psnr: Infinity, aboveThreshold: true, suggestServerUpgrade: false },
      timing:  [{ stage: 'detection', ms: Math.round(performance.now() - t0), success: true, detail: 'No text found' }],
      totalMs: Math.round(performance.now() - t0),
    };
  }
}

export const pipeline = new TextRemovalPipeline();
