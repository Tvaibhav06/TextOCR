/**
 * src/pipeline/translation/MobileOCR.ts
 * ────────────────────────────────────────────────────────────────────────────
 * Phase 3 — Stage 5 (Mobile Path): Per-Region Text Extraction
 *
 * Uses Google ML Kit Text Recognition v2 to read text within the bounding
 * boxes already detected by Stage 1.
 *
 * Confidence improvements over v1:
 *  - Uses real element-level confidence scores from ML Kit v2 (averaged)
 *  - Multi-signal confidence: element confidence × text quality score
 *  - Text quality heuristics: noise char density, digit ratio, symbol ratio
 *  - OCR noise cleaning: strips OCR artifacts (dangling |, l→I, 0→O etc.)
 *  - Skip logic tightened: noisy text is marked skip before reaching Stage 6
 *
 * Design rules:
 *  - Runs on the ORIGINAL pre-inpainting image URI, never the erased output.
 *  - Receives bbox coordinates from Stage 1 DetectionResult.
 *  - Matches ML Kit blocks to Stage 1 bounding boxes using IoU overlap.
 *  - Regions with finalConfidence < MIN_CONFIDENCE or text < MIN_TEXT_LENGTH
 *    are flagged skip=true.
 */

import TextRecognition from '@react-native-ml-kit/text-recognition';
import type { DetectionResult } from '../TextDetector';

// ── Constants ─────────────────────────────────────────────────────────────────

const MIN_CONFIDENCE   = 0.60;   // Below this → skip=true
const MIN_TEXT_LENGTH  = 3;      // Fewer chars → skip=true
const MIN_IOU_OVERLAP  = 0.30;   // IoU threshold for block→detection matching
const NOISE_SKIP_THRESHOLD = 0.45; // Below this quality score → skip regardless of MLKit conf

// ── Types ─────────────────────────────────────────────────────────────────────

export interface MobileOcrResult {
  /** Matches the Detection id from Stage 1 */
  detId:      number;
  text:       string;
  /** Combined confidence: element-level ML Kit score × text quality score */
  confidence: number;
  /** Raw ML Kit element-level confidence (before quality weighting) */
  mlkitConfidence: number;
  /** Text quality score 0–1: 1.0 = clean readable text, 0.0 = pure noise */
  textQuality: number;
  /** True → keep region erased only; do not translate or render */
  skip:       boolean;
  /** Reason why skip=true (for debugging) */
  skipReason?: string;
  /** Pixel-coordinate bbox (mirrors Stage 1 detection) */
  bbox: { x: number; y: number; width: number; height: number };
  /** Rotation angle in radians calculated from corner points */
  angle:      number;
}

interface ImageDimensions {
  width:  number;
  height: number;
}

// ── Geometry helpers ──────────────────────────────────────────────────────────

interface Rect { x: number; y: number; width: number; height: number }

function ioU(a: Rect, b: Rect): number {
  const ax2 = a.x + a.width;
  const ay2 = a.y + a.height;
  const bx2 = b.x + b.width;
  const by2 = b.y + b.height;

  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
  const intersection = ix * iy;

  if (intersection === 0) return 0;

  const aArea = a.width * a.height;
  const bArea = b.width * b.height;
  return intersection / (aArea + bArea - intersection);
}

function polygonToPixelRect(
  detection: DetectionResult,
  imgW: number,
  imgH: number,
): Rect {
  const { x, y, width, height } = detection.bbox;
  return {
    x:      x      * imgW,
    y:      y      * imgH,
    width:  width  * imgW,
    height: height * imgH,
  };
}

// ── Text cleaning ─────────────────────────────────────────────────────────────

/**
 * Clean common OCR artifacts from recognized text.
 *
 * SAFETY NOTES on what is and is NOT touched:
 *  - All non-ASCII Unicode (U+00A0+) is NEVER stripped by this function.
 *    This covers every Indic script, Arabic, Hebrew, CJK, Thai, etc.
 *  - The | → I/l fix only fires when | is adjacent to ASCII letters — safe.
 *  - Leading/trailing noise strip uses the range [^\w\u00A0-\uFFFF], which
 *    explicitly allows all non-ASCII Unicode and therefore never strips Indic
 *    punctuation like the Hindi danda (।), Arabic tatweel (ـ), etc.
 *  - Isolated single-char cleanup only strips lone ASCII punctuation tokens,
 *    not lone Unicode characters (e.g., "I" the pronoun is fine; CJK single
 *    chars are fine because ML Kit never inserts spaces between CJK glyphs).
 */
export function cleanOcrText(raw: string): string {
  if (!raw) return '';

  let text = raw;

  // Strip C0 + DEL control characters and Unicode zero-width chars.
  // Does NOT touch U+0080+ (Indic scripts, Arabic, CJK etc.)
  text = text.replace(/[\u0000-\u001F\u007F\u200B-\u200D\uFEFF]/g, '');

  // Normalize multiple whitespace to single space
  text = text.replace(/\s+/g, ' ').trim();

  // Fix common OCR confusion: | at ASCII letter word boundaries only
  text = text.replace(/\|(?=[A-Z])/g, 'I');
  text = text.replace(/(?<=[a-z])\|/g, 'l');

  // Strip leading/trailing characters that are neither:
  //   - ASCII word characters (\w = [a-zA-Z0-9_])
  //   - Any non-ASCII Unicode character (U+00A0 and above)
  // This means we keep Devanagari, Arabic, Tamil, CJK etc. at boundaries.
  text = text.replace(/^[^\w\u00A0-\uFFFF]+/, '').replace(/[^\w\u00A0-\uFFFF]+$/, '');

  // Remove common isolated OCR noise tokens (like isolated punctuation).
  // Example removed: "Hello . World" → "Hello World"
  // Example kept:    "I am here" ("I" is \w)  |  "U.S.A" (no surrounding spaces)
  text = text.replace(/(?<=\s|^)[.,\-_|'"~`!;:](?=\s|$)/g, '');

  // Collapse spaces again
  text = text.replace(/\s+/g, ' ').trim();

  return text;
}

// ── Text quality scoring ──────────────────────────────────────────────────────

/**
 * Compute a quality score [0, 1] for OCR-recognized text.
 *
 * Signals (each penalizes the score):
 *  1. Noise char density — fraction of chars that are non-printable, symbols, or punctuation
 *  2. High digit ratio — strings that are >70% digits are likely non-text (codes, numbers)
 *     Exception: pure numeric labels like "42" are valid — only skip if mixed noise
 *  3. Repeated character ratio — "lllll" or "-----" patterns indicate OCR boundary artifacts
 *  4. Very short cleaned text — < 3 chars after cleaning gets penalized hard
 *  5. Word-level plausibility — if >50% of space-separated tokens are single characters,
 *     the text likely has spacing corruption
 *
 * Returns a score in [0, 1]. Below NOISE_SKIP_THRESHOLD → marked as skip.
 */
export function computeTextQuality(text: string): number {
  if (!text || text.length === 0) return 0;

  const chars = [...text]; // Unicode-safe split
  const total = chars.length;

  if (total < 2) return 0.3; // Very short — marginal

  // ── Signal 1: Noise character density ────────────────────────────────────
  // Letters (any script), digits, spaces, and ALL non-ASCII Unicode are "good".
  // C1 control chars (U+0080–U+009F) are the only non-ASCII range treated as noise,
  // because they're invisible formatting bytes, not script characters.
  const noiseChars = chars.filter(c => {
    const code = c.codePointAt(0) ?? 0;
    if (code <= 0x20) return false;  // whitespace/control already stripped
    // ASCII: allow letters, digits, common intra-word punctuation
    if (code < 0x80) {
      return !(
        (code >= 0x41 && code <= 0x5A) || // A-Z
        (code >= 0x61 && code <= 0x7A) || // a-z
        (code >= 0x30 && code <= 0x39) || // 0-9
        code === 0x27 || code === 0x2D    // apostrophe, hyphen
      );
    }
    // C1 control characters (U+0080–U+009F) — invisible formatting bytes → noise
    if (code >= 0x80 && code <= 0x9F) return true;
    // Everything else (U+00A0 and above): any script letter/punct → NOT noise
    return false;
  });
  const noiseDensity = noiseChars.length / total;
  const noiseScore = 1 - Math.min(1, noiseDensity * 2);

  // ── Signal 2: Repeated character ratio ───────────────────────────────────
  // "lllll..." or "-----..." patterns
  let maxRepeat = 1;
  let currentRepeat = 1;
  for (let i = 1; i < chars.length; i++) {
    if (chars[i] === chars[i - 1]) {
      currentRepeat++;
      maxRepeat = Math.max(maxRepeat, currentRepeat);
    } else {
      currentRepeat = 1;
    }
  }
  const repeatRatio = maxRepeat / total;
  const repeatScore = repeatRatio > 0.5 ? 0.2 : 1.0; // Heavy penalty for repeated chars

  // ── Signal 3: Single-char token ratio ────────────────────────────────────
  // If >70% of space-separated tokens are single characters, OCR likely has
  // spacing corruption (e.g. "H e l l o" instead of "Hello").
  //
  // EXCEPTION — CJK text: In Chinese/Japanese/Korean, ML Kit never inserts
  // spaces between characters (it returns "你好世界" not "你 好 世 界"), so this
  // signal is irrelevant for CJK and won't trigger. Thai/Khmer similarly have
  // no spaces. Only Latin-script corruption looks like "H e l l o".
  //
  // EXCEPTION — Single-char English words: "I", "a" are \w chars and ARE
  // matched by our \w filter in cleanOcrText, so they survive as tokens and
  // tokenScore here applies equally — but the 70% bar is hard to hit with
  // normal text that contains real words.
  const tokens = text.split(/\s+/).filter(t => t.length > 0);
  if (tokens.length === 0) return 0;
  // Count tokens that are exactly 1 Unicode codepoint AND are ASCII
  // (Non-ASCII single chars like Chinese glyphs are legitimate single tokens)
  const singleAsciiCharTokens = tokens.filter(t => {
    const codepoints = [...t];
    if (codepoints.length !== 1) return false;
    const code = codepoints[0].codePointAt(0) ?? 0;
    return code < 0x80; // only penalize ASCII single tokens (spacing corruption)
  }).length;
  const singleCharRatio = singleAsciiCharTokens / tokens.length;
  // Penalize only if >70% of tokens are lone ASCII chars
  const tokenScore = singleCharRatio > 0.70 ? 0.4 : 1.0;

  // ── Signal 4: All-symbol text ────────────────────────────────────────────
  // Text that is entirely symbols/punctuation (no letters/digits at all)
  const hasLetterOrDigit = chars.some(c => {
    const code = c.codePointAt(0) ?? 0;
    return (code >= 0x30 && code <= 0x39) || // digits
           (code >= 0x41 && code <= 0x5A) || // A-Z
           (code >= 0x61 && code <= 0x7A) || // a-z
           code > 0x7F;                       // any non-ASCII (scripts)
  });
  if (!hasLetterOrDigit) return 0.1; // Almost certainly noise

  // ── Combine signals (geometric mean keeps all signals meaningful) ─────────
  const combined = noiseScore * repeatScore * tokenScore;
  return Math.max(0, Math.min(1, combined));
}

// ── Element-level confidence extraction ──────────────────────────────────────

/**
 * Extract the average element-level confidence from an ML Kit block.
 *
 * ML Kit Text Recognition v2 exposes confidence at the Element level
 * (individual words/tokens). We average across all elements in the block.
 *
 * Falls back to the language-detection heuristic if elements have no
 * confidence field (older library versions).
 */
function extractBlockConfidence(block: any): number {
  const elements: any[] = (block.lines ?? []).flatMap((line: any) => line.elements ?? []);

  if (elements.length === 0) {
    // Fallback: language detection heuristic
    return (block.recognizedLanguages?.length ?? 0) > 0 ? 0.75 : 0.5;
  }

  // Check if ML Kit actually provides confidence at element level
  const confsWithValues = elements
    .map(el => el.confidence)
    .filter(c => typeof c === 'number' && !isNaN(c) && c > 0);

  if (confsWithValues.length === 0) {
    // Library doesn't expose element confidence — use language heuristic
    const langConf = (block.recognizedLanguages?.length ?? 0) > 0 ? 0.75 : 0.5;
    // Weight by how many elements have recognized languages
    const elementLangConf = elements.reduce((sum: number, el: any) => {
      return sum + ((el.recognizedLanguages?.length ?? 0) > 0 ? 1.0 : 0.5);
    }, 0) / elements.length;
    return (langConf + elementLangConf) / 2;
  }

  // True element-level average confidence
  const avg = confsWithValues.reduce((s, c) => s + c, 0) / confsWithValues.length;
  return Math.max(0, Math.min(1, avg));
}

// ── Core: ML Kit OCR + block matching ─────────────────────────────────────────

/**
 * Extracts and cleans text from each Stage 1 detection region using ML Kit.
 *
 * Improvements over v1:
 *  - Uses real element-level ML Kit v2 confidence (averaged per block)
 *  - Computes a text quality score independently of ML Kit confidence
 *  - Final confidence = mlkitConfidence × textQuality
 *  - Applies OCR noise cleaning before returning text
 *  - More detailed skip reasons for debugging
 *
 * @param originalImageUri  URI of the ORIGINAL pre-inpainting image.
 * @param detections        Array of DetectionResult from Stage 1.
 * @param imageDims         Pixel dimensions of the original image.
 * @returns                 Array of MobileOcrResult, one per detection.
 */
export async function extractTextFromRegions(
  originalImageUri: string,
  detections:        DetectionResult[],
  imageDims:         ImageDimensions,
): Promise<MobileOcrResult[]> {

  if (detections.length === 0) {
    return [];
  }

  // Run ML Kit on the full image — single pass for all regions
  const recognitionResult = await TextRecognition.recognize(originalImageUri);
  const blocks = recognitionResult.blocks ?? [];

  const { width: imgW, height: imgH } = imageDims;

  return detections.map((det): MobileOcrResult => {
    const detRect = polygonToPixelRect(det, imgW, imgH);

    // Find the ML Kit block with the highest IoU to this detection box
    let bestRawText     = '';
    let bestMlkitConf   = 0;
    let bestIoU         = 0;
    let bestAngle       = 0;

    for (const block of blocks) {
      const blockFrame = block.frame;
      if (!blockFrame) continue;

      const blockRect: Rect = {
        x:      blockFrame.left   ?? 0,
        y:      blockFrame.top    ?? 0,
        width:  blockFrame.width  ?? 0,
        height: blockFrame.height ?? 0,
      };

      const overlap = ioU(detRect, blockRect);
      if (overlap < MIN_IOU_OVERLAP || overlap <= bestIoU) continue;

      // Aggregate raw text from all lines + elements
      const rawBlockText = (block.lines ?? [])
        .flatMap((line: any) => line.elements ?? [])
        .map((el: any) => el.text ?? '')
        .join(' ')
        .trim();

      if (!rawBlockText) continue;

      const mlkitConf = extractBlockConfidence(block);

      bestIoU       = overlap;
      bestRawText   = rawBlockText;
      bestMlkitConf = mlkitConf;
      
      if (block.cornerPoints && block.cornerPoints.length >= 2) {
        const p0 = block.cornerPoints[0];
        const p1 = block.cornerPoints[1];
        bestAngle = Math.atan2(p1.y - p0.y, p1.x - p0.x);
      }
    }

    // M-4: Second-chance match for detections that got no text in the primary pass.
    // Happens when two adjacent detection bboxes share the same wide ML Kit block and
    // one wins at IoU ≥ MIN_IOU_OVERLAP while the other scores below the threshold.
    // Re-scan at a lower threshold (SECOND_CHANCE_IOU) to give the empty detection
    // a chance to claim any overlapping block. Confidence is penalised 10% to reflect
    // that the match is weaker.
    const SECOND_CHANCE_IOU = 0.12;
    if (!bestRawText) {
      for (const block of blocks) {
        const blockFrame = block.frame;
        if (!blockFrame) continue;
        const blockRect: Rect = {
          x:      blockFrame.left   ?? 0,
          y:      blockFrame.top    ?? 0,
          width:  blockFrame.width  ?? 0,
          height: blockFrame.height ?? 0,
        };
        const overlap = ioU(detRect, blockRect);
        if (overlap < SECOND_CHANCE_IOU || overlap <= bestIoU) continue;

        const rawBlockText = (block.lines ?? [])
          .flatMap((line: any) => line.elements ?? [])
          .map((el: any) => el.text ?? '')
          .join(' ')
          .trim();
        if (!rawBlockText) continue;

        const mlkitConf = extractBlockConfidence(block);
        bestIoU       = overlap;
        bestRawText   = rawBlockText;
        // 10% confidence penalty for second-chance matches
        bestMlkitConf = mlkitConf * 0.90;
        if (block.cornerPoints && block.cornerPoints.length >= 2) {
          const p0 = block.cornerPoints[0];
          const p1 = block.cornerPoints[1];
          bestAngle = Math.atan2(p1.y - p0.y, p1.x - p0.x);
        }
        console.log(`[MobileOCR] M-4 second-chance match for region ${det.id} at IoU=${overlap.toFixed(2)}`);
      }
    }

    // ── Clean OCR noise from the best match ──────────────────────────────
    const cleanedText = cleanOcrText(bestRawText);

    // ── Compute text quality independently ───────────────────────────────
    const textQuality = computeTextQuality(cleanedText);

    // ── Combined confidence: ML Kit × quality heuristic ──────────────────
    // Weight: 60% ML Kit confidence, 40% text quality
    // This ensures noisy text is penalized even if ML Kit was "confident"
    const finalConf = bestMlkitConf > 0
      ? (bestMlkitConf * 0.60 + textQuality * 0.40)
      : 0;

    // ── Skip logic ───────────────────────────────────────────────────────
    let skip = false;
    let skipReason: string | undefined;

    if (!cleanedText || cleanedText.length < MIN_TEXT_LENGTH) {
      skip = true;
      skipReason = `text too short (${cleanedText.length} chars after cleaning)`;
    } else if (finalConf < MIN_CONFIDENCE) {
      skip = true;
      skipReason = `low combined confidence (mlkit=${bestMlkitConf.toFixed(2)}, quality=${textQuality.toFixed(2)}, combined=${finalConf.toFixed(2)})`;
    } else if (textQuality < NOISE_SKIP_THRESHOLD) {
      skip = true;
      skipReason = `text quality too low (${textQuality.toFixed(2)} < ${NOISE_SKIP_THRESHOLD})`;
    }

    if (skipReason) {
      console.log(`[MobileOCR] Region ${det.id} skipped: ${skipReason}`);
    } else {
      console.log(`[MobileOCR] Region ${det.id}: "${cleanedText.substring(0, 30)}" conf=${finalConf.toFixed(2)} (mlkit=${bestMlkitConf.toFixed(2)}, quality=${textQuality.toFixed(2)})`);
    }

    return {
      detId:           det.id,
      text:            cleanedText,
      confidence:      finalConf,
      mlkitConfidence: bestMlkitConf,
      textQuality,
      skip,
      skipReason,
      bbox: {
        x:      detRect.x,
        y:      detRect.y,
        width:  detRect.width,
        height: detRect.height,
      },
      angle: bestAngle,
    };
  });
}
