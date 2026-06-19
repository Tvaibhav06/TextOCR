/**
 * src/pipeline/translation/SkiaRenderer.tsx
 * ────────────────────────────────────────────────────────────────────────────
 * Phase 3 — Stage 7 (Mobile Path): Live Text Overlay Renderer
 *
 * Uses @shopify/react-native-skia's Paragraph API (HarfBuzz + ICU built-in)
 * to render translated text as a live canvas overlay on the inpainted image.
 *
 * Stage 7 is a VISUAL APPROXIMATION stage:
 *  - Matches font family (serif / sans-serif / monospace) via classical heuristic.
 *  - Matches weight (bold / regular) via stroke-width estimation.
 *  - Samples text color via K-Means(k=2) on region pixels.
 *  - Draws a background fill rectangle for contrast safety.
 *  - Falls back silently at each step — never forces a bad render.
 *  - Italic is NOT attempted (Skia has no synthetic skew API).
 *  - BiDi / RTL handled automatically by Skia ParagraphBuilder + ICU.
 *
 * Design invariants:
 *  - ADDITIVE — does not change any Stage 1–4 UI or logic.
 *  - Controlled by FEATURES.STAGE7_MOBILE_ENABLED in src/config/features.ts.
 *  - onRenderComplete fires with (renderedCount, totalCount) for quality gating.
 *  - All font loading uses array syntax [ require(...) ] to avoid the
 *    "typefacesToLoad[familyName].map is not a function" crash.
 */

import React, { useEffect, useMemo, useState, useRef } from 'react';
import { View, LayoutChangeEvent, ActivityIndicator, Text as RNText, StyleSheet } from 'react-native';
import * as FileSystem from 'expo-file-system';
import {
  Canvas,
  Image,
  Paragraph,
  RoundedRect,
  Skia,
  TextAlign,
  FontSlant,
  useFonts,
  useImage,
  Group,
  ColorType,
} from '@shopify/react-native-skia';
import type { MobileOcrResult } from './MobileOCR';
import type { MobileTranslationResult } from './MobileTranslate';

// ── Types ─────────────────────────────────────────────────────────────────────

/** Coarse font family class — DISPLAY / HANDWRITTEN collapse to SANS_SERIF. */
export type FontClass = 'SANS_SERIF' | 'SERIF' | 'MONOSPACE';

/** Font weight — only BOLD and REGULAR are supported in the bundled font set. */
export type FontWeight = 'REGULAR' | 'BOLD';

export interface SkiaRendererProps {
  /** URI of the inpainted (text-erased) image — rendered as the base layer. */
  inpaintedUri:    string;
  /** URI of the ORIGINAL (pre-inpainting) image — used for color/style sampling. */
  originalUri:     string;
  /** Pixel width of the image (used to set Canvas size). */
  imageWidth:      number;
  /** Pixel height of the image. */
  imageHeight:     number;
  /** OCR results from Stage 5 — provides bbox coordinates + raw text. */
  ocrResults:      MobileOcrResult[];
  /** Translation results from Stage 6 — provides translated text. */
  translationResults: MobileTranslationResult[];
  /** Target language BCP-47 code. Used to pick the correct Noto font. */
  targetLang:      string;
  /**
   * Called after Stage 7 rendering completes.
   * renderedCount = how many regions were actually drawn.
   * totalCount    = total non-skipped, non-failed regions.
   */
  onRenderComplete?: (renderedCount: number, totalCount: number) => void;
}

// ── RTL language codes ────────────────────────────────────────────────────────

const RTL_LANGS = new Set(['ar', 'he', 'fa', 'ur', 'ps', 'sd', 'ks']);

function isRtl(lang: string): boolean {
  return RTL_LANGS.has(lang.toLowerCase());
}

// ── Cap-height ratios — bbox_h × ratio ≈ readable cap height ─────────────────

const CAP_RATIOS: Record<string, number> = {
  hi: 0.68, mr: 0.68, ne: 0.68, sa: 0.68,
  bn: 0.70, as: 0.70,
  ta: 0.72, te: 0.72, kn: 0.72, ml: 0.72,
  gu: 0.70, pa: 0.70,
  ar: 0.70, ur: 0.70,
  he: 0.72,
  zh: 0.85, ja: 0.85, ko: 0.85,
};

function estimateFontSize(bboxH: number, lang: string): number {
  const ratio = CAP_RATIOS[lang.toLowerCase()] ?? 0.72;
  // Font size = Cap Height / Ratio. bboxH serves as the max available cap height.
  return Math.max(8, Math.floor(bboxH / ratio));
}

// ── Font mapping — (FontClass, weight, lang) → bundled Noto asset name ───────
//
// Three-level fallback mirrors the PC renderer's FontSelector:
//   Level 0: exact (class, weight, script) match
//   Level 1: (SANS_SERIF, REGULAR, script)
//   Level 2: NotoSans-Regular (always present, always safe)
//
// Notes:
//   - Malayalam: NotoSansMalayalam-Regular.ttf now bundled (Stage 7 P1 complete).
//   - CJK (zh/ja/ko): No CJK Noto bundled (file size ~30MB). Falls back to NotoSans-Regular.
//     This renders tofu boxes for CJK — acceptable for POC; PC path handles CJK correctly.

function getFontAssetName(fontClass: FontClass, weight: FontWeight, lang: string): string {
  const lc   = lang.toLowerCase();
  const bold = weight === 'BOLD';

  // ── Script-specific fonts (ignore fontClass — script determines font) ─────
  if (['hi','mr','ne','sa','mai','doi','kok','brx'].includes(lc)) {
    if (fontClass === 'SERIF') return 'NotoSerifDevanagari-Regular';
    return bold ? 'NotoSansDevanagari-Bold' : 'NotoSansDevanagari-Regular';
  }
  if (['bn','as','mni'].includes(lc))  return 'NotoSansBengali-Regular';   // no bold bundled
  if (lc === 'ta') return 'NotoSansTamil-Regular';
  if (lc === 'te') return 'NotoSansTelugu-Regular';
  if (lc === 'kn') return 'NotoSansKannada-Regular';
  if (lc === 'ml') return bold ? 'NotoSansMalayalam-Bold' : 'NotoSansMalayalam-Regular';
  if (lc === 'gu') return 'NotoSansGujarati-Regular';
  if (lc === 'pa') return 'NotoSansGurmukhi-Regular';
  if (['ar','ur','fa','ks','sd'].includes(lc)) {
    return bold ? 'NotoNaskhArabic-Bold' : 'NotoNaskhArabic-Regular';
  }
  if (lc === 'he') return 'NotoSansHebrew-Regular';
  if (['zh','ja','ko'].includes(lc))   return 'NotoSans-Regular';  // no CJK font bundled → L2

  // ── Latin / Cyrillic / default — use font class and weight ───────────────
  if (fontClass === 'SERIF') {
    return bold ? 'NotoSerif-Bold' : 'NotoSerif-Regular';
  }
  if (fontClass === 'MONOSPACE') {
    return bold ? 'NotoSansMono-Bold' : 'NotoSansMono-Regular';
  }
  // SANS_SERIF (also DISPLAY + HANDWRITTEN collapse here)
  return bold ? 'NotoSans-Bold' : 'NotoSans-Regular';
}

// ── Classical font-family classification ──────────────────────────────────────
//
// JS port of the PC's FontClassifier (renderer.py).
// Runs on the RGBA pixel buffer from the original image region.
// No TFLite model required.
//
// Algorithm:
//   1. Convert RGBA → grayscale luminance.
//   2. Binarize (Otsu-equivalent threshold on the luminance histogram).
//   3. Serif score — erode vertical stems, measure horizontal residual runs.
//   4. Monospace score — CoV of column-projection run lengths.
//   5. Default → SANS_SERIF.
//
// Confidence < 0.45 on both serif and mono → SANS_SERIF fallback.

export function classifyFontFamily(
  rgbaPixels: Uint8Array,
  width:      number,
  height:     number,
): FontClass {
  if (!rgbaPixels || rgbaPixels.length < width * height * 4) return 'SANS_SERIF';
  if (width < 8 || height < 8) return 'SANS_SERIF';

  // ── Grayscale ─────────────────────────────────────────────────────────────
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = rgbaPixels[i * 4];
    const g = rgbaPixels[i * 4 + 1];
    const b = rgbaPixels[i * 4 + 2];
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }

  // ── Otsu binarization ─────────────────────────────────────────────────────
  const hist = new Array(256).fill(0);
  for (const v of gray) hist[v]++;
  const total = width * height;
  let sumAll = 0;
  for (let i = 0; i < 256; i++) sumAll += i * hist[i];
  let wB = 0, sumB = 0, maxVar = 0, threshold = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;
    const betweenVar = wB * wF * (mB - mF) ** 2;
    if (betweenVar > maxVar) { maxVar = betweenVar; threshold = t; }
  }

  // Foreground = dark pixels (text on light bg) or light on dark — detect by majority
  const darkCount = gray.filter(v => v < threshold).length;
  const binary    = new Uint8Array(width * height);
  if (darkCount < total * 0.5) {
    // Dark text on light bg
    for (let i = 0; i < total; i++) binary[i] = gray[i] < threshold ? 255 : 0;
  } else {
    // Light text on dark bg
    for (let i = 0; i < total; i++) binary[i] = gray[i] >= threshold ? 255 : 0;
  }

  const fgCount = binary.reduce((s, v) => s + (v > 0 ? 1 : 0), 0);
  if (fgCount < 20) return 'SANS_SERIF'; // not enough foreground pixels

  // ── Monospace score — CoV of horizontal run lengths between column gaps ───
  const colProj = new Array(width).fill(0);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (binary[y * width + x] > 0) colProj[x]++;
    }
  }
  const runs: number[] = [];
  let run = 0;
  for (let x = 0; x < width; x++) {
    if (colProj[x] > 0) {
      run++;
    } else {
      if (run > 2) runs.push(run);
      run = 0;
    }
  }
  if (run > 2) runs.push(run);

  let monoScore = 0;
  if (runs.length >= 2) {
    const mean = runs.reduce((s, r) => s + r, 0) / runs.length;
    const std  = Math.sqrt(runs.reduce((s, r) => s + (r - mean) ** 2, 0) / runs.length);
    const cov  = std / Math.max(mean, 1e-9);
    monoScore  = Math.max(0, 1 - cov / 0.30);
  }
  if (monoScore > 0.65) return 'MONOSPACE';

  // ── Serif score — vertical erosion residual → horizontal serif runs ───────
  // Simplified: look at top/bottom row fg pixels relative to middle
  // Counts pixels in top 20% and bottom 20% of height (serif terminal region)
  const serifZone = Math.max(2, Math.floor(height * 0.20));
  let serifCandidates = 0;
  let totalFg = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (binary[y * width + x] === 0) continue;
      totalFg++;
      if (y < serifZone || y >= height - serifZone) serifCandidates++;
    }
  }
  const serifScore = totalFg > 0 ? Math.min(1, (serifCandidates / totalFg) * 3.5) : 0;
  if (serifScore > 0.55) return 'SERIF';

  return 'SANS_SERIF';
}

// ── Stroke-width estimation → font weight ─────────────────────────────────────
//
// Port of PC's WeightEstimator._estimate_weight() using distanceTransform.
// Approximation: for each fg pixel, we compute its minimum distance to a
// background pixel. The median of those distances × 2 ≈ stroke width.
// stroke_px >= 5 → BOLD, else REGULAR.

export function estimateWeight(
  rgbaPixels: Uint8Array,
  width:      number,
  height:     number,
): FontWeight {
  if (!rgbaPixels || width < 4 || height < 4) return 'REGULAR';

  // Grayscale + binarize (same as classifyFontFamily)
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    gray[i] = Math.round(
      0.299 * rgbaPixels[i*4] + 0.587 * rgbaPixels[i*4+1] + 0.114 * rgbaPixels[i*4+2]
    );
  }
  const hist = new Array(256).fill(0);
  for (const v of gray) hist[v]++;
  let sumAll = 0, wB = 0, sumB = 0, maxVar = 0, thr = 128;
  for (let i = 0; i < 256; i++) sumAll += i * hist[i];
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = gray.length - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sumAll - sumB) / wF;
    const v = wB * wF * (mB - mF) ** 2;
    if (v > maxVar) { maxVar = v; thr = t; }
  }
  const darkCount = gray.filter(v => v < thr).length;
  const binary    = new Uint8Array(width * height);
  if (darkCount < gray.length * 0.5) {
    for (let i = 0; i < gray.length; i++) binary[i] = gray[i] < thr ? 255 : 0;
  } else {
    for (let i = 0; i < gray.length; i++) binary[i] = gray[i] >= thr ? 255 : 0;
  }

  // Approximate distanceTransform: for each fg pixel, BFS distance to nearest bg
  // Full BFS is too slow in JS for large regions; use a fast approximation:
  // scan each fg pixel's immediate 8-neighbors; if any bg neighbor exists, dist=1.
  // Otherwise sample a 5×5 window. Median of sampled distances × 2 ≈ stroke_px.
  const dists: number[] = [];
  const sampleStep = Math.max(1, Math.floor(Math.sqrt(width * height) / 30));

  for (let y = 0; y < height; y += sampleStep) {
    for (let x = 0; x < width; x += sampleStep) {
      if (binary[y * width + x] === 0) continue;
      // Check expanding radius for nearest bg pixel
      let dist = 0;
      outer: for (let r = 1; r <= 8; r++) {
        for (let dy = -r; dy <= r; dy++) {
          for (let dx = -r; dx <= r; dx++) {
            if (Math.abs(dx) !== r && Math.abs(dy) !== r) continue;
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
            if (binary[ny * width + nx] === 0) { dist = r; break outer; }
          }
        }
      }
      if (dist > 0) dists.push(dist);
    }
  }

  if (dists.length === 0) return 'REGULAR';
  dists.sort((a, b) => a - b);
  const medianHalf = dists[Math.floor(dists.length / 2)];
  const strokePx   = medianHalf * 2.0;
  return strokePx >= 5.0 ? 'BOLD' : 'REGULAR';
}

// ── K-Means color sampling ────────────────────────────────────────────────────
//
// k=2 clustering on pixel values. Returns the darker cluster centroid as
// text color (BGR as [r,g,b]) and border median of inpainted crop as bg color.
// Falls back to defaults if pixel data is unavailable.

function kMeans2(pixels: Uint8Array, stride: number): [number, number, number] {
  // stride = 4 for RGBA; we use channels 0,1,2 (RGB)
  const n = Math.floor(pixels.length / stride);
  if (n < 2) return [26, 26, 26]; // fallback: near-black

  // Initialize centroids from first and last pixel
  let c0 = [pixels[0], pixels[1], pixels[2]];
  let c1 = [pixels[(n - 1) * stride], pixels[(n - 1) * stride + 1], pixels[(n - 1) * stride + 2]];

  for (let iter = 0; iter < 8; iter++) {
    const s0 = [0, 0, 0]; let n0 = 0;
    const s1 = [0, 0, 0]; let n1 = 0;
    for (let i = 0; i < n; i++) {
      const r = pixels[i * stride], g = pixels[i * stride + 1], b = pixels[i * stride + 2];
      const d0 = (r-c0[0])**2 + (g-c0[1])**2 + (b-c0[2])**2;
      const d1 = (r-c1[0])**2 + (g-c1[1])**2 + (b-c1[2])**2;
      if (d0 < d1) { s0[0]+=r; s0[1]+=g; s0[2]+=b; n0++; }
      else          { s1[0]+=r; s1[1]+=g; s1[2]+=b; n1++; }
    }
    if (n0 > 0) c0 = [s0[0]/n0, s0[1]/n0, s0[2]/n0];
    if (n1 > 0) c1 = [s1[0]/n1, s1[1]/n1, s1[2]/n1];
  }

  // The minority cluster (fewer pixels) is typically the text color,
  // since text surface area is smaller than the background surface area.
  // This works universally for light-on-dark AND dark-on-light text!
  let n0_final = 0; let n1_final = 0;
  for (let i = 0; i < n; i++) {
    const r = pixels[i * stride], g = pixels[i * stride + 1], b = pixels[i * stride + 2];
    const d0 = (r-c0[0])**2 + (g-c0[1])**2 + (b-c0[2])**2;
    const d1 = (r-c1[0])**2 + (g-c1[1])**2 + (b-c1[2])**2;
    if (d0 < d1) n0_final++; else n1_final++;
  }
  const text = n0_final < n1_final ? c0 : c1;
  return [Math.round(text[0]), Math.round(text[1]), Math.round(text[2])];
}

function borderMedian(pixels: Uint8Array, width: number, height: number, border = 4): [number,number,number] {
  if (!pixels || width <= 0 || height <= 0) return [255, 255, 255];
  
  const rs: number[] = [], gs: number[] = [], bs: number[] = [];
  const safeBorder = Math.max(1, Math.min(border, Math.floor(width / 2), Math.floor(height / 2)));
  
  for (let x = 0; x < width; x++) {
    for (let b = 0; b < safeBorder; b++) {
      const top    = (b * width + x) * 4;
      const bottom = ((height - 1 - b) * width + x) * 4;
      if (top >= 0 && top < pixels.length && bottom >= 0 && bottom < pixels.length) {
        rs.push(pixels[top], pixels[bottom]);
        gs.push(pixels[top+1], pixels[bottom+1]);
        bs.push(pixels[top+2], pixels[bottom+2]);
      }
    }
  }
  for (let y = safeBorder; y < height - safeBorder; y++) {
    for (let b = 0; b < safeBorder; b++) {
      const left  = (y * width + b) * 4;
      const right = (y * width + (width - 1 - b)) * 4;
      if (left >= 0 && left < pixels.length && right >= 0 && right < pixels.length) {
        rs.push(pixels[left], pixels[right]);
        gs.push(pixels[left+1], pixels[right+1]);
        bs.push(pixels[left+2], pixels[right+2]);
      }
    }
  }
  
  if (rs.length === 0) return [255, 255, 255];
  
  const med = (arr: number[]) => { 
    arr.sort((a,b)=>a-b); 
    return arr[Math.floor(arr.length/2)] ?? 255; 
  };
  return [med(rs), med(gs), med(bs)];
}
// ── Per-region style attributes (computed once per render) ────────────────────

interface RegionStyle {
  fontClass:   FontClass;
  weight:      FontWeight;
  textColor:   ReturnType<typeof Skia.Color>;
  bgColor:     ReturnType<typeof Skia.Color>;
  bgAlpha:     number;  // 0–1
}

const DEFAULT_STYLE: RegionStyle = {
  fontClass:  'SANS_SERIF',
  weight:     'REGULAR',
  textColor:  Skia.Color('#1A1A1A'),
  bgColor:    Skia.Color('rgba(255,255,255,0.0)'),
  bgAlpha:    0,
};

// ── Paragraph builder ─────────────────────────────────────────────────────────

function buildParagraph(
  text:       string,
  baseFontSize: number,
  textColor:  ReturnType<typeof Skia.Color>,
  fontName:   string,
  maxWidth:   number,
  maxHeight:  number,
  rtl:        boolean,
  weight:     FontWeight,
  fontMgr:    any
) {
  let fontSize = baseFontSize;
  let para: ReturnType<typeof Skia.ParagraphBuilder.Make> | any = null;

  for (let i = 0; i < 5; i++) {
    const paraStyle = {
      textAlign:     TextAlign.Center,
      textDirection: rtl ? 1 : 0,
      maxLines:      4,
      ellipsis:      '…',
    };

    const builder = Skia.ParagraphBuilder.Make(paraStyle, fontMgr);
    builder.pushStyle({
      color:        textColor,
      fontSize:     fontSize,
      fontFamilies: [fontName, 'NotoSans-Regular'],
      fontStyle:    {
        weight: weight === 'BOLD' ? 700 : 400,
        width:  5,
        slant:  FontSlant.Upright,
      },
    });
    builder.addText(text);
    builder.pop();

    para = builder.build();
    para.layout(maxWidth);

    const metrics = para.getLineMetrics ? para.getLineMetrics() : [];
    const lineCount = metrics.length > 0 ? metrics.length : 1;
    const longestLine = para.getLongestLine ? para.getLongestLine() : 0;
    
    const fitsHorizontally = longestLine <= maxWidth;
    const fitsVertically = para.getHeight() <= maxHeight;

    if (lineCount === 1) {
      // Single line text: Trust the mathematically perfect cap-height font size.
      // Ignore para.getHeight() because typographic ascenders/descenders artificially inflate it.
      if (fitsHorizontally) break;
    } else {
      // Multi-line text: Must fit entirely within the original bounding box height and width.
      if (fitsHorizontally && fitsVertically) break;
    }

    if (fontSize <= 8) {
      break;
    }
    fontSize = Math.floor(fontSize * 0.85);
  }
  return para;
}

// ── Robust Local Image Loader ─────────────────────────────────────────────────

function useLocalSkImage(uri: string) {
  const [image, setImage] = useState<ReturnType<typeof Skia.Image.MakeImageFromEncoded> | null>(null);

  useEffect(() => {
    let active = true;
    if (!uri) return;

    FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 })
      .then(b64 => {
        if (!active) return;
        const data = Skia.Data.fromBase64(b64);
        if (!data) {
          console.warn(`[SkiaRenderer] Failed to decode base64 for ${uri}`);
          return;
        }
        const img = Skia.Image.MakeImageFromEncoded(data);
        if (!img) {
          console.warn(`[SkiaRenderer] Failed to decode image from data for ${uri}`);
          return;
        }
        setImage(img);
      })
      .catch(err => {
        console.warn(`[SkiaRenderer] Failed to load local image: ${uri}`, err);
      });

    return () => { active = false; };
  }, [uri]);

  return image;
}

// ── Main Component ────────────────────────────────────────────────────────────

export function TranslatedOverlay({
  inpaintedUri,
  originalUri,
  imageWidth,
  imageHeight,
  ocrResults,
  translationResults,
  targetLang,
  onRenderComplete,
}: SkiaRendererProps): JSX.Element {
  const inpaintedImage = useLocalSkImage(inpaintedUri);
  const originalImage  = useLocalSkImage(originalUri);
  const rtl = isRtl(targetLang);

  // ── Load all Noto font assets into Skia memory ────────────────────────────
  // useFonts MUST receive { 'FamilyName': [require(...)] } — array syntax.
  // A bare require() value (a number in Metro) triggers the
  // "typefacesToLoad[familyName].map is not a function" crash.
  const customFonts = useFonts({
    'NotoSans-Regular':              [require('../../../assets/fonts/NotoSans-Regular.ttf')],
    'NotoSans-Bold':                 [require('../../../assets/fonts/NotoSans-Bold.ttf')],
    'NotoSerif-Regular':             [require('../../../assets/fonts/NotoSerif-Regular.ttf')],
    'NotoSerif-Bold':                [require('../../../assets/fonts/NotoSerif-Bold.ttf')],
    'NotoSansMono-Regular':          [require('../../../assets/fonts/NotoSansMono-Regular.ttf')],
    'NotoSansMono-Bold':             [require('../../../assets/fonts/NotoSansMono-Bold.ttf')],
    'NotoSansDevanagari-Regular':    [require('../../../assets/fonts/NotoSansDevanagari-Regular.ttf')],
    'NotoSansDevanagari-Bold':       [require('../../../assets/fonts/NotoSansDevanagari-Bold.ttf')],
    'NotoSerifDevanagari-Regular':   [require('../../../assets/fonts/NotoSerifDevanagari-Regular.ttf')],
    'NotoSansBengali-Regular':       [require('../../../assets/fonts/NotoSansBengali-Regular.ttf')],
    'NotoSansTamil-Regular':         [require('../../../assets/fonts/NotoSansTamil-Regular.ttf')],
    'NotoSansTelugu-Regular':        [require('../../../assets/fonts/NotoSansTelugu-Regular.ttf')],
    'NotoSansKannada-Regular':       [require('../../../assets/fonts/NotoSansKannada-Regular.ttf')],
    'NotoSansMalayalam-Regular':     [require('../../../assets/fonts/NotoSansMalayalam-Regular.ttf')],
    'NotoSansMalayalam-Bold':        [require('../../../assets/fonts/NotoSansMalayalam-Bold.ttf')],
    'NotoSansGujarati-Regular':      [require('../../../assets/fonts/NotoSansGujarati-Regular.ttf')],
    'NotoSansGurmukhi-Regular':      [require('../../../assets/fonts/NotoSansGurmukhi-Regular.ttf')],
    'NotoNaskhArabic-Regular':       [require('../../../assets/fonts/NotoNaskhArabic-Regular.ttf')],
    'NotoNaskhArabic-Bold':          [require('../../../assets/fonts/NotoNaskhArabic-Bold.ttf')],
    'NotoSansHebrew-Regular':        [require('../../../assets/fonts/NotoSansHebrew-Regular.ttf')],
  });
  const fontsLoaded = customFonts != null;

  // ── Per-region style extraction — runs once after fonts and images load ───
  // Extracts text color, bg color, font class, and weight from pixel data.
  const [regionStyles, setRegionStyles] = useState<RegionStyle[]>([]);

  useEffect(() => {
    if (!fontsLoaded || !originalImage || !inpaintedImage) return;

    // Extract pixels for both images from Skia
    const origInfo    = originalImage.getImageInfo();
    const inpInfo     = inpaintedImage.getImageInfo();
    
    const origPixels  = originalImage.readPixels(0, 0, origInfo);
    const inpPixels   = inpaintedImage.readPixels(0, 0, inpInfo);

    // We check the native color format so we can normalize BGRA to RGBA.
    // (We do not pass a custom ImageInfo to readPixels because origInfo might be a HostObject that fails to spread).
    const isOrigNativeBgra = origInfo.colorType === 6; // 6 = BGRA_8888
    const isInpNativeBgra  = inpInfo.colorType === 6;

    const styles: RegionStyle[] = ocrResults.map((ocr) => {
      try {
        if (!origPixels || !inpPixels) return DEFAULT_STYLE;

        const { x, y, width: bw, height: bh } = ocr.bbox;

        // Clamp bbox to image bounds
        const x0  = Math.max(0, Math.round(x));
        const y0  = Math.max(0, Math.round(y));
        const x1  = Math.min(origInfo.width,  x0 + Math.round(bw));
        const y1  = Math.min(origInfo.height, y0 + Math.round(bh));
        const cw  = x1 - x0;
        const ch  = y1 - y0;

        if (cw < 4 || ch < 4) return DEFAULT_STYLE;

        // Crop original image region (RGBA stride = width * 4)
        const origStride = origInfo.width * 4;
        const origCrop   = new Uint8Array(cw * ch * 4);
        for (let row = 0; row < ch; row++) {
          const srcOff = (y0 + row) * origStride + x0 * 4;
          const dstOff = row * cw * 4;
          origCrop.set(
            (origPixels as Uint8Array).subarray(srcOff, srcOff + cw * 4),
            dstOff,
          );
          // Explicit channel normalization if Skia failed to fulfill RGBA request
          if (isOrigNativeBgra) {
            for (let i = 0; i < cw; i++) {
              const p = dstOff + i * 4;
              const b = origCrop[p];
              origCrop[p] = origCrop[p + 2];
              origCrop[p + 2] = b;
            }
          }
        }

        // ── Crop inpainted region for bg color ──
        // Scale coordinates carefully since the backend might have downscaled inpaintedImage!
        const scaleX = inpInfo.width / origInfo.width;
        const scaleY = inpInfo.height / origInfo.height;

        const ix0 = Math.max(0, Math.floor(x * scaleX));
        const iy0 = Math.max(0, Math.floor(y * scaleY));
        const ix1 = Math.min(inpInfo.width, Math.ceil((x + bw) * scaleX));
        const iy1 = Math.min(inpInfo.height, Math.ceil((y + bh) * scaleY));
        const icw = Math.max(0, ix1 - ix0);
        const ich = Math.max(0, iy1 - iy0);

        let inpCrop: Uint8Array;
        if (icw < 2 || ich < 2) {
          // Fallback to a fake 2x2 crop if scaled box is too tiny
          inpCrop = new Uint8Array(2 * 2 * 4).fill(255);
        } else {
          const inpStride = inpInfo.width * 4;
          inpCrop = new Uint8Array(icw * ich * 4);
          for (let row = 0; row < ich; row++) {
            const srcOff = (iy0 + row) * inpStride + ix0 * 4;
            const dstOff = row * icw * 4;
            inpCrop.set(
              (inpPixels as Uint8Array).subarray(srcOff, srcOff + icw * 4),
              dstOff,
            );
            // Explicit channel normalization
            if (isInpNativeBgra) {
              for (let i = 0; i < icw; i++) {
                const p = dstOff + i * 4;
                const b = inpCrop[p];
                inpCrop[p] = inpCrop[p + 2];
                inpCrop[p + 2] = b;
              }
            }
          }
        }

        // Font classification (skip for very small regions)
        const fontClass = (cw >= 16 && ch >= 8)
          ? classifyFontFamily(origCrop, cw, ch)
          : 'SANS_SERIF';

        // Weight estimation
        const weight = (cw >= 8 && ch >= 8)
          ? estimateWeight(origCrop, cw, ch)
          : 'REGULAR';

        // Text color from original region
        const [tr, tg, tb] = kMeans2(origCrop, 4);
        const textColor = Skia.Color(`rgb(${tr},${tg},${tb})`);

        // Background color from inpainted border
        const [br, bg, bb] = borderMedian(inpCrop, icw, ich);
        const bgColor  = Skia.Color(`rgba(${br},${bg},${bb},0.85)`);

        // Contrast ratio check (simplified WCAG luminance)
        const lum = (r: number, g: number, b: number) => {
          const [rr, gg, bb2] = [r/255, g/255, b/255].map(
            v => v <= 0.04045 ? v/12.92 : ((v+0.055)/1.055)**2.4
          );
          return 0.2126*rr + 0.7152*gg + 0.0722*bb2;
        };
        const l1 = lum(tr, tg, tb);
        const l2 = lum(br, bg, bb);
        const contrast = (Math.max(l1,l2)+0.05) / (Math.min(l1,l2)+0.05);
        // Show bg fill only when contrast is low (< 3:1) — otherwise transparent
        const bgAlpha = contrast < 3.0 ? 0.85 : 0.0;

        return { fontClass, weight, textColor, bgColor, bgAlpha };

      } catch (e) {
        console.warn('[SkiaRenderer] Style extraction failed for region', ocr.detId, e);
        return DEFAULT_STYLE;
      }
    });

    setRegionStyles(styles);
  }, [fontsLoaded, originalImage, inpaintedImage, ocrResults, targetLang]);

  // ── Build paragraphs — memoized, only after fonts + styles are ready ──────
  const { regions, renderedCount, totalCount } = useMemo(() => {
    if (!fontsLoaded || regionStyles.length === 0) {
      return { regions: [], renderedCount: 0, totalCount: 0 };
    }

    const paired = ocrResults.map((ocr, i) => ({
      ocr,
      tr:    translationResults[i],
      style: regionStyles[i] ?? DEFAULT_STYLE,
    }));

    const eligible = paired.filter(({ ocr, tr }) =>
      !ocr.skip && tr && tr.engine !== 'failed' && tr.text?.trim()
    );

    const built = eligible.map(({ ocr, tr, style }) => {
      try {
        const fontSize = estimateFontSize(ocr.bbox.height, targetLang);
        const fontName = getFontAssetName(style.fontClass, style.weight, targetLang);

        const para = buildParagraph(
          tr.text,
          fontSize,
          style.textColor,
          fontName,
          ocr.bbox.width,
          ocr.bbox.height,
          rtl,
          style.weight,
          customFonts
        );
        return { ocr, para, style, ok: true };
      } catch (e) {
        console.warn('[SkiaRenderer] Paragraph build failed for region', ocr.detId, e);
        return { ocr, para: null, style, ok: false };
      }
    }).filter(r => r.ok && r.para != null);

    console.log(`[SkiaRenderer] Built ${built.length}/${eligible.length} paragraphs for lang=${targetLang}`);

    return {
      regions:      built,
      renderedCount: built.length,
      totalCount:   eligible.length,
    };
  }, [ocrResults, translationResults, targetLang, rtl, fontsLoaded, regionStyles]);

  // ── Fire quality callback once paragraphs are built ───────────────────────
  const onRenderCompleteRef = useRef(onRenderComplete);
  useEffect(() => {
    onRenderCompleteRef.current = onRenderComplete;
  }, [onRenderComplete]);

  useEffect(() => {
    if (fontsLoaded && regionStyles.length > 0 && onRenderCompleteRef.current) {
      onRenderCompleteRef.current(renderedCount, totalCount);
    }
  }, [fontsLoaded, regionStyles, renderedCount, totalCount]);

  const [layout, setLayout] = useState({ width: 0, height: 0 });

  return (
    <View
      style={{ flex: 1, width: '100%' }}
      onLayout={(e: LayoutChangeEvent) => setLayout(e.nativeEvent.layout)}
    >
      {!fontsLoaded ? (
        <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
          <ActivityIndicator size="large" color="#1A73E8" />
        </View>
      ) : (
        <>
      {(!inpaintedImage || layout.width === 0 || layout.height === 0) && (
        <View style={{ position: 'absolute', top: 20, left: 20, backgroundColor: 'rgba(0,0,0,0.8)', padding: 10, zIndex: 999 }}>
          <RNText style={{ color: 'white' }}>Debug Info:</RNText>
          <RNText style={{ color: 'white' }}>layout: {layout.width}x{layout.height}</RNText>
          <RNText style={{ color: 'white' }}>inpaintedImage: {inpaintedImage ? 'LOADED' : 'NULL'}</RNText>
          <RNText style={{ color: 'white' }}>originalImage: {originalImage ? 'LOADED' : 'NULL'}</RNText>
          <RNText style={{ color: 'white' }}>inpaintedUri: {inpaintedUri}</RNText>
        </View>
      )}
      {layout.width > 0 && layout.height > 0 && inpaintedImage && (() => {
        const scale = Math.min(
          layout.width  / imageWidth,
          layout.height / imageHeight,
        );
        const tx = (layout.width  - imageWidth  * scale) / 2;
        const ty = (layout.height - imageHeight * scale) / 2;

        return (
          <Canvas style={{ width: layout.width, height: layout.height }}>
            <Group transform={[{ translateX: tx }, { translateY: ty }, { scale }]}>

              {/* Layer 1: Inpainted base image */}
              <Image
                image={inpaintedImage}
                x={0}
                y={0}
                width={imageWidth}
                height={imageHeight}
                fit="fill"
              />

              {/* Layer 2: Per-region background fill + translated text overlay */}
              {regions.map(({ ocr, para, style }, idx) => (
                <Group key={`region-${ocr.detId}-${idx}`}>

                  {/* Background fill — only when contrast is low */}
                  {style.bgAlpha > 0 && (
                    <RoundedRect
                      x={ocr.bbox.x}
                      y={ocr.bbox.y}
                      width={ocr.bbox.width}
                      height={ocr.bbox.height}
                      r={2}
                      color={style.bgColor}
                    />
                  )}

                  {/* Translated text paragraph (Vertically centered) */}
                  <Paragraph
                    paragraph={para!}
                    x={ocr.bbox.x}
                    y={ocr.bbox.y + Math.max(0, (ocr.bbox.height - para!.getHeight()) / 2)}
                    width={ocr.bbox.width}
                  />
                </Group>
              ))}
            </Group>
          </Canvas>
        );
      })()}
      </>
      )}
    </View>
  );
}
