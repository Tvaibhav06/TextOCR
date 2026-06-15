"""
python-backend/pipeline/renderer.py
────────────────────────────────────────────────────────────────────────────────
Phase 3 — Stage 7 (PC Path): Font Matching & Text Re-Render

This module is ADDITIVE. It does NOT modify core.py, ocr.py, or translation.py.
It receives OcrResult + TranslationResult lists and a pair of images, and
returns a new BGR image with translated text rendered in-place.

Pipeline:
  FontAttributeExtractor → FontSelector → TextRenderer

Font Attribute Extraction (per region):
  1. FontClassifier   — classical serif/sans/mono detection (no model required)
  2. WeightEstimator  — cv2.distanceTransform → stroke width → THIN/REGULAR/BOLD
  3. ItalicDetector   — PCA skew angle on foreground pixels → bool
  4. ColorExtractor   — K-Means(k=2) for text color, border median for bg color
  5. SizeEstimator    — cap-height ratio per target script

Rendering (per region):
  1. FontSelector     — (FontClass, Weight, script) → Noto font file path
  2. Shaping          — uharfbuzz shapes text using OpenType GSUB/GPOS tables
  3. BiDi             — python-bidi visual reorder for RTL scripts
  4. Rasterization    — freetype-py renders shaped glyphs to NumPy array
  5. Composite        — cv2 alpha-composites text onto inpainted image

Constraints:
  - Pillow is NOT used (pinned at 9.5.0 for simple-lama-inpainting compatibility).
  - All rendering is NumPy + cv2 + freetype-py.
  - Stages 1–4 (/api/process endpoint) are completely untouched.

Usage:
    from pipeline.renderer import TextRenderer, FontAttributeExtractor
    extractor = FontAttributeExtractor()
    renderer  = TextRenderer()
    result_bgr = renderer.render_all(inpainted_bgr, original_bgr,
                                     ocr_results, translations, target_lang)
"""

from __future__ import annotations

import enum
import logging
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("textremover.renderer")

# ── Font directory ─────────────────────────────────────────────────────────────
FONTS_DIR = Path(__file__).parent / "fonts"

# ── RTL language codes ─────────────────────────────────────────────────────────
RTL_LANGS = {"ar", "he", "fa", "ur", "ps", "sd", "ks"}

# ── Script → primary Unicode block start (for auto-script detection) ───────────
_SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali":    (0x0980, 0x09FF),
    "tamil":      (0x0B80, 0x0BFF),
    "telugu":     (0x0C00, 0x0C7F),
    "kannada":    (0x0C80, 0x0CFF),
    "malayalam":  (0x0D00, 0x0D7F),
    "gujarati":   (0x0A80, 0x0AFF),
    "gurmukhi":   (0x0A00, 0x0A7F),
    "oriya":      (0x0B00, 0x0B7F),
    "sinhala":    (0x0D80, 0x0DFF),
    "arabic":     (0x0600, 0x06FF),
    "hebrew":     (0x0590, 0x05FF),
    "cjk":        (0x4E00, 0x9FFF),
    "hangul":     (0xAC00, 0xD7FF),
    "cyrillic":   (0x0400, 0x04FF),
}


# ════════════════════════════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════════════════════════════

class FontClass(enum.Enum):
    SANS_SERIF  = "sans_serif"
    SERIF       = "serif"
    MONOSPACE   = "monospace"
    DISPLAY     = "display"
    HANDWRITTEN = "handwritten"


class FontWeight(enum.Enum):
    THIN    = "thin"
    REGULAR = "regular"
    BOLD    = "bold"


# ════════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class FontAttrs:
    """All visual attributes extracted from a single detection region."""
    font_class:  FontClass              = FontClass.SANS_SERIF
    weight:      FontWeight             = FontWeight.REGULAR
    is_italic:   bool                   = False
    is_rtl:      bool                   = False
    size_px:     int                    = 16
    text_color:  Tuple[int, int, int]   = (0, 0, 0)      # BGR
    bg_color:    Tuple[int, int, int]   = (255, 255, 255) # BGR
    fallback_level: int                 = 0  # 0=full 1=script-only 2=universal


# ════════════════════════════════════════════════════════════════════════════════
# FONT CLASSIFIER — classical, no model required
# ════════════════════════════════════════════════════════════════════════════════

class FontClassifier:
    """
    Classical font family classifier operating on a cropped grayscale text region.

    Approach:
      - Serif detection: measure stroke-end horizontal/vertical terminations
        using morphological hit-or-miss patterns.
      - Monospace detection: measure coefficient of variation of run lengths
        in the horizontal projection.
      - Display / Handwritten: confidence threshold — if serif and mono scores
        are both low AND the region has high local curvature variance, classify
        as DISPLAY. Handwritten detection is not attempted (falls back to
        SANS_SERIF per approved plan).

    Returns FontClass with a confidence float (0.0–1.0).
    Below MIN_CONFIDENCE → caller uses level-1 fallback (SANS_SERIF).
    """

    MIN_CONFIDENCE = 0.50

    def classify(self,
                 crop_bgr: np.ndarray) -> Tuple[FontClass, float]:
        """
        Classify the font family of a text crop.

        Args:
            crop_bgr: Cropped BGR image of the original text region.

        Returns:
            (FontClass, confidence)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return FontClass.SANS_SERIF, 0.0

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) \
               if crop_bgr.ndim == 3 else crop_bgr.copy()

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if binary.mean() > 200:
            # Inverted (white text on dark bg) — flip
            binary = cv2.bitwise_not(binary)

        # Guard: need at least a few text pixels
        fg_count = int(np.count_nonzero(binary))
        if fg_count < 20:
            return FontClass.SANS_SERIF, 0.0

        serif_score = self._serif_score(binary)
        mono_score  = self._mono_score(binary)

        if mono_score > 0.65:
            return FontClass.MONOSPACE, mono_score
        if serif_score > 0.55:
            return FontClass.SERIF, serif_score
        return FontClass.SANS_SERIF, 1.0 - serif_score

    # ── Private helpers ───────────────────────────────────────────────────────

    def _serif_score(self, binary: np.ndarray) -> float:
        """
        Estimate serif likelihood.

        Serifs appear as short horizontal strokes at the top/bottom of vertical
        stems. We detect them by eroding with a small vertical kernel and then
        looking for horizontal runs in the difference mask (the removed pixels
        are candidate serifs).
        """
        h, w = binary.shape
        if h < 8 or w < 8:
            return 0.0

        # Thin vertical kernel — keeps vertical stems
        vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, h // 5)))
        stems  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_k)
        # Residual = candidate serif pixels
        residual = cv2.subtract(binary, stems)

        # Horizontal runs in residual ≥ 2px → serif termination
        horiz_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        serif_px = cv2.morphologyEx(residual, cv2.MORPH_OPEN, horiz_k)

        fg_total    = float(np.count_nonzero(binary))
        serif_total = float(np.count_nonzero(serif_px))
        return min(1.0, serif_total / max(1.0, fg_total) * 8.0)

    def _mono_score(self, binary: np.ndarray) -> float:
        """
        Estimate monospace likelihood via horizontal run-length CoV.

        In monospace fonts, character widths are nearly equal, so the
        coefficient of variation (std/mean) of horizontal run lengths is low.
        """
        h, w = binary.shape
        if w < 10:
            return 0.0

        # Horizontal projection — count fg pixels per row
        row_sums = binary.sum(axis=1).astype(float) / 255.0
        # Collapse to run-lengths of connected horizontal groups
        # Approximate: look at gaps in column-projection
        col_proj = binary.sum(axis=0).astype(float) / 255.0
        # Find zero-columns (gaps between characters)
        gaps = col_proj < 1.0
        # Run lengths between gaps
        runs = []
        run = 0
        for g in gaps:
            if not g:
                run += 1
            else:
                if run > 2:
                    runs.append(run)
                run = 0
        if run > 2:
            runs.append(run)
        if len(runs) < 2:
            return 0.0
        arr  = np.array(runs, dtype=float)
        cov  = arr.std() / max(arr.mean(), 1e-9)
        # Low CoV → monospace (CoV < 0.15 is very uniform)
        return max(0.0, 1.0 - cov / 0.30)


# ════════════════════════════════════════════════════════════════════════════════
# FONT ATTRIBUTE EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════════

# Cap-height ratio per target language script (bbox_h × ratio ≈ cap height)
_CAP_RATIOS: dict[str, float] = {
    "hi": 0.68, "mr": 0.68, "ne": 0.68, "sa": 0.68,   # Devanagari
    "bn": 0.70, "as": 0.70, "mni": 0.70,               # Bengali
    "ta": 0.72,                                          # Tamil
    "te": 0.72, "kn": 0.72, "ml": 0.72,                # South Dravidian
    "gu": 0.70, "pa": 0.70,                             # Gujarati / Gurmukhi
    "ar": 0.70, "ur": 0.70, "fa": 0.70,                # Arabic
    "he": 0.72,                                          # Hebrew
    "zh": 0.85, "ja": 0.85, "ko": 0.85,                # CJK (square glyphs)
    "default": 0.72,
}


class FontAttributeExtractor:
    """
    Extracts all FontAttrs from a single detection region.

    Input: original_bgr crop + inpainted_bgr crop + target language code.
    """

    def __init__(self) -> None:
        self._classifier = FontClassifier()

    def extract(self,
                original_crop: np.ndarray,
                inpainted_crop: np.ndarray,
                bbox_h: int,
                target_lang: str) -> FontAttrs:
        """
        Full attribute extraction for one region.

        Falls back gracefully at each step. Never raises.
        """
        attrs = FontAttrs()
        attrs.is_rtl  = target_lang.lower() in RTL_LANGS
        attrs.size_px = max(8, int(bbox_h * _CAP_RATIOS.get(
                            target_lang.lower(), _CAP_RATIOS["default"])))

        # ── Font class ────────────────────────────────────────────────────────
        try:
            fc, conf = self._classifier.classify(original_crop)
            if conf >= FontClassifier.MIN_CONFIDENCE:
                attrs.font_class = fc
                attrs.fallback_level = 0
            else:
                attrs.font_class = FontClass.SANS_SERIF
                attrs.fallback_level = 1
        except Exception as exc:
            log.debug("FontClassifier failed: %s", exc)
            attrs.fallback_level = 1

        # ── Weight ────────────────────────────────────────────────────────────
        try:
            attrs.weight = self._estimate_weight(original_crop)
        except Exception as exc:
            log.debug("Weight estimation failed: %s", exc)

        # ── Italic ────────────────────────────────────────────────────────────
        try:
            attrs.is_italic = self._detect_italic(original_crop)
        except Exception as exc:
            log.debug("Italic detection failed: %s", exc)

        # ── Colours ───────────────────────────────────────────────────────────
        try:
            attrs.text_color = self._dominant_color(original_crop)
        except Exception as exc:
            log.debug("Text color extraction failed: %s", exc)

        try:
            attrs.bg_color = self._border_median(inpainted_crop)
        except Exception as exc:
            log.debug("BG color extraction failed: %s", exc)

        return attrs

    # ── Weight estimation ─────────────────────────────────────────────────────

    def _estimate_weight(self, crop: np.ndarray) -> FontWeight:
        """
        Stroke width → weight using cv2.distanceTransform on the foreground mask.

        Thin:    median stroke px < 2
        Regular: 2 ≤ median < 4
        Bold:    median ≥ 4
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if binary.mean() > 200:
            binary = cv2.bitwise_not(binary)

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        fg_dist = dist[binary > 0]
        if fg_dist.size == 0:
            return FontWeight.REGULAR

        # Distance transform max-value ≈ half stroke width
        median_half = float(np.median(fg_dist))
        stroke_px = median_half * 2.0

        if stroke_px < 2.5:
            return FontWeight.THIN
        if stroke_px < 5.0:
            return FontWeight.REGULAR
        return FontWeight.BOLD

    # ── Italic detection ──────────────────────────────────────────────────────

    def _detect_italic(self, crop: np.ndarray,
                       skew_threshold_deg: float = 5.0) -> bool:
        """
        PCA on foreground pixel coordinates. If the first eigenvector deviates
        > skew_threshold_deg from vertical, flag italic.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if binary.mean() > 200:
            binary = cv2.bitwise_not(binary)

        ys, xs = np.where(binary > 0)
        if len(xs) < 20:
            return False

        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
        # Principal axis angle from horizontal
        angle_from_horiz = math.degrees(math.atan2(float(vt[0, 1]),
                                                    float(vt[0, 0])))
        # Convert to angle from vertical
        angle_from_vert = abs(90.0 - abs(angle_from_horiz))
        return angle_from_vert > skew_threshold_deg

    # ── Colour extraction ─────────────────────────────────────────────────────

    def _dominant_color(self, crop: np.ndarray) -> Tuple[int, int, int]:
        """
        K-Means (k=2) on the crop pixels. Returns the darker cluster centroid
        as the text (foreground) colour in BGR.
        """
        if crop.size == 0:
            return (0, 0, 0)
        pixels = crop.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 2:
            return tuple(int(v) for v in pixels[0])
        _, labels, centers = cv2.kmeans(
            pixels, 2,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            3,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        # Darker centroid = text colour (lower sum)
        c0_brightness = float(centers[0].sum())
        c1_brightness = float(centers[1].sum())
        text_center = centers[0] if c0_brightness < c1_brightness else centers[1]
        return tuple(int(v) for v in text_center)

    def _border_median(self, crop: np.ndarray,
                        border_px: int = 5) -> Tuple[int, int, int]:
        """
        Median colour of the border strip of the inpainted crop.
        This estimates the background colour at the region boundary.
        """
        if crop.size == 0 or crop.shape[0] < border_px * 2 or crop.shape[1] < border_px * 2:
            return (255, 255, 255)

        top    = crop[:border_px, :]
        bottom = crop[-border_px:, :]
        left   = crop[:, :border_px]
        right  = crop[:, -border_px:]
        border = np.concatenate([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ], axis=0).astype(np.float32)

        median = np.median(border, axis=0)
        return tuple(int(v) for v in median)


# ════════════════════════════════════════════════════════════════════════════════
# FONT SELECTOR — (FontClass, FontWeight, script) → font file path
# ════════════════════════════════════════════════════════════════════════════════

# Maps (FontClass.value, FontWeight.value, script_key) → filename
# script_key is derived from target_lang; default = "latin"
_FONT_MATRIX: dict[tuple, str] = {
    # ── SANS_SERIF ─────────────────────────────────────────────────────────────
    ("sans_serif", "regular", "latin"):      "NotoSans-Regular.ttf",
    ("sans_serif", "thin",    "latin"):      "NotoSans-Regular.ttf",     # no Thin variant
    ("sans_serif", "bold",    "latin"):      "NotoSans-Bold.ttf",
    ("sans_serif", "regular", "cyrillic"):   "NotoSans-Regular.ttf",
    ("sans_serif", "bold",    "cyrillic"):   "NotoSans-Bold.ttf",
    ("sans_serif", "regular", "devanagari"): "NotoSansDevanagari-Regular.ttf",
    ("sans_serif", "thin",    "devanagari"): "NotoSansDevanagari-Regular.ttf",
    ("sans_serif", "bold",    "devanagari"): "NotoSansDevanagari-Bold.ttf",
    ("sans_serif", "regular", "bengali"):    "NotoSansBengali-Regular.ttf",
    ("sans_serif", "bold",    "bengali"):    "NotoSansBengali-Bold.ttf",
    ("sans_serif", "regular", "tamil"):      "NotoSansTamil-Regular.ttf",
    ("sans_serif", "bold",    "tamil"):      "NotoSansTamil-Regular.ttf",
    ("sans_serif", "regular", "telugu"):     "NotoSansTelugu-Regular.ttf",
    ("sans_serif", "regular", "kannada"):    "NotoSansKannada-Regular.ttf",
    ("sans_serif", "regular", "malayalam"):  "NotoSansMalayalam-Regular.ttf",
    ("sans_serif", "bold",    "malayalam"):  "NotoSansMalayalam-Bold.ttf",
    ("sans_serif", "regular", "gujarati"):   "NotoSansGujarati-Regular.ttf",
    ("sans_serif", "regular", "gurmukhi"):   "NotoSansGurmukhi-Regular.ttf",
    ("sans_serif", "regular", "oriya"):      "NotoSansOriya-Regular.ttf",
    ("sans_serif", "regular", "sinhala"):    "NotoSansSinhala-Regular.ttf",
    ("sans_serif", "regular", "arabic"):     "NotoNaskhArabic-Regular.ttf",
    ("sans_serif", "bold",    "arabic"):     "NotoNaskhArabic-Bold.ttf",
    ("sans_serif", "regular", "hebrew"):     "NotoSansHebrew-Regular.ttf",
    ("sans_serif", "regular", "cjk"):        "NotoSansCJKsc-Regular.otf",
    # ── SERIF ──────────────────────────────────────────────────────────────────
    ("serif", "regular", "latin"):           "NotoSerif-Regular.ttf",
    ("serif", "thin",    "latin"):           "NotoSerif-Regular.ttf",
    ("serif", "bold",    "latin"):           "NotoSerif-Bold.ttf",
    ("serif", "regular", "devanagari"):      "NotoSerifDevanagari-Regular.ttf",
    ("serif", "bold",    "devanagari"):      "NotoSerifDevanagari-Regular.ttf",
    # Serif → fall back to sans for scripts without Serif Noto variant
    # ── MONOSPACE ──────────────────────────────────────────────────────────────
    ("monospace", "regular", "latin"):       "NotoSansMono-Regular.ttf",
    ("monospace", "bold",    "latin"):       "NotoSansMono-Bold.ttf",
    # Monospace has no Indic / CJK variants → falls back to sans_serif
    # ── DISPLAY / HANDWRITTEN → sans_serif fallback (per approved plan) ────────
}

# Language → script key
_LANG_TO_SCRIPT: dict[str, str] = {
    "hi": "devanagari", "mr": "devanagari", "ne": "devanagari",
    "sa": "devanagari", "mai": "devanagari", "doi": "devanagari",
    "kok": "devanagari", "brx": "devanagari",
    "bn": "bengali", "as": "bengali", "mni": "bengali",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "gu": "gujarati",
    "pa": "gurmukhi",
    "or": "oriya",
    "si": "sinhala",
    "ar": "arabic", "ur": "arabic", "fa": "arabic",
    "ks": "arabic", "sd": "arabic",
    "he": "hebrew",
    "zh": "cjk", "ja": "cjk", "ko": "cjk",
    "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic",
}


class FontSelector:
    """
    Maps (FontClass, FontWeight, target_lang) to the best available bundled font.

    Three-level fallback:
      Level 0: exact (class, weight, script) match
      Level 1: (SANS_SERIF, REGULAR, script) match
      Level 2: NotoSans-Regular.ttf (universal Latin fallback)
    """

    def select(self,
               attrs: FontAttrs,
               target_lang: str) -> Tuple[Path, int]:
        """
        Returns (font_path, fallback_level_used).
        fallback_level: 0 = exact, 1 = script-only, 2 = universal.
        """
        script = _LANG_TO_SCRIPT.get(target_lang.lower(), "latin")

        # Level 0: exact match
        key0 = (attrs.font_class.value, attrs.weight.value, script)
        fn = _FONT_MATRIX.get(key0)
        if fn:
            p = FONTS_DIR / fn
            if p.exists():
                return p, 0

        # Level 1: script-safe default (SANS_SERIF + REGULAR for this script)
        key1 = ("sans_serif", "regular", script)
        fn1 = _FONT_MATRIX.get(key1)
        if fn1:
            p1 = FONTS_DIR / fn1
            if p1.exists():
                log.debug("FontSelector L1 fallback: %s (wanted %s)", fn1, key0)
                return p1, 1

        # Level 2: universal fallback
        universal = FONTS_DIR / "NotoSans-Regular.ttf"
        if universal.exists():
            log.warning("FontSelector L2 fallback to NotoSans-Regular (wanted %s)", key0)
            return universal, 2

        raise FileNotFoundError(
            f"No Noto font found for {key0}. "
            "Run: python scripts/download_fonts.py"
        )


# ════════════════════════════════════════════════════════════════════════════════
# TEXT RENDERER — uharfbuzz + freetype-py + python-bidi + cv2
# ════════════════════════════════════════════════════════════════════════════════

def _detect_script_from_text(text: str) -> str:
    """Detect the primary script of a string from Unicode codepoints."""
    for ch in text:
        cp = ord(ch)
        for script, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                return script
    return "latin"


def _contrast_ratio(c1: Tuple[int, int, int],
                     c2: Tuple[int, int, int]) -> float:
    """Approximate WCAG relative luminance contrast ratio."""
    def lum(c):
        r, g, b = c[2] / 255, c[1] / 255, c[0] / 255  # BGR→RGB
        r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    l1, l2 = lum(c1), lum(c2)
    bright, dark = max(l1, l2), min(l1, l2)
    return (bright + 0.05) / (dark + 0.05)


class TextRenderer:
    """
    Renders translated text onto the inpainted image using:
      - uharfbuzz for OpenType text shaping
      - python-bidi for RTL visual reordering
      - freetype-py for glyph rasterization
      - cv2 for alpha compositing

    One instance can be reused across multiple requests (fonts cached).
    """

    def __init__(self) -> None:
        self._selector   = FontSelector()
        self._extractor  = FontAttributeExtractor()
        self._ft_cache: dict[str, object] = {}   # path → freetype.Face
        self._hb_cache:  dict[str, object] = {}  # path → (hb.Font, hb.Face)

    # ── Public API ────────────────────────────────────────────────────────────

    def render_all(self,
                   inpainted_bgr:  np.ndarray,
                   original_bgr:   np.ndarray,
                   ocr_results:    list,          # List[OcrResult]
                   translations:   list,          # List[TranslationResult]
                   target_lang:    str) -> np.ndarray:
        """
        Main entry point.

        Args:
            inpainted_bgr: The output of Stage 1–4 (text erased).
            original_bgr:  The original image BEFORE inpainting (for style extraction).
            ocr_results:   Stage 5 output — one per detection region.
            translations:  Stage 6 output — one per detection region (same order).
            target_lang:   BCP-47 target language code.

        Returns:
            A new BGR image with translated text rendered in-place.
            Regions where skip=True or engine='failed' are left erased.
        """
        output = inpainted_bgr.copy()
        h, w   = output.shape[:2]

        paired = list(zip(ocr_results, translations))
        log.info("[Stage 7] Rendering %d regions for lang=%s", len(paired), target_lang)

        for ocr, tr in paired:
            # Honour skip flags from previous stages
            if ocr.skip:
                log.debug("[Stage 7] Region %d: OCR skip — left erased", ocr.det_id)
                continue
            if tr.engine == "failed":
                log.debug("[Stage 7] Region %d: translation failed — left erased", ocr.det_id)
                continue
            if not tr.text or not tr.text.strip():
                continue

            # Clamp bbox to image bounds
            x  = max(0, min(ocr.x, w - 1))
            y  = max(0, min(ocr.y, h - 1))
            x2 = max(x + 1, min(ocr.x + ocr.w, w))
            y2 = max(y + 1, min(ocr.y + ocr.h, h))

            orig_crop = original_bgr[y:y2, x:x2]
            inp_crop  = inpainted_bgr[y:y2, x:x2]

            try:
                attrs = self._extractor.extract(
                    orig_crop, inp_crop,
                    bbox_h=y2 - y,
                    target_lang=target_lang,
                )
                output = self._render_region(
                    output, tr.text, attrs,
                    target_lang=target_lang,
                    bbox=(x, y, x2 - x, y2 - y),
                )
            except Exception as exc:
                log.error("[Stage 7] Region %d render failed: %s", ocr.det_id, exc)
                # Level 3 fallback: leave region erased, continue

        log.info("[Stage 7] Rendering complete.")
        return output

    # ── Per-region rendering ──────────────────────────────────────────────────

    def _render_region(self,
                       canvas:      np.ndarray,
                       text:        str,
                       attrs:       FontAttrs,
                       target_lang: str,
                       bbox:        Tuple[int, int, int, int]) -> np.ndarray:
        """Render a single translated text region onto the canvas."""
        x, y, bw, bh = bbox

        # ── Select font ───────────────────────────────────────────────────────
        try:
            font_path, _ = self._selector.select(attrs, target_lang)
        except FileNotFoundError as exc:
            log.warning("[Stage 7] Font not found: %s", exc)
            return canvas

        # ── BiDi reorder for RTL ──────────────────────────────────────────────
        display_text = text
        if attrs.is_rtl:
            try:
                from bidi.algorithm import get_display
                display_text = get_display(text)
            except Exception as exc:
                log.debug("BiDi failed: %s", exc)

        # ── Shape with HarfBuzz ───────────────────────────────────────────────
        try:
            glyphs, positions = self._shape_text(
                display_text, font_path, attrs.size_px)
        except Exception as exc:
            log.warning("[Stage 7] HarfBuzz shaping failed: %s", exc)
            glyphs, positions = None, None

        # ── Render glyphs with FreeType ───────────────────────────────────────
        if glyphs is not None:
            text_layer = self._rasterize_glyphs(
                glyphs, positions, font_path, attrs, bw, bh)
        else:
            # Fallback: cv2.putText (no shaping)
            text_layer = self._cv2_fallback_render(
                display_text, attrs, bw, bh)

        if text_layer is None:
            return canvas

        # ── Contrast check — add outline if contrast < 3:1 ───────────────────
        if _contrast_ratio(attrs.text_color, attrs.bg_color) < 3.0:
            # draw thin outline in bg colour to ensure readability
            outline_color = attrs.bg_color
            kernel = np.ones((3, 3), np.uint8)
            if text_layer.shape[2] == 4:
                alpha_ch = text_layer[:, :, 3]
                dilated  = cv2.dilate(alpha_ch, kernel, iterations=1)
                outline_alpha = cv2.subtract(dilated, alpha_ch)
                outline_layer = np.zeros_like(text_layer)
                outline_layer[:, :, :3] = outline_color
                outline_layer[:, :, 3]  = outline_alpha
                text_layer = self._composite_rgba(text_layer, outline_layer)

        # ── Composite onto canvas ─────────────────────────────────────────────
        th, tw = text_layer.shape[:2]
        cy2 = min(y + th, canvas.shape[0])
        cx2 = min(x + tw, canvas.shape[1])
        region = canvas[y:cy2, x:cx2]

        if text_layer.shape[2] == 4:
            alpha = text_layer[:cy2 - y, :cx2 - x, 3:4].astype(np.float32) / 255.0
            bgr   = text_layer[:cy2 - y, :cx2 - x, :3].astype(np.float32)
            region_f = region.astype(np.float32)
            blended  = bgr * alpha + region_f * (1.0 - alpha)
            canvas[y:cy2, x:cx2] = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            canvas[y:cy2, x:cx2] = text_layer[:cy2 - y, :cx2 - x]

        return canvas

    # ── HarfBuzz shaping ──────────────────────────────────────────────────────

    def _shape_text(self,
                    text:      str,
                    font_path: Path,
                    size_px:   int):
        """
        Shape text using uharfbuzz.
        Returns (glyph_infos, glyph_positions) tuples or raises on failure.
        """
        import uharfbuzz as hb

        key = str(font_path)
        if key not in self._hb_cache:
            blob = hb.Blob.from_file_path(str(font_path))
            hb_face = hb.Face(blob)
            hb_font = hb.Font(hb_face)
            # Set font scale in 26.6 fixed-point (size_px << 6)
            scale = size_px * 64
            hb_font.scale = (scale, scale)
            self._hb_cache[key] = (hb_font, hb_face)

        hb_font, _ = self._hb_cache[key]

        # Update scale for this request's size
        scale = size_px * 64
        hb_font.scale = (scale, scale)

        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()

        # OpenType features for complex scripts
        features = {
            "kern": True,
            "liga": True,   # standard ligatures
            "calt": True,   # contextual alternates
            "mark": True,   # mark positioning
            "mkmk": True,   # mark-to-mark
        }
        hb.shape(hb_font, buf, features)

        return buf.glyph_infos, buf.glyph_positions

    # ── FreeType glyph rasterization ─────────────────────────────────────────

    def _get_ft_face(self, font_path: Path, size_px: int):
        """Load (and cache) a FreeType face, set pixel size."""
        import freetype
        key = str(font_path)
        if key not in self._ft_cache:
            self._ft_cache[key] = freetype.Face(str(font_path))
        face = self._ft_cache[key]
        face.set_pixel_sizes(0, size_px)
        return face

    def _rasterize_glyphs(self,
                           glyph_infos,
                           glyph_positions,
                           font_path:  Path,
                           attrs:      FontAttrs,
                           max_width:  int,
                           bbox_h:     int) -> Optional[np.ndarray]:
        """
        Rasterize HarfBuzz-shaped glyph sequence using FreeType.
        Returns BGRA NumPy array sized (bbox_h, max_width, 4).
        """
        import freetype

        size_px = attrs.size_px
        face = self._get_ft_face(font_path, size_px)

        # Compute total advance width for adaptive scaling
        total_advance = sum(pos.x_advance for pos in glyph_positions) >> 6  # 26.6→px
        if total_advance == 0:
            return None

        # Adaptive scale-down if text is wider than bbox
        scale = 1.0
        if total_advance > max_width and total_advance > 0:
            scale = max_width / total_advance
            new_size = max(6, int(size_px * scale))
            face.set_pixel_sizes(0, new_size)
            # Re-shape is expensive — approximate by rescaling positions
            glyph_positions_scaled = [
                type("P", (), {
                    "x_advance": int(p.x_advance * scale),
                    "y_advance": int(p.y_advance * scale),
                    "x_offset":  int(p.x_offset  * scale),
                    "y_offset":  int(p.y_offset  * scale),
                })()
                for p in glyph_positions
            ]
            effective_size = new_size
        else:
            glyph_positions_scaled = list(glyph_positions)
            effective_size = size_px

        # Canvas
        layer = np.zeros((bbox_h, max_width, 4), dtype=np.uint8)

        # Baseline position: ~80% down from top
        baseline_y = int(bbox_h * 0.80)

        # RTL: start from right
        if attrs.is_rtl:
            total_w = sum(p.x_advance for p in glyph_positions_scaled) >> 6
            pen_x = max(0, max_width - total_w)
        else:
            pen_x = 0

        pen_y = baseline_y

        tc_bgr = attrs.text_color  # already BGR

        for info, pos in zip(glyph_infos, glyph_positions_scaled):
            gid = info.codepoint
            try:
                face.load_glyph(gid, freetype.FT_LOAD_RENDER)
            except Exception:
                continue

            bm = face.glyph.bitmap
            if bm.width == 0 or bm.rows == 0:
                pen_x += pos.x_advance >> 6
                continue

            gx = pen_x + face.glyph.bitmap_left + (pos.x_offset >> 6)
            gy = pen_y - face.glyph.bitmap_top  + (pos.y_offset >> 6)

            # Clamp to layer bounds
            x0  = max(0, gx);    x1  = min(max_width, gx + bm.width)
            y0  = max(0, gy);    y1  = min(bbox_h,    gy + bm.rows)
            bx0 = x0 - gx;       by0 = y0 - gy

            if x0 >= x1 or y0 >= y1:
                pen_x += pos.x_advance >> 6
                continue

            glyph_alpha = np.frombuffer(bm.buffer, dtype=np.uint8
                          ).reshape(bm.rows, bm.width)[by0:by0 + (y1 - y0),
                                                       bx0:bx0 + (x1 - x0)]

            layer[y0:y1, x0:x1, 0] = tc_bgr[0]
            layer[y0:y1, x0:x1, 1] = tc_bgr[1]
            layer[y0:y1, x0:x1, 2] = tc_bgr[2]
            # Accumulate alpha (in case glyphs overlap slightly)
            existing = layer[y0:y1, x0:x1, 3].astype(np.uint16)
            new_alpha = np.minimum(255,
                        existing + glyph_alpha.astype(np.uint16)).astype(np.uint8)
            layer[y0:y1, x0:x1, 3] = new_alpha

            pen_x += pos.x_advance >> 6
            pen_y -= pos.y_advance >> 6

        return layer

    # ── cv2 fallback render (no HarfBuzz) ────────────────────────────────────

    def _cv2_fallback_render(self,
                              text:   str,
                              attrs:  FontAttrs,
                              bw:     int,
                              bh:     int) -> Optional[np.ndarray]:
        """
        Last-resort renderer using cv2.putText. No shaping — Latin-only quality.
        Used when uharfbuzz fails or font is not a TTF.
        """
        layer = np.zeros((bh, bw, 4), dtype=np.uint8)
        scale = attrs.size_px / 30.0
        thickness = max(1, int(scale))
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), baseline = cv2.getTextSize(text, font_face, scale, thickness)
        x = max(0, (bw - tw) // 2)
        y = max(th, int(bh * 0.75))

        temp = np.zeros((bh, bw, 3), dtype=np.uint8)
        cv2.putText(temp, text, (x, y), font_face, scale,
                    attrs.text_color, thickness, cv2.LINE_AA)

        # Convert RGB content to BGRA with alpha from text pixels
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        layer[:, :, :3] = temp
        layer[:, :, 3]  = gray  # use brightness as alpha
        return layer

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _composite_rgba(front: np.ndarray,
                         back:  np.ndarray) -> np.ndarray:
        """Alpha-composite `back` under `front` (both BGRA)."""
        fa = front[:, :, 3:4].astype(np.float32) / 255.0
        ba = back[ :, :, 3:4].astype(np.float32) / 255.0
        out_a = fa + ba * (1.0 - fa)
        out_bgr = (front[:, :, :3].astype(np.float32) * fa +
                   back[ :, :, :3].astype(np.float32) * ba * (1.0 - fa))
        out_a_safe = np.where(out_a > 0, out_a, 1.0)
        out_bgr /= out_a_safe
        result = np.zeros_like(front)
        result[:, :, :3] = np.clip(out_bgr, 0, 255).astype(np.uint8)
        result[:, :, 3]  = np.clip(out_a * 255, 0, 255).astype(np.uint8)
        return result
