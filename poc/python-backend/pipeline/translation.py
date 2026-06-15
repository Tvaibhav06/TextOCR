"""
python-backend/pipeline/translation.py
────────────────────────────────────────────────────────────────────────────────
Phase 3 — Stage 6 (PC Path): Translation Router

Design rules:
  - This module is ADDITIVE.  It does not modify core.py or ocr.py.
  - Routing is deterministic (see ROUTING TABLE below).
  - Engines are lazy-loaded on first use.
  - If the primary engine fails, we fall back cleanly to the other.
  - If both fail, we return TranslationResult(engine='failed') with the
    original text preserved, so Stage 7 can decide to skip that region.

ROUTING TABLE
─────────────────────────────────────────────────────────────────────────────
 Source        Target        Primary engine    Fallback
─────────────────────────────────────────────────────────────────────────────
 Indic         English       IndicTrans2       Argos Translate
 English       Indic         IndicTrans2       Argos Translate
 Indic         Indic         IndicTrans2       Argos Translate
 anything else             Argos Translate    (no further fallback; flag error)
─────────────────────────────────────────────────────────────────────────────

Usage:
    from pipeline.translation import TranslationRouter, TranslationResult
    router = TranslationRouter()
    result = router.translate("Hello World", src="en", tgt="hi")
    # result.text   → "हैलो वर्ल्ड"
    # result.engine → "indictrans2"
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger("textremover.translation")

# ── Indic language → Flores-200 code map (IndicTrans2 requirement) ─────────────
# Full 22 official Indian languages + commonly requested neighbours
INDIC_FLORES_MAP: Dict[str, str] = {
    "hi":  "hin_Deva",   # Hindi
    "bn":  "ben_Beng",   # Bengali
    "ta":  "tam_Taml",   # Tamil
    "te":  "tel_Telu",   # Telugu
    "mr":  "mar_Deva",   # Marathi
    "gu":  "guj_Gujr",   # Gujarati
    "kn":  "kan_Knda",   # Kannada
    "ml":  "mal_Mlym",   # Malayalam
    "pa":  "pan_Guru",   # Punjabi
    "ur":  "urd_Arab",   # Urdu
    "or":  "ory_Orya",   # Odia
    "as":  "asm_Beng",   # Assamese
    "sa":  "san_Deva",   # Sanskrit
    "si":  "sin_Sinh",   # Sinhala
    "ne":  "npi_Deva",   # Nepali
    "mai": "mai_Deva",   # Maithili
    "doi": "doi_Deva",   # Dogri
    "mni": "mni_Beng",   # Manipuri (Meitei)
    "sat": "sat_Olck",   # Santali
    "kok": "kok_Deva",   # Konkani
    "ks":  "kas_Arab",   # Kashmiri
    "brx": "brx_Deva",   # Bodo
    "sd":  "snd_Arab",   # Sindhi
}
ENGLISH_FLORES = "eng_Latn"


def _is_indic(lang: str) -> bool:
    return lang.lower() in INDIC_FLORES_MAP


def _is_english(lang: str) -> bool:
    return lang.lower() in ("en", "eng")


def _use_indic_engine(src: str, tgt: str) -> bool:
    """Return True iff at least one side is Indic and the other is Indic or English."""
    s, t = src.lower(), tgt.lower()
    either_indic = _is_indic(s) or _is_indic(t)
    both_valid   = (_is_indic(s) or _is_english(s)) and (_is_indic(t) or _is_english(t))
    return either_indic and both_valid


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    """Output of a single translation call."""
    text:     str             # Translated text (or original if engine='failed')
    engine:   str             # 'indictrans2' | 'argos' | 'failed'
    src_lang: str = ""
    tgt_lang: str = ""
    error:    Optional[str] = None


# ── IndicTrans2 Engine ────────────────────────────────────────────────────────

class IndicTrans2Engine:
    """
    Wraps ai4bharat/IndicTransToolkit.

    Three model variants (loaded lazily per direction):
      • indictrans2-en-indic-1B   (EN → Indic)
      • indictrans2-indic-en-1B   (Indic → EN)
      • indictrans2-indic-indic-1B (Indic → Indic)

    Cache directory is controlled by the INDICTRANS2_CACHE_DIR env var.
    Defaults to ~/.cache/indictrans2 .
    """

    def __init__(self) -> None:
        self._models:     Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._ip = None   # IndicProcessor singleton (stateless)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _cache_dir(self) -> str:
        return os.getenv(
            "INDICTRANS2_CACHE_DIR",
            os.path.expanduser("~/.cache/indictrans2"),
        )

    def _direction(self, src: str, tgt: str) -> str:
        s, t = src.lower(), tgt.lower()
        if _is_indic(s) and _is_indic(t):
            return "indic-indic"
        if _is_english(s):
            return "en-indic"
        return "indic-en"

    def _load_model(self, direction: str) -> None:
        if direction in self._models:
            return
        model_name = f"ai4bharat/indictrans2-{direction}-1B"
        cache = self._cache_dir()
        log.info("[IndicTrans2] Loading model '%s' (cache: %s) …", model_name, cache)
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. "
                "Run: pip install transformers>=4.40.0 sentencepiece sacremoses"
            ) from exc
        self._tokenizers[direction] = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache, trust_remote_code=True
        )
        self._models[direction] = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=cache, trust_remote_code=True
        )
        log.info("[IndicTrans2] Model '%s' loaded.", direction)

    def _get_processor(self):
        if self._ip is None:
            try:
                from IndicTransToolkit import IndicProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "IndicTransToolkit is not installed. "
                    "Run: pip install git+https://github.com/AI4Bharat/IndicTransToolkit.git"
                ) from exc
            self._ip = IndicProcessor(inference=True)
        return self._ip

    # ── Public API ────────────────────────────────────────────────────────────

    def translate(self, text: str, src: str, tgt: str) -> str:
        """Translate text from src to tgt. Returns translated string."""
        s, t = src.lower(), tgt.lower()
        direction = self._direction(s, t)
        self._load_model(direction)

        src_flores = INDIC_FLORES_MAP.get(s, ENGLISH_FLORES)
        tgt_flores = INDIC_FLORES_MAP.get(t, ENGLISH_FLORES)

        ip = self._get_processor()
        batch = ip.preprocess_batch([text], src_lang=src_flores, tgt_lang=tgt_flores)
        tok = self._tokenizers[direction]
        tokens = tok(batch, return_tensors="pt", padding=True, truncation=True,
                     max_length=512)
        outputs = self._models[direction].generate(
            **tokens, num_beams=4, max_length=256, early_stopping=True
        )
        decoded = tok.batch_decode(outputs, skip_special_tokens=True)
        translated = ip.postprocess_batch(decoded, lang=tgt_flores)
        return translated[0] if translated else ""


# ── Argos Translate Engine ────────────────────────────────────────────────────

class ArgosEngine:
    """
    Wraps argostranslate for all non-Indic pairs.

    Language packages are downloaded and cached to disk on first use.
    Set ARGOS_PACKAGES_DIR to override the cache location.
    """

    def __init__(self) -> None:
        self._loaded = False

    def _ensure_argos(self):
        if self._loaded:
            return
        try:
            import argostranslate.package    # noqa: F401
            import argostranslate.translate  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "argostranslate is not installed. Run: pip install argostranslate>=1.9.1"
            ) from exc
        # Allow overriding package dir
        pkg_dir = os.getenv("ARGOS_PACKAGES_DIR")
        if pkg_dir:
            import argostranslate.settings as _s
            _s.data_dir = pkg_dir
        self._loaded = True

    def _ensure_package(self, src: str, tgt: str) -> None:
        import argostranslate.package as _pkg
        installed = _pkg.get_installed_packages()
        for p in installed:
            if p.from_code == src and p.to_code == tgt:
                return   # already cached
        log.info("[Argos] Package %s→%s not found; downloading …", src, tgt)
        _pkg.update_package_index()
        available = _pkg.get_available_packages()
        pkg = next(
            (p for p in available if p.from_code == src and p.to_code == tgt),
            None,
        )
        if pkg is None:
            raise ValueError(
                f"No Argos Translate package available for {src}→{tgt}. "
                "Check https://www.argosopentech.com/argospm/index/ for supported pairs."
            )
        log.info("[Argos] Downloading package %s→%s …", src, tgt)
        import argostranslate.package as _pkg2
        _pkg2.install_from_path(pkg.download())
        log.info("[Argos] Package %s→%s installed.", src, tgt)

    def translate(self, text: str, src: str, tgt: str) -> str:
        self._ensure_argos()
        self._ensure_package(src, tgt)
        import argostranslate.translate as _tr
        return _tr.translate(text, src, tgt)


# ── Translation Router ────────────────────────────────────────────────────────

class TranslationRouter:
    """
    Deterministic routing between IndicTrans2 and Argos Translate.

    Both engines are lazy-loaded on first use.  The router can be
    instantiated once per worker process and reused across requests.
    """

    def __init__(self) -> None:
        self._indic = IndicTrans2Engine()
        self._argos = ArgosEngine()

    # ── Public API ────────────────────────────────────────────────────────────

    def translate(self,
                  text: str,
                  src:  str,
                  tgt:  str) -> TranslationResult:
        """
        Translate text from src to tgt.

        Returns a TranslationResult with:
          .text   — translated string (or original if engine='failed')
          .engine — which engine was used
          .error  — error message if both engines failed
        """
        text = self._normalize(text)
        if not text:
            return TranslationResult(text="", engine="skipped",
                                     src_lang=src, tgt_lang=tgt)

        if src.lower() == tgt.lower():
            # No translation needed — same language
            return TranslationResult(text=text, engine="passthrough",
                                     src_lang=src, tgt_lang=tgt)

        # ── Attempt primary engine ────────────────────────────────────────────
        if _use_indic_engine(src, tgt):
            try:
                result = self._indic.translate(text, src, tgt)
                if result and result.strip():
                    return TranslationResult(
                        text=self._format_case(result, text),
                        engine="indictrans2",
                        src_lang=src,
                        tgt_lang=tgt,
                    )
                log.warning(
                    "[Router] IndicTrans2 returned empty for %s→%s; falling back to Argos.",
                    src, tgt,
                )
            except Exception as exc:
                log.warning(
                    "[Router] IndicTrans2 failed for %s→%s (%s); falling back to Argos.",
                    src, tgt, exc,
                )

        # ── Argos: primary for non-Indic pairs, fallback for Indic ───────────
        try:
            result = self._argos.translate(text, src, tgt)
            return TranslationResult(
                text=self._format_case(result, text),
                engine="argos",
                src_lang=src,
                tgt_lang=tgt,
            )
        except Exception as exc:
            log.error(
                "[Router] Both engines failed for %s→%s: %s", src, tgt, exc
            )
            return TranslationResult(
                text=text,   # preserve original so Stage 7 can skip gracefully
                engine="failed",
                src_lang=src,
                tgt_lang=tgt,
                error=str(exc),
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """Basic OCR noise cleanup before translation."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        # OCR sometimes confuses '|', 'l', '1' with 'I' before uppercase letters
        text = re.sub(r"[|l1](?=[A-Z])", "I", text)
        return text

    @staticmethod
    def _format_case(translated: str, original: str) -> str:
        """Mirror the original's casing style on the translation."""
        if original.isupper():
            return translated.upper()
        if original.istitle():
            return translated.title()
        return translated
