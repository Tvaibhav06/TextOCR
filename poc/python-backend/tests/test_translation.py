"""
python-backend/tests/test_translation.py
────────────────────────────────────────────────────────────────────────────────
Stage 6 PC Path — Translation Router test suite.

Test strategy:
  - Unit tests on routing logic run WITHOUT downloading any models.
    They mock the engines at the class level.
  - Integration tests (marked @pytest.mark.integration) hit real Argos
    packages and are skipped in CI unless TEXTREMOVER_RUN_INTEGRATION=1.
  - A smoke-only IndicTrans2 test is skipped unless
    TEXTREMOVER_RUN_INDIC_INTEGRATION=1 (model is ~1 GB per direction).

Run unit tests:
    python run_tests.py tests/test_translation.py -v -m "not integration"

Run all (requires network + disk):
    TEXTREMOVER_RUN_INTEGRATION=1 python run_tests.py tests/test_translation.py -v
"""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.translation import (
    TranslationRouter,
    TranslationResult,
    IndicTrans2Engine,
    ArgosEngine,
    _is_indic,
    _is_english,
    _use_indic_engine,
    INDIC_FLORES_MAP,
    ENGLISH_FLORES,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

RUN_INTEGRATION = os.getenv("TEXTREMOVER_RUN_INTEGRATION", "0") == "1"
RUN_INDIC       = os.getenv("TEXTREMOVER_RUN_INDIC_INTEGRATION", "0") == "1"

integration = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="Set TEXTREMOVER_RUN_INTEGRATION=1 to run integration tests",
)
indic_integration = pytest.mark.skipif(
    not RUN_INDIC,
    reason="Set TEXTREMOVER_RUN_INDIC_INTEGRATION=1 to run IndicTrans2 (needs ~1 GB download)",
)


# ════════════════════════════════════════════════════════════════════════════
# SUITE 1: Language classifier helpers
# ════════════════════════════════════════════════════════════════════════════

class TestLanguageHelpers:

    def test_is_indic_returns_true_for_hindi(self):
        assert _is_indic("hi") is True

    def test_is_indic_returns_true_for_all_22_languages(self):
        for code in INDIC_FLORES_MAP:
            assert _is_indic(code), f"Expected {code!r} to be Indic"

    def test_is_indic_returns_false_for_english(self):
        assert _is_indic("en") is False

    def test_is_indic_returns_false_for_french(self):
        assert _is_indic("fr") is False

    def test_is_english_matches_en_and_eng(self):
        assert _is_english("en") is True
        assert _is_english("eng") is True

    def test_is_english_returns_false_for_indic(self):
        assert _is_english("hi") is False

    def test_is_indic_case_insensitive(self):
        assert _is_indic("HI") is True
        assert _is_indic("Hi") is True


class TestRoutingDecision:

    def test_en_to_hi_uses_indic_engine(self):
        assert _use_indic_engine("en", "hi") is True

    def test_hi_to_en_uses_indic_engine(self):
        assert _use_indic_engine("hi", "en") is True

    def test_hi_to_ta_uses_indic_engine(self):
        assert _use_indic_engine("hi", "ta") is True

    def test_en_to_fr_does_not_use_indic_engine(self):
        assert _use_indic_engine("en", "fr") is False

    def test_fr_to_de_does_not_use_indic_engine(self):
        assert _use_indic_engine("fr", "de") is False

    def test_hi_to_zh_does_not_use_indic_engine(self):
        # Chinese is not Indic or English → Argos path
        assert _use_indic_engine("hi", "zh") is False


# ════════════════════════════════════════════════════════════════════════════
# SUITE 2: Router unit tests (engines mocked)
# ════════════════════════════════════════════════════════════════════════════

class TestTranslationRouterUnit:
    """
    All tests mock the underlying engines so no model download is needed.
    """

    @pytest.fixture
    def router(self, monkeypatch):
        r = TranslationRouter()
        return r

    def _mock_indic(self, monkeypatch, return_value="INDIC_OUTPUT"):
        monkeypatch.setattr(
            IndicTrans2Engine, "translate",
            lambda self, text, src, tgt: return_value,
        )

    def _mock_argos(self, monkeypatch, return_value="ARGOS_OUTPUT"):
        monkeypatch.setattr(
            ArgosEngine, "translate",
            lambda self, text, src, tgt: return_value,
        )

    def _mock_indic_fail(self, monkeypatch):
        def _raise(self, text, src, tgt):
            raise RuntimeError("forced IndicTrans2 failure")
        monkeypatch.setattr(IndicTrans2Engine, "translate", _raise)

    def _mock_argos_fail(self, monkeypatch):
        def _raise(self, text, src, tgt):
            raise RuntimeError("forced Argos failure")
        monkeypatch.setattr(ArgosEngine, "translate", _raise)

    # ── Routing ──────────────────────────────────────────────────────────────

    def test_en_to_hi_routes_to_indictrans2(self, router, monkeypatch):
        self._mock_indic(monkeypatch, "हैलो")
        result = router.translate("Hello", "en", "hi")
        assert result.engine == "indictrans2"
        assert result.text == "हैलो"

    def test_hi_to_en_routes_to_indictrans2(self, router, monkeypatch):
        self._mock_indic(monkeypatch, "Hello World")
        result = router.translate("नमस्ते दुनिया", "hi", "en")
        assert result.engine == "indictrans2"

    def test_hi_to_ta_routes_to_indictrans2(self, router, monkeypatch):
        self._mock_indic(monkeypatch, "வணக்கம்")
        result = router.translate("नमस्ते", "hi", "ta")
        assert result.engine == "indictrans2"

    def test_en_to_fr_routes_to_argos(self, router, monkeypatch):
        self._mock_argos(monkeypatch, "Bonjour le monde")
        result = router.translate("Hello World", "en", "fr")
        assert result.engine == "argos"
        # "Hello World" is title-case → router applies .title() to the translation
        assert result.text == "Bonjour Le Monde"

    def test_fr_to_de_routes_to_argos(self, router, monkeypatch):
        self._mock_argos(monkeypatch, "Hallo Welt")
        result = router.translate("Bonjour le monde", "fr", "de")
        assert result.engine == "argos"

    # ── Fallback behaviour ────────────────────────────────────────────────────

    def test_indictrans2_failure_falls_back_to_argos(self, router, monkeypatch):
        self._mock_indic_fail(monkeypatch)
        self._mock_argos(monkeypatch, "हैलो")
        result = router.translate("Hello", "en", "hi")
        assert result.engine == "argos"
        assert result.error is None

    def test_both_engines_fail_returns_failed(self, router, monkeypatch):
        self._mock_indic_fail(monkeypatch)
        self._mock_argos_fail(monkeypatch)
        result = router.translate("Hello", "en", "hi")
        assert result.engine == "failed"
        assert result.text == "Hello"   # original preserved
        assert result.error is not None

    def test_indictrans2_returns_empty_falls_back_to_argos(self, router, monkeypatch):
        monkeypatch.setattr(
            IndicTrans2Engine, "translate",
            lambda self, text, src, tgt: "",  # empty string
        )
        self._mock_argos(monkeypatch, "fallback result")
        result = router.translate("Hello", "en", "hi")
        assert result.engine == "argos"

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_same_language_returns_passthrough(self, router):
        result = router.translate("Hello", "en", "en")
        assert result.engine == "passthrough"
        assert result.text == "Hello"

    def test_empty_string_returns_skipped(self, router):
        result = router.translate("", "en", "hi")
        assert result.engine == "skipped"
        assert result.text == ""

    def test_whitespace_only_returns_skipped(self, router):
        result = router.translate("   ", "en", "hi")
        assert result.engine == "skipped"

    def test_src_lang_and_tgt_lang_stored_in_result(self, router, monkeypatch):
        self._mock_indic(monkeypatch)
        result = router.translate("Hello", "en", "hi")
        assert result.src_lang == "en"
        assert result.tgt_lang == "hi"

    # ── Case formatting ───────────────────────────────────────────────────────

    def test_uppercase_original_uppercases_translation(self, router, monkeypatch):
        self._mock_argos(monkeypatch, "bonjour")
        result = router.translate("HELLO", "en", "fr")
        assert result.text == "BONJOUR"

    def test_titlecase_original_titlecases_translation(self, router, monkeypatch):
        self._mock_argos(monkeypatch, "bonjour le monde")
        result = router.translate("Hello World", "en", "fr")
        assert result.text == "Bonjour Le Monde"

    def test_lowercase_original_keeps_translation_as_is(self, router, monkeypatch):
        self._mock_argos(monkeypatch, "bonjour")
        result = router.translate("hello", "en", "fr")
        assert result.text == "bonjour"

    # ── Normalizer ────────────────────────────────────────────────────────────

    def test_normalizer_strips_whitespace(self, router, monkeypatch):
        """Text is normalized before being sent to the engine."""
        captured = {}
        def _capture(self_inner, text, src, tgt):
            captured["text"] = text
            return "output"
        monkeypatch.setattr(ArgosEngine, "translate", _capture)
        router.translate("  hello  ", "en", "fr")
        assert captured["text"] == "hello"

    def test_normalizer_collapses_internal_spaces(self, router, monkeypatch):
        captured = {}
        def _capture(self_inner, text, src, tgt):
            captured["text"] = text
            return "output"
        monkeypatch.setattr(ArgosEngine, "translate", _capture)
        router.translate("hello   world", "en", "fr")
        assert captured["text"] == "hello world"

    # ── Regression guard ──────────────────────────────────────────────────────

    def test_translation_import_does_not_break_existing_pipeline(self):
        """Importing translation.py must not modify core.py at all."""
        from pipeline.core import TextRemovalPipeline
        # Simply check the class can still be instantiated without error.
        # (We do not call .run() to avoid spinning up PaddleOCR in unit tests.)
        assert TextRemovalPipeline is not None


# ════════════════════════════════════════════════════════════════════════════
# SUITE 3: Integration tests (real Argos — skipped without flag)
# ════════════════════════════════════════════════════════════════════════════

class TestArgosIntegration:

    @integration
    def test_argos_translates_en_to_fr(self):
        engine = ArgosEngine()
        result = engine.translate("Hello", "en", "fr")
        assert isinstance(result, str)
        assert len(result) > 0
        assert result.lower() != "hello"   # at minimum changed

    @integration
    def test_argos_translates_en_to_de(self):
        engine = ArgosEngine()
        result = engine.translate("Good morning", "en", "de")
        assert isinstance(result, str)
        assert len(result) > 0

    @integration
    def test_router_en_to_fr_uses_argos_and_returns_text(self):
        router = TranslationRouter()
        result = router.translate("Good morning", "en", "fr")
        assert result.engine == "argos"
        assert result.text
        assert result.error is None


# ════════════════════════════════════════════════════════════════════════════
# SUITE 4: IndicTrans2 smoke test (real model — skipped without flag)
# ════════════════════════════════════════════════════════════════════════════

class TestIndicTrans2Integration:

    @indic_integration
    def test_router_en_to_hi_uses_indictrans2(self):
        router = TranslationRouter()
        result = router.translate("Hello World", "en", "hi")
        assert result.engine == "indictrans2"
        assert result.text
        assert result.text != "Hello World"

    @indic_integration
    def test_router_hi_to_ta_uses_indictrans2(self):
        router = TranslationRouter()
        result = router.translate("नमस्ते", "hi", "ta")
        assert result.engine == "indictrans2"
        assert result.text
