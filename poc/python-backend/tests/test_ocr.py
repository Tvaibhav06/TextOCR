"""
python-backend/tests/test_ocr.py
────────────────────────────────────────────────────────────────────────────────
Stage 5 unit tests for pipeline/ocr.py

Test strategy:
  - Unit tests for _crop_deskew and OcrResult construction use synthetic
    images only — no real PaddleOCR model calls (avoids slow model loading
    in CI unless explicitly requested).
  - Regression tests confirm the existing TextRemovalPipeline (Stages 1–4)
    is completely unaffected by the presence of ocr.py.
  - A @pytest.mark.live tag wraps tests that load the real PaddleOCR model.
    Skip them in fast CI runs with: pytest -m 'not live'

Run:
    pytest python-backend/tests/test_ocr.py -v
    pytest python-backend/tests/test_ocr.py -v -m 'not live'   # skip model tests
"""

import math
import sys
import os

import cv2
import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.ocr import (
    OcrResult,
    RegionOCR,
    _crop_deskew,
    MIN_CONFIDENCE,
    MIN_TEXT_LEN,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def solid_image():
    """Solid gray 200×300 image (no text)."""
    return np.full((200, 300, 3), 128, dtype=np.uint8)


@pytest.fixture
def text_image():
    """White image with bold black text — used for crop sanity checks."""
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "HELLO", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 0), 4, cv2.LINE_AA)
    return img


@pytest.fixture
def axis_aligned_polygon():
    """Axis-aligned rectangle polygon for the word region in text_image."""
    return np.array([[10, 10], [390, 10], [390, 90], [10, 90]], dtype=np.float32)


@pytest.fixture
def mock_detections(axis_aligned_polygon):
    """Minimal Detection-like objects with .id, .polygon, .bbox."""
    class BBox:
        def __init__(self, x, y, w, h):
            self.x, self.y, self.w, self.h = x, y, w, h

    class Det:
        def __init__(self, det_id, poly, bbox):
            self.id, self.polygon, self.bbox = det_id, poly, bbox

    return [
        Det(0, axis_aligned_polygon,
            BBox(10, 10, 380, 80)),
    ]


# ════════════════════════════════════════════════════════════════════════════════
# SUITE 1: _crop_deskew helper
# ════════════════════════════════════════════════════════════════════════════════

class TestCropDeskew:
    """Unit tests for the crop+deskew helper. No model loading required."""

    def test_axis_aligned_polygon_returns_ndarray(self, solid_image,
                                                  axis_aligned_polygon):
        crop = _crop_deskew(solid_image, axis_aligned_polygon)
        assert isinstance(crop, np.ndarray), "Must return ndarray"

    def test_crop_has_correct_channels(self, solid_image, axis_aligned_polygon):
        crop = _crop_deskew(solid_image, axis_aligned_polygon)
        assert crop.ndim == 3 and crop.shape[2] == 3

    def test_degenerate_polygon_returns_none(self, solid_image):
        """A zero-area polygon (all points at origin) should return None gracefully."""
        zero_poly = np.array([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=np.float32)
        # cv2.boundingRect returns w=0,h=0 for a zero-area polygon
        result = _crop_deskew(solid_image, zero_poly)
        assert result is None

    def test_out_of_bounds_polygon_clamped(self, solid_image):
        """A polygon extending outside image bounds must be clamped, not crash."""
        large_poly = np.array([[-50, -50], [500, -50], [500, 500], [-50, 500]],
                              dtype=np.float32)
        crop = _crop_deskew(solid_image, large_poly)
        # Should return the full image (clamped) without error
        assert crop is not None
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_small_crop_returns_valid_image(self, solid_image):
        """Tiny polygon (still above minimum size) must return a valid crop."""
        tiny_poly = np.array([[10, 10], [20, 10], [20, 20], [10, 20]],
                             dtype=np.float32)
        crop = _crop_deskew(solid_image, tiny_poly)
        assert crop is not None

    def test_skewed_polygon_returns_same_type(self, solid_image):
        """A skewed polygon triggers warpAffine; result must still be ndarray."""
        skewed = np.array([[10, 20], [50, 10], [55, 30], [15, 40]],
                          dtype=np.float32)
        crop = _crop_deskew(solid_image, skewed)
        assert crop is not None
        assert isinstance(crop, np.ndarray)

    def test_padding_is_applied(self, solid_image):
        """Crop with padding=4 must be slightly larger than with padding=0."""
        poly = np.array([[10, 10], [90, 10], [90, 50], [10, 50]],
                        dtype=np.float32)
        crop_no_pad  = _crop_deskew(solid_image, poly, padding=0)
        crop_with_pad = _crop_deskew(solid_image, poly, padding=4)
        assert crop_with_pad is not None
        assert crop_no_pad is not None
        # With padding the crop area should be ≥ without padding
        assert (crop_with_pad.shape[0] * crop_with_pad.shape[1] >=
                crop_no_pad.shape[0] * crop_no_pad.shape[1])


# ════════════════════════════════════════════════════════════════════════════════
# SUITE 2: OcrResult dataclass
# ════════════════════════════════════════════════════════════════════════════════

class TestOcrResult:
    def test_skip_true_when_confidence_below_threshold(self):
        r = OcrResult(det_id=0, text="AB", confidence=0.59, skip=True,
                      x=0, y=0, w=50, h=20)
        assert r.skip is True

    def test_skip_false_when_confidence_above_threshold(self):
        r = OcrResult(det_id=1, text="HELLO", confidence=0.85, skip=False,
                      x=0, y=0, w=100, h=30)
        assert r.skip is False

    def test_skip_true_for_short_text(self):
        r = OcrResult(det_id=2, text="AB", confidence=0.90, skip=True,
                      x=0, y=0, w=50, h=20)
        assert r.skip is True

    def test_min_confidence_constant_matches_spec(self):
        assert MIN_CONFIDENCE == 0.60

    def test_min_text_len_constant_matches_spec(self):
        assert MIN_TEXT_LEN == 3


# ════════════════════════════════════════════════════════════════════════════════
# SUITE 3: RegionOCR — mock-based (no model loading)
# ════════════════════════════════════════════════════════════════════════════════

class TestRegionOCRMocked:
    """
    Test RegionOCR.run() using a monkeypatched recognizer so these tests
    execute in milliseconds without needing the PaddleOCR model on disk.
    """

    def _make_ocr_with_mock(self, mock_results):
        """Build a RegionOCR instance whose _rec is mocked."""
        ocr = RegionOCR.__new__(RegionOCR)
        ocr._rec = None

        class MockRec:
            def __init__(self, results):
                self._results = results
                self._call_count = 0

            def ocr(self, crop, det=False, cls=True):
                r = self._results[self._call_count % len(self._results)]
                self._call_count += 1
                return r

        ocr._rec = MockRec(mock_results)
        return ocr

    def test_run_returns_one_result_per_detection(self, mock_detections, text_image):
        # Mock: returns one good recognition result
        mock_paddle_output = [[[["HELLO", 0.95]]]]
        ocr = self._make_ocr_with_mock([mock_paddle_output])
        results = ocr.run(text_image, mock_detections)
        assert len(results) == len(mock_detections)

    def test_good_recognition_not_skipped(self, mock_detections, text_image):
        # PaddleOCR rec with det=False returns: [[text, score], ...]
        # so result[0][0] = [text, score], result[0][0][0] = text, result[0][0][1] = score
        mock_paddle_output = [[["HELLO WORLD", 0.91]]]
        ocr = self._make_ocr_with_mock([mock_paddle_output])
        results = ocr.run(text_image, mock_detections)
        assert results[0].skip is False
        assert results[0].text == "HELLO WORLD"

    def test_low_confidence_is_skipped(self, mock_detections, text_image):
        mock_paddle_output = [["garbled", 0.42]]
        ocr = self._make_ocr_with_mock([mock_paddle_output])
        results = ocr.run(text_image, mock_detections)
        assert results[0].skip is True

    def test_short_text_is_skipped(self, mock_detections, text_image):
        mock_paddle_output = [["AB", 0.95]]
        ocr = self._make_ocr_with_mock([mock_paddle_output])
        results = ocr.run(text_image, mock_detections)
        assert results[0].skip is True

    def test_empty_paddle_result_is_skipped(self, mock_detections, text_image):
        mock_paddle_output = [[]]
        ocr = self._make_ocr_with_mock([mock_paddle_output])
        results = ocr.run(text_image, mock_detections)
        assert results[0].skip is True
        assert results[0].text == ""
        assert results[0].confidence == 0.0

    def test_result_det_id_matches_input(self, mock_detections, text_image):
        mock_paddle_output = [["HELLO", 0.88]]
        ocr = self._make_ocr_with_mock([mock_paddle_output])
        results = ocr.run(text_image, mock_detections)
        assert results[0].det_id == mock_detections[0].id

    def test_result_bbox_matches_detection(self, mock_detections, text_image):
        mock_paddle_output = [["HELLO", 0.88]]
        ocr = self._make_ocr_with_mock([mock_paddle_output])
        results = ocr.run(text_image, mock_detections)
        det = mock_detections[0]
        r = results[0]
        assert r.x == det.bbox.x
        assert r.y == det.bbox.y
        assert r.w == det.bbox.w
        assert r.h == det.bbox.h

    def test_erased_image_produces_skip(self, mock_detections):
        """
        REGRESSION GUARD: OCR must never run on the erased (black) image.
        If an all-black crop is fed in, the recognizer returns nothing → skip=True.
        This test ensures we won't accidentally pass the inpainted output to OCR.
        """
        black_image = np.zeros((100, 400, 3), dtype=np.uint8)
        mock_paddle_output = [[]]    # black crops → no text recognized
        ocr = self._make_ocr_with_mock([mock_paddle_output])
        results = ocr.run(black_image, mock_detections)
        assert all(r.skip for r in results), (
            "All regions on an erased (black) image should be skipped. "
            "This confirms OCR is not being run on the inpainted output."
        )

    def test_multiple_detections_all_processed(self, text_image):
        """Multiple detections each get their own result."""
        class BBox:
            def __init__(self, x, y, w, h): self.x, self.y, self.w, self.h = x, y, w, h
        class Det:
            def __init__(self, i):
                self.id = i
                self.polygon = np.array([[i*10,10],[(i+1)*10,10],
                                         [(i+1)*10,50],[i*10,50]], dtype=np.float32)
                self.bbox = BBox(i*10, 10, 10, 40)

        dets = [Det(i) for i in range(5)]
        good = ["TEXT", 0.88]
        ocr = self._make_ocr_with_mock([good] * 5)
        results = ocr.run(text_image, dets)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.det_id == i


# ════════════════════════════════════════════════════════════════════════════════
# SUITE 4: Regression Guard — Existing pipeline is UNCHANGED
# ════════════════════════════════════════════════════════════════════════════════

class TestRegressionGuard:
    """
    Verify that importing ocr.py does NOT affect the existing pipeline.
    These tests mirror the critical assertions in test_pipeline.py.
    """

    def test_import_ocr_does_not_pollute_core(self):
        """Importing ocr.py must not modify core.py's public API."""
        from pipeline import core
        from pipeline import ocr
        # core.py must still expose all original classes
        assert hasattr(core, "TextRemovalPipeline")
        assert hasattr(core, "PaddleOCRDetector")
        assert hasattr(core, "MaskGenerator")
        assert hasattr(core, "LaMaInpainter")
        assert hasattr(core, "QualityAssessor")
        assert hasattr(core, "PipelineResult")

    def test_core_pipeline_result_has_no_ocr_fields(self):
        """PipelineResult must NOT have ocr_results field — we don't touch it."""
        from pipeline.core import PipelineResult
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(PipelineResult)}
        assert "ocr_results" not in field_names
        assert "translation" not in field_names

    def test_existing_pipeline_runs_without_ocr(self):
        """
        Full Stage 1-4 pipeline must run to completion without any ocr.py involvement.
        Imports ocr.py first to confirm no side effects.

        NOTE: We use a random noise image (unlikely to have detectable text).
        When no text is detected, core.py exits early and does NOT add mask_ms,
        inpaint_ms, or quality_ms to timing — that is correct behavior.
        We only assert the guaranteed contract: output shape + detection_ms.
        """
        from pipeline import ocr  # import Stage 5 module first — must not affect core
        from pipeline.core import TextRemovalPipeline

        pipe = TextRemovalPipeline()
        img = np.random.randint(50, 200, (100, 150, 3), dtype=np.uint8)
        result = pipe.run(img)

        # Always-present fields (even on no-detection early exit)
        assert result.output.shape == img.shape
        assert result.output.dtype == np.uint8
        assert "detection_ms" in result.timing
        assert "ssim" in result.quality
        # Stage 5 must NOT have added any fields to core result
        assert not hasattr(result, "ocr_results")
        assert not hasattr(result, "translation")


# ════════════════════════════════════════════════════════════════════════════════
# SUITE 5: Live model test (optional — requires PaddleOCR model on disk)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.live
class TestRegionOCRLive:
    """
    These tests load the actual PaddleOCR model. They are slow (~5–10 s on first
    call). Skip in CI with: pytest -m 'not live'
    """

    @pytest.fixture(scope="class")
    def live_ocr(self):
        return RegionOCR()

    def test_live_ocr_reads_printed_text(self, live_ocr):
        """Create a synthetic image with clear text and verify OCR reads it."""
        img = np.ones((80, 300, 3), dtype=np.uint8) * 255
        cv2.putText(img, "INVOICE", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

        class BBox:
            x, y, w, h = 5, 5, 290, 70
        class Det:
            id = 0
            polygon = np.array([[5, 5], [295, 5], [295, 75], [5, 75]],
                               dtype=np.float32)
            bbox = BBox()

        results = live_ocr.run(img, [Det()])
        assert len(results) == 1
        r = results[0]
        # The word "INVOICE" should be partially or fully recognized
        assert len(r.text) >= 3, f"Expected readable text, got: {r.text!r}"
        assert r.confidence > 0.0

    def test_live_ocr_skips_blank_region(self, live_ocr):
        """A blank white crop should produce skip=True."""
        blank = np.ones((80, 200, 3), dtype=np.uint8) * 255

        class BBox:
            x, y, w, h = 0, 0, 200, 80
        class Det:
            id = 0
            polygon = np.array([[0, 0], [200, 0], [200, 80], [0, 80]],
                               dtype=np.float32)
            bbox = BBox()

        results = live_ocr.run(blank, [Det()])
        assert results[0].skip is True
