"""
python-backend/tests/test_pipeline.py
────────────────────────────────────────────────────────────────────────────────
pytest integration test suite for the backend pipeline.

Run:  pytest python-backend/tests/test_pipeline.py -v
"""

import pytest
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.core import (
    BBox, Detection, MaskResult,
    MSERDetector, MaskGenerator, LaMaInpainter,
    QualityAssessor, TextRemovalPipeline,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def gradient_image():
    """Gradient background — realistic for inpainting quality tests."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        img[y] = [int(y * 0.5), int(100 + y * 0.2), int(200 - y * 0.3)]
    return img


@pytest.fixture
def image_with_text(gradient_image):
    img = gradient_image.copy()
    cv2.putText(img, "SAMPLE WATERMARK", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)
    cv2.putText(img, "Another text line", (80, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20,20,200), 3)
    return img


@pytest.fixture
def simple_mask():
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[80:130, 50:540] = 255
    mask[330:380, 80:520] = 255
    return MaskResult(mask, (mask>0).sum()/(480*640), int((mask>0).sum()))


@pytest.fixture
def pipeline():
    return TextRemovalPipeline()


# ════════════════════════════════════════════════════════════════════════════
# SUITE 1: BBox utility
# ════════════════════════════════════════════════════════════════════════════

class TestBBox:
    def test_area_correct(self):
        assert BBox(0, 0, 100, 50).area == 5000

    def test_iou_identical_boxes_is_1(self):
        b = BBox(10, 10, 100, 50)
        assert b.iou(b) == pytest.approx(1.0)

    def test_iou_non_overlapping_is_0(self):
        a = BBox(0,   0, 50, 50)
        b = BBox(100, 0, 50, 50)
        assert a.iou(b) == 0.0

    def test_iou_partial_overlap(self):
        a = BBox(0, 0, 100, 100)
        b = BBox(50, 0, 100, 100)
        iou = a.iou(b)
        assert 0 < iou < 1


# ════════════════════════════════════════════════════════════════════════════
# SUITE 2: MaskGenerator
# ════════════════════════════════════════════════════════════════════════════

class TestMaskGenerator:
    MG = MaskGenerator()

    def _make_detection(self, x, y, w, h):
        poly = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.int32)
        return Detection(0, BBox(x,y,w,h), poly, 0.9, 'mser')

    def test_single_rect_fills_region(self):
        det = self._make_detection(10, 10, 100, 30)
        m = self.MG.generate([det], 100, 200, dilation_px=0)
        assert m.data[20, 50] == 255    # inside
        assert m.data[5, 5]   == 0      # outside

    def test_dilation_expands_mask(self):
        det = self._make_detection(50, 50, 100, 20)
        m0 = self.MG.generate([det], 200, 300, dilation_px=0)
        m3 = self.MG.generate([det], 200, 300, dilation_px=3)
        assert m3.masked_pixels > m0.masked_pixels

    def test_empty_detections_produces_zero_mask(self):
        m = self.MG.generate([], 100, 200, dilation_px=3)
        assert m.coverage == 0
        assert m.data.sum() == 0

    def test_coverage_is_fraction_of_image(self):
        det = self._make_detection(0, 0, 100, 100)
        m = self.MG.generate([det], 200, 200, dilation_px=0)
        expected = (100 * 100) / (200 * 200)
        assert m.coverage == pytest.approx(expected, abs=0.01)

    def test_multiple_regions_union(self):
        d1 = self._make_detection(10, 10, 50, 20)
        d2 = self._make_detection(200, 10, 50, 20)
        m = self.MG.generate([d1, d2], 100, 400, dilation_px=0)
        assert m.data[20, 30]  == 255   # inside d1
        assert m.data[20, 225] == 255   # inside d2
        assert m.data[20, 130] == 0     # gap between

    def test_full_image_detection_saturates_coverage(self):
        det = self._make_detection(0, 0, 200, 200)
        m = self.MG.generate([det], 200, 200, dilation_px=0)
        assert m.coverage == pytest.approx(1.0, abs=0.01)


# ════════════════════════════════════════════════════════════════════════════
# SUITE 3: LaMaInpainter
# ════════════════════════════════════════════════════════════════════════════

class TestLaMaInpainter:
    INP = LaMaInpainter()

    def test_output_shape_matches_input(self, image_with_text, simple_mask):
        result, _ = self.INP.inpaint(image_with_text, simple_mask, radius=5, method='telea')
        assert result.shape == image_with_text.shape

    def test_output_dtype_is_uint8(self, image_with_text, simple_mask):
        result, _ = self.INP.inpaint(image_with_text, simple_mask, radius=5, method='telea')
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("method", ["telea", "ns"])
    def test_both_methods_produce_valid_output(self, image_with_text, simple_mask, method):
        result, _ = self.INP.inpaint(image_with_text, simple_mask, radius=5, method=method)
        assert result is not None
        assert not np.all(result == 0)

    def test_unmasked_pixels_unchanged(self, gradient_image):
        """Pixels outside the mask must remain identical to the input."""
        mask_data = np.zeros((480, 640), dtype=np.uint8)
        mask_data[100:120, 100:400] = 255
        mask = MaskResult(mask_data, mask_data.mean()/255, int((mask_data>0).sum()))
        result, _ = self.INP.inpaint(gradient_image, mask, radius=5, method='telea')
        outside = mask_data == 0
        np.testing.assert_array_equal(result[outside], gradient_image[outside])

    def test_ssim_above_threshold_on_gradient(self, gradient_image):
        """Classic inpainting of text on a gradient must achieve SSIM > 0.90
        (measured by our own QualityAssessor to avoid an external skimage dep)."""
        # Draw text onto the gradient
        img_text = gradient_image.copy()
        cv2.putText(img_text, "REMOVE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
        # Build a matching mask
        mask_data = np.zeros((480, 640), dtype=np.uint8)
        cv2.putText(mask_data, "REMOVE", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 7)
        from pipeline.core import MaskResult, QualityAssessor
        mask = MaskResult(mask_data, mask_data.mean()/255, int((mask_data>0).sum()))
        # Inpaint
        result, _ = self.INP.inpaint(img_text, mask, radius=5, method='telea')
        # Score with our own QualityAssessor (no external dep)
        qa = QualityAssessor()
        metrics = qa.assess(gradient_image, result, mask, threshold=0.90)
        assert metrics['ssim'] > 0.90, f"SSIM {metrics['ssim']:.3f} < 0.90"


# ════════════════════════════════════════════════════════════════════════════
# SUITE 4: QualityAssessor
# ════════════════════════════════════════════════════════════════════════════

class TestQualityAssessor:
    QA = QualityAssessor()

    def _full_mask(self, h=64, w=64):
        data = np.ones((h,w), dtype=np.uint8) * 255
        return MaskResult(data, 1.0, h*w)

    def test_identical_images_ssim_is_1(self):
        img = np.random.randint(0, 255, (64,64,3), dtype=np.uint8)
        m = self.QA.assess(img, img, self._full_mask())
        assert m['ssim'] == pytest.approx(1.0, abs=0.01)

    def test_identical_images_mse_is_0(self):
        img = np.ones((64,64,3), dtype=np.uint8) * 128
        m = self.QA.assess(img, img, self._full_mask())
        assert m['mse'] == 0.0

    def test_black_vs_white_ssim_low(self):
        black = np.zeros((64,64,3), dtype=np.uint8)
        white = np.ones((64,64,3), dtype=np.uint8) * 255
        m = self.QA.assess(black, white, self._full_mask())
        assert m['ssim'] < 0.3

    def test_above_threshold_correct(self):
        img = np.ones((64,64,3), dtype=np.uint8) * 128
        m = self.QA.assess(img, img, self._full_mask(), threshold=0.82)
        assert m['above_threshold'] is True

    def test_empty_mask_returns_ssim_1(self):
        img = np.zeros((64,64,3), dtype=np.uint8)
        empty = MaskResult(np.zeros((64,64),np.uint8), 0.0, 0)
        m = self.QA.assess(img, img, empty)
        assert m['ssim'] == 1.0

    def test_psnr_is_inf_for_identical(self):
        img = np.ones((64,64,3), dtype=np.uint8) * 100
        m = self.QA.assess(img, img, self._full_mask())
        assert m['psnr'] == float('inf')


# ════════════════════════════════════════════════════════════════════════════
# SUITE 5: Full Pipeline Integration
# ════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    def test_pipeline_output_shape_preserved(self, pipeline, image_with_text):
        result = pipeline.run(image_with_text)
        assert result.output.shape == image_with_text.shape

    def test_pipeline_returns_timing_for_all_stages(self, pipeline, image_with_text):
        result = pipeline.run(image_with_text)
        assert 'detection_ms' in result.timing
        assert 'mask_ms'      in result.timing
        assert 'inpaint_ms'   in result.timing
        assert 'quality_ms'   in result.timing  # previously untested — Stage 4 must record timing

    def test_pipeline_no_text_image_returns_original(self, pipeline, gradient_image):
        result = pipeline.run(gradient_image)
        if result.detections == []:
            np.testing.assert_array_equal(result.output, gradient_image)

    def test_pipeline_quality_dict_has_required_keys(self, pipeline, image_with_text):
        result = pipeline.run(image_with_text)
        assert 'ssim'  in result.quality
        assert 'psnr'  in result.quality
        assert 'above_threshold' in result.quality

    def test_pipeline_mask_coverage_bounded_01(self, pipeline, image_with_text):
        result = pipeline.run(image_with_text)
        assert 0.0 <= result.mask.coverage <= 1.0

    def test_total_ms_equals_sum_of_stage_times(self, pipeline, image_with_text):
        """total_ms must equal the sum of the individual stage timings.

        Previously, sum(timing.values()) was called *after* 'total_ms' was
        inserted into the dict, meaning it would have included itself.
        This test would have caught that bug.
        """
        result = pipeline.run(image_with_text)
        stage_keys = [k for k in result.timing if k != 'total_ms' and isinstance(result.timing[k], (int, float))]
        expected   = round(sum(result.timing[k] for k in stage_keys), 1)
        assert result.timing['total_ms'] == pytest.approx(expected, abs=1.0)

    @pytest.mark.parametrize("method", ["telea", "ns"])
    def test_pipeline_both_inpaint_methods(self, pipeline, image_with_text, method):
        result = pipeline.run(image_with_text, inpaint_method=method)
        assert result.output is not None
        assert result.output.dtype == np.uint8

    @pytest.mark.slow
    def test_pipeline_performance_720p_under_3s(self, pipeline):
        """720p image must complete in under 3 seconds on CPU (CI machine).

        Mark: @pytest.mark.slow  — skip with:  pytest -m 'not slow'
        """
        img = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
        cv2.putText(img, "Benchmark Text", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
        result = pipeline.run(img)
        assert result.timing['total_ms'] < 3000, \
            f"Pipeline took {result.timing['total_ms']}ms > 3000ms"
