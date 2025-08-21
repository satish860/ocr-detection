"""Tests for base64 image functionality."""

from pathlib import Path

import pytest

from ocr_detection import OCRDetection, detect_ocr

# Path to test data folder
TEST_DATA_DIR = Path(__file__).parent / "test_data"


class TestBase64Images:
    """Test base64 image functionality."""

    def test_detect_ocr_with_images(self):
        """Test the detect_ocr convenience function with image output."""
        # Use the available test PDF
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        result = detect_ocr(str(pdf_path), include_images=True, image_dpi=72)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert "pages" in result
        assert result["status"] in ["true", "false", "partial"]
        assert isinstance(result["pages"], list)

        # Check if page_images is included when include_images=True
        assert "page_images" in result
        assert isinstance(result["page_images"], dict)

        # Verify page indexing consistency
        ocr_pages = set(result["pages"])
        image_pages = set(result["page_images"].keys())
        assert ocr_pages == image_pages, "Page images should only be for pages needing OCR"

        # Verify image data format
        for page_num, image_b64 in result["page_images"].items():
            assert isinstance(page_num, int)
            assert page_num > 0, "Page numbers should be 1-indexed"
            assert isinstance(image_b64, str)
            assert len(image_b64) > 0, "Image data should not be empty"
            # PNG images typically start with this base64 prefix
            assert image_b64.startswith('iVBORw0KGgo') or len(image_b64) > 100

    def test_detect_ocr_without_images(self):
        """Test that no images are included when include_images=False."""
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        result = detect_ocr(str(pdf_path), include_images=False)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert "pages" in result

        # Verify no page_images when include_images=False
        assert "page_images" not in result or len(result.get("page_images", {})) == 0

    def test_ocr_detection_class_with_images(self):
        """Test OCRDetection class with image parameters."""
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        # Test with PNG format
        detector_png = OCRDetection(
            include_images=True,
            image_format="png",
            image_dpi=96
        )
        result_png = detector_png.detect(str(pdf_path))

        if "page_images" in result_png:
            for _page_num, image_b64 in result_png["page_images"].items():
                # PNG images should start with this base64 prefix
                assert image_b64.startswith('iVBORw0KGgo') or len(image_b64) > 100

        # Test with JPEG format
        detector_jpeg = OCRDetection(
            include_images=True,
            image_format="jpeg",
            image_dpi=96
        )
        result_jpeg = detector_jpeg.detect(str(pdf_path))

        if "page_images" in result_jpeg:
            for _page_num, image_b64 in result_jpeg["page_images"].items():
                # JPEG images typically start with this base64 prefix
                assert image_b64.startswith('/9j/') or len(image_b64) > 100

    def test_page_indexing_consistency(self):
        """Test that page numbers are consistent between pages and page_images."""
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        detector = OCRDetection(include_images=True)
        result = detector.detect(str(pdf_path))

        if result["pages"]:  # If there are pages needing OCR
            # All pages needing OCR should have corresponding images
            for page_num in result["pages"]:
                assert page_num in result["page_images"], f"Page {page_num} should have an image"
                assert page_num > 0, f"Page number {page_num} should be 1-indexed"

        # All images should correspond to pages needing OCR
        for page_num in result["page_images"]:
            assert page_num in result["pages"], f"Image for page {page_num} should only exist if page needs OCR"

    def test_image_override_parameter(self):
        """Test that include_images parameter can override class default."""
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        # Create detector with include_images=False by default
        detector = OCRDetection(include_images=False)

        # Override to include images
        result_with_images = detector.detect(str(pdf_path), include_images=True)
        if result_with_images["pages"]:  # If pages need OCR
            assert "page_images" in result_with_images

        # Use default (no images)
        result_no_images = detector.detect(str(pdf_path))
        assert "page_images" not in result_no_images or len(result_no_images.get("page_images", {})) == 0

    def test_different_dpi_settings(self):
        """Test that different DPI settings work."""
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        # Test with low DPI
        result_low_dpi = detect_ocr(str(pdf_path), include_images=True, image_dpi=72)

        # Test with high DPI
        result_high_dpi = detect_ocr(str(pdf_path), include_images=True, image_dpi=200)

        # Both should produce valid results
        assert result_low_dpi["status"] == result_high_dpi["status"]
        assert result_low_dpi["pages"] == result_high_dpi["pages"]

        # Higher DPI should generally produce larger images
        if "page_images" in result_low_dpi and "page_images" in result_high_dpi:
            for page_num in result_low_dpi["page_images"]:
                if page_num in result_high_dpi["page_images"]:
                    low_dpi_size = len(result_low_dpi["page_images"][page_num])
                    high_dpi_size = len(result_high_dpi["page_images"][page_num])
                    # High DPI should generally be larger (allowing some tolerance)
                    assert high_dpi_size >= low_dpi_size * 0.8
