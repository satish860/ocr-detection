
"""Integration tests for the simple OCR detection API."""

from pathlib import Path

import pytest

from ocr_detection import OCRDetection, OCRStatus, detect_ocr

# Path to test data folder - update this path as needed
TEST_DATA_DIR = Path(__file__).parent / "test_data"


class TestSimpleAPI:
    """Integration tests for the simple API without mocking."""

    def test_detect_ocr_function_basic(self):
        """Test the detect_ocr convenience function."""
        # Use the available test PDF
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        result = detect_ocr(str(pdf_path))

        # Verify the result structure
        assert isinstance(result, dict)
        assert "status" in result
        assert "pages" in result
        assert result["status"] in ["true", "false", "partial"]
        assert isinstance(result["pages"], list)

        # If pages are returned, they should be positive integers
        for page_num in result["pages"]:
            assert isinstance(page_num, int)
            assert page_num > 0

    def test_ocr_detection_class_initialization(self):
        """Test OCRDetection class initialization with different parameters."""
        # Default initialization
        detector1 = OCRDetection()
        assert detector1.confidence_threshold == 0.5
        assert detector1.parallel

        # Custom initialization
        detector2 = OCRDetection(confidence_threshold=0.8, parallel=False)
        assert detector2.confidence_threshold == 0.8
        assert not detector2.parallel

    def test_ocr_detection_class_detect_method(self):
        """Test the OCRDetection.detect() method."""
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        detector = OCRDetection()
        result = detector.detect(str(pdf_path))

        # Verify basic structure
        assert isinstance(result, dict)
        assert "status" in result
        assert "pages" in result

        # Test status values
        status = result["status"]
        pages = result["pages"]

        if status == "false":
            assert pages == [], "No pages should need OCR when status is 'false'"
        elif status == "true":
            assert len(pages) > 0, "All pages should be listed when status is 'true'"
        else:  # partial
            assert status == "partial"
            assert len(pages) > 0, "Some pages should be listed when status is 'partial'"

    def test_parallel_processing_option(self):
        """Test parallel processing option."""
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        # Test with parallel processing disabled
        detector = OCRDetection(parallel=False)
        result_sequential = detector.detect(str(pdf_path))

        # Test with parallel processing enabled
        detector_parallel = OCRDetection(parallel=True)
        result_parallel = detector_parallel.detect(str(pdf_path))

        # Results should be the same regardless of parallel setting
        assert result_sequential["status"] == result_parallel["status"]
        assert set(result_sequential["pages"]) == set(result_parallel["pages"])

    def test_detect_with_override_parallel(self):
        """Test overriding parallel setting in detect method."""
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        # Initialize with parallel=False but override in detect
        detector = OCRDetection(parallel=False)

        # Override with parallel=True
        result1 = detector.detect(str(pdf_path), parallel=True)
        assert isinstance(result1, dict)

        # Override with max_workers
        result2 = detector.detect(str(pdf_path), parallel=True, max_workers=2)
        assert isinstance(result2, dict)

    def test_confidence_threshold_effect(self):
        """Test that different confidence thresholds may produce different results."""
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        # Low threshold - fewer pages might need OCR
        detector_low = OCRDetection(confidence_threshold=0.3)
        result_low = detector_low.detect(str(pdf_path))

        # High threshold - more pages might need OCR
        detector_high = OCRDetection(confidence_threshold=0.9)
        result_high = detector_high.detect(str(pdf_path))

        # Both should return valid results
        assert result_low["status"] in ["true", "false", "partial"]
        assert result_high["status"] in ["true", "false", "partial"]

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        detector = OCRDetection()

        with pytest.raises(FileNotFoundError) as exc_info:
            detector.detect("non_existent_file.pdf")

        assert "Document not found" in str(exc_info.value)

    def test_detect_ocr_with_path_object(self):
        """Test that Path objects work as input."""
        pdf_path = TEST_DATA_DIR / "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        # Test with Path object
        result = detect_ocr(pdf_path)
        assert isinstance(result, dict)
        assert "status" in result
        assert "pages" in result

    def test_multiple_pdfs_if_available(self):
        """Test with multiple PDFs if available in test_data folder."""
        test_pdfs = list(TEST_DATA_DIR.glob("*.pdf")) if TEST_DATA_DIR.exists() else []

        if len(test_pdfs) == 0:
            pytest.skip("No test PDFs found in test_data folder")

        detector = OCRDetection()

        for pdf_path in test_pdfs[:3]:  # Test up to 3 PDFs
            result = detector.detect(str(pdf_path))

            # Each should return a valid result
            assert isinstance(result, dict)
            assert "status" in result
            assert "pages" in result
            assert result["status"] in ["true", "false", "partial"]

            print(f"\n{pdf_path.name}: status={result['status']}, pages={result['pages']}")


class TestOCRStatus:
    """Test the OCRStatus enum values."""

    def test_ocr_status_values(self):
        """Verify OCRStatus enum has expected values."""
        assert OCRStatus.TRUE.value == "true"
        assert OCRStatus.FALSE.value == "false"
        assert OCRStatus.PARTIAL.value == "partial"


def test_imports():
    """Test that all expected exports are available."""
    from ocr_detection import OCRDetection, OCRStatus, detect_ocr

    assert callable(detect_ocr)
    assert OCRDetection is not None
    assert OCRStatus is not None


if __name__ == "__main__":
    # Run a quick test if executed directly
    print("Running quick integration test...")

    # Check if test data exists
    test_dir = Path(__file__).parent / "test_data"
    if not test_dir.exists():
        print(f"Please create a 'test_data' folder at: {test_dir}")
        print("And add test PDF files to it.")
    else:
        pdfs = list(test_dir.glob("*.pdf"))
        if pdfs:
            print(f"Found {len(pdfs)} test PDF(s)")
            for pdf in pdfs[:1]:
                print(f"\nTesting with: {pdf.name}")
                result = detect_ocr(str(pdf))
                print(f"Result: {result}")
        else:
            print("No PDF files found in test_data folder")
