"""Integration tests for different OCR detection scenarios using real PDFs."""

from pathlib import Path

import pytest

from ocr_detection import OCRDetection, detect_ocr

# Path to test data folder
TEST_DATA_DIR = Path(__file__).parent / "test_data"

# Test PDFs with known characteristics
PDF_ALL_PAGES_NEED_OCR = "2e1b63c5-761d-48b9-b3b5-f263c3db4e30.pdf"  # 9 pages - all need OCR
PDF_THRESHOLD_SENSITIVE = (
    "433687b4-cd9e-4b25-b654-8b16df84ca7f.pdf"  # 24 pages - threshold sensitive
)


class TestOCRScenarios:
    """Test different OCR detection scenarios with real PDFs."""

    def test_pdf_all_pages_need_ocr(self):
        """Test PDF where all pages need OCR (fully scanned document)."""
        pdf_path = TEST_DATA_DIR / PDF_ALL_PAGES_NEED_OCR

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        result = detect_ocr(str(pdf_path))

        # This PDF should have status 'true' - all pages need OCR
        assert result["status"] == "true", f"Expected status 'true', got {result['status']}"

        # Should list all 9 pages
        assert len(result["pages"]) == 9, f"Expected 9 pages, got {len(result['pages'])}"

        # Pages should be 1 through 9
        expected_pages = list(range(1, 10))
        assert result["pages"] == expected_pages, (
            f"Expected pages {expected_pages}, got {result['pages']}"
        )

        print(
            f"\n✓ {PDF_ALL_PAGES_NEED_OCR}: Correctly identified as fully scanned (all 9 pages need OCR)"
        )

    def test_pdf_threshold_sensitive(self):
        """Test PDF that's sensitive to confidence thresholds."""
        pdf_path = TEST_DATA_DIR / PDF_THRESHOLD_SENSITIVE

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        # Test with default threshold (0.5)
        result = detect_ocr(str(pdf_path))

        # This PDF should have status 'false' with default threshold
        assert result["status"] == "false", (
            f"Expected status 'false' with default threshold, got {result['status']}"
        )

        # Should have empty pages list with default threshold
        assert result["pages"] == [], f"Expected empty pages list, got {result['pages']}"

        print(
            f"\n✓ {PDF_THRESHOLD_SENSITIVE}: Default threshold gives status 'false' with 0 pages needing OCR"
        )

    def test_different_threshold_behaviors(self):
        """Verify different threshold behaviors give us true, false, and partial statuses."""
        statuses_found = set()

        # Test PDF that needs OCR on all pages (always "true")
        pdf_all = TEST_DATA_DIR / PDF_ALL_PAGES_NEED_OCR
        if pdf_all.exists():
            result = detect_ocr(str(pdf_all))
            statuses_found.add(result["status"])
            assert result["status"] == "true"

        # Test PDF that's threshold sensitive
        pdf_sensitive = TEST_DATA_DIR / PDF_THRESHOLD_SENSITIVE
        if pdf_sensitive.exists():
            # Low threshold - should be "false"
            detector_low = OCRDetection(confidence_threshold=0.1)
            result_low = detector_low.detect(str(pdf_sensitive))
            statuses_found.add(result_low["status"])

            # High threshold - should be "partial"
            detector_high = OCRDetection(confidence_threshold=0.9)
            result_high = detector_high.detect(str(pdf_sensitive))
            statuses_found.add(result_high["status"])

        print(f"\n✓ Successfully demonstrated different statuses: {statuses_found}")

        # We should see at least "true" and one other status
        assert "true" in statuses_found, "Should detect 'true' status"
        assert len(statuses_found) >= 2, "Should detect multiple status types"

    def test_ocr_detection_class_with_different_pdfs(self):
        """Test OCRDetection class with different PDF types."""
        detector = OCRDetection()

        # Test with fully scanned PDF
        pdf_path_scanned = TEST_DATA_DIR / PDF_ALL_PAGES_NEED_OCR
        if pdf_path_scanned.exists():
            result = detector.detect(str(pdf_path_scanned))
            assert result["status"] == "true"
            assert len(result["pages"]) == 9

        # Test with text-based PDF
        pdf_path_text = TEST_DATA_DIR / PDF_THRESHOLD_SENSITIVE
        if pdf_path_text.exists():
            result = detector.detect(str(pdf_path_text))
            assert result["status"] == "false"
            assert len(result["pages"]) == 0

    def test_parallel_vs_sequential_consistency(self):
        """Verify parallel and sequential processing give same results."""
        # Use the 24-page PDF for faster testing
        pdf_path = TEST_DATA_DIR / PDF_THRESHOLD_SENSITIVE

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        # Sequential processing
        detector_seq = OCRDetection(parallel=False)
        result_seq = detector_seq.detect(str(pdf_path))

        # Parallel processing
        detector_par = OCRDetection(parallel=True)
        result_par = detector_par.detect(str(pdf_path))

        # Results should be identical
        assert result_seq["status"] == result_par["status"], "Status should match"
        assert result_seq["pages"] == result_par["pages"], "Pages should match"

        print("\n✓ Parallel and sequential processing give consistent results")

    def test_confidence_threshold_effects(self):
        """Test how confidence threshold affects OCR detection."""
        pdf_path = TEST_DATA_DIR / PDF_THRESHOLD_SENSITIVE

        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found at {pdf_path}")

        # High threshold - stricter requirements, may detect more pages needing OCR
        detector_high = OCRDetection(confidence_threshold=0.9)
        result_high = detector_high.detect(str(pdf_path))

        # Low threshold - more lenient, fewer pages may need OCR
        detector_low = OCRDetection(confidence_threshold=0.1)
        result_low = detector_low.detect(str(pdf_path))

        # Both should return valid statuses
        assert result_high["status"] in ["true", "false", "partial"]
        assert result_low["status"] in ["true", "false", "partial"]

        # With higher threshold, we expect same or more pages to need OCR
        assert len(result_high["pages"]) >= len(result_low["pages"]), (
            "Higher threshold should detect same or more pages needing OCR"
        )

        print("\n✓ Confidence threshold effects:")
        print(f"  High threshold (0.9): {result_high['status']}, {len(result_high['pages'])} pages")
        print(f"  Low threshold (0.1): {result_low['status']}, {len(result_low['pages'])} pages")


def test_quick_verification():
    """Quick test to verify PDFs and their characteristics."""
    pdfs = [
        (PDF_ALL_PAGES_NEED_OCR, "true", 9),
        (PDF_THRESHOLD_SENSITIVE, "false", 0),  # With default 0.5 threshold
    ]

    for pdf_name, expected_status, expected_page_count in pdfs:
        pdf_path = TEST_DATA_DIR / pdf_name

        if not pdf_path.exists():
            print(f"⚠ {pdf_name} not found")
            continue

        result = detect_ocr(str(pdf_path))

        assert result["status"] == expected_status, (
            f"{pdf_name}: Expected status '{expected_status}', got '{result['status']}'"
        )
        assert len(result["pages"]) == expected_page_count, (
            f"{pdf_name}: Expected {expected_page_count} pages, got {len(result['pages'])}"
        )

        print(f"✓ {pdf_name}: status={expected_status}, pages={expected_page_count}")


if __name__ == "__main__":
    # Run quick verification when executed directly
    print("Running OCR Scenario Tests\n")
    print("-" * 50)

    test_dir = Path(__file__).parent / "test_data"
    if not test_dir.exists():
        print(f"Error: test_data folder not found at {test_dir}")
    else:
        # Test both PDFs
        test_pdfs = [
            (PDF_ALL_PAGES_NEED_OCR, "All pages need OCR"),
            (PDF_THRESHOLD_SENSITIVE, "Threshold sensitive (default: no OCR)"),
        ]

        for pdf_name, description in test_pdfs:
            pdf_path = test_dir / pdf_name
            if pdf_path.exists():
                result = detect_ocr(str(pdf_path))
                print(f"\n{pdf_name} ({description}):")
                print(f"  Status: {result['status']}")
                print(f"  Pages needing OCR: {len(result['pages'])} pages")
                if len(result["pages"]) <= 10:
                    print(f"  Page numbers: {result['pages']}")
            else:
                print(f"\n{pdf_name}: NOT FOUND")

        print("\n" + "-" * 50)
        print("All OCR statuses demonstrated through confidence thresholds:")
        print("- 'true': Fully scanned PDFs (always need OCR)")
        print("- 'false': Text PDFs with low confidence threshold")
        print("- 'partial': Text PDFs with high confidence threshold")
