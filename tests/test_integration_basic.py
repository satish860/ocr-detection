#!/usr/bin/env python3
"""Basic integration test for parallel processing functionality."""

import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_detection import OCRDetection, PDFAnalyzer, detect_ocr


def download_pdf(url, filename):
    """Download PDF from URL to temporary file."""
    try:
        print(f"Downloading PDF from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        temp_file = Path(tempfile.gettempdir()) / filename
        with open(temp_file, "wb") as f:
            f.write(response.content)

        print(f"Downloaded {len(response.content)} bytes to {temp_file}")
        return temp_file
    except Exception as e:
        print(f"Failed to download PDF: {e}")
        return None


def test_basic_functionality():
    """Test basic parallel processing functionality with a small PDF."""
    print("=== Basic Integration Test ===")

    # Use the provided PDF file
    pdf_url = "https://files.edgestore.dev/kv3hoirymwcmuuoj/publicFiles/_public/e168fb66-4c8b-4ddb-a807-414cb0ca72fd.pdf"
    pdf_path = download_pdf(pdf_url, "basic_test.pdf")

    if not pdf_path:
        print("Failed to download test PDF")
        pytest.fail("Failed to download test PDF")

    try:
        # Test that our classes work with real PDFs
        print("\nTesting PDFAnalyzer...")
        with PDFAnalyzer(pdf_path) as analyzer:
            total_pages = len(analyzer.doc)
            print(f"PDF has {total_pages} pages")

            # Test sequential
            sequential_results = analyzer.analyze_all_pages()
            print(f"Sequential analysis: {len(sequential_results)} pages processed")

            # Test parallel (should fall back to sequential for small PDFs)
            parallel_results = analyzer.analyze_all_pages_parallel()
            print(f"Parallel analysis: {len(parallel_results)} pages processed")

            # Test auto method
            auto_results = analyzer.analyze_all_pages_auto(parallel=True)
            print(f"Auto analysis: {len(auto_results)} pages processed")

            # Verify consistency
            assert len(sequential_results) == len(parallel_results) == len(auto_results)
            print("All methods returned consistent results")

        # Test Simple API
        print("\nTesting Simple API...")
        result = detect_ocr(pdf_path, parallel=True)
        print(f"Simple API result: {result}")

        # Test OCRDetection class
        detector = OCRDetection(parallel=True)
        result2 = detector.detect(pdf_path)
        assert result == result2
        print("OCRDetection class works correctly")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Test failed with exception: {e}")
    finally:
        if pdf_path:
            pdf_path.unlink(missing_ok=True)


def test_performance_comparison():
    """Test performance difference between sequential and parallel."""
    print("\n=== Performance Comparison Test ===")

    # Use the provided PDF file
    pdf_url = "https://files.edgestore.dev/kv3hoirymwcmuuoj/publicFiles/_public/e168fb66-4c8b-4ddb-a807-414cb0ca72fd.pdf"
    pdf_path = download_pdf(pdf_url, "performance_test.pdf")

    if not pdf_path:
        print("Failed to download test PDF")
        pytest.fail("Failed to download test PDF")

    try:
        # Test multiple runs to see timing differences
        print("\nTesting timing differences...")

        sequential_times = []
        parallel_times = []

        for _ in range(3):
            with PDFAnalyzer(pdf_path) as analyzer:
                # Sequential
                start = time.time()
                seq_results = analyzer.analyze_all_pages()
                sequential_times.append(time.time() - start)

                # Parallel
                start = time.time()
                par_results = analyzer.analyze_all_pages_parallel()
                parallel_times.append(time.time() - start)

                # Verify results are consistent
                assert len(seq_results) == len(par_results)

        avg_seq = sum(sequential_times) / len(sequential_times)
        avg_par = sum(parallel_times) / len(parallel_times)

        print(f"Average sequential time: {avg_seq:.3f}s")
        print(f"Average parallel time: {avg_par:.3f}s")
        print(f"Times collected over {len(sequential_times)} runs")

        # Assert that we got valid timing results
        assert len(sequential_times) == 3
        assert len(parallel_times) == 3
        assert all(t > 0 for t in sequential_times)
        assert all(t > 0 for t in parallel_times)

    except Exception as e:
        print(f"Performance test failed: {e}")
        pytest.fail(f"Performance test failed with exception: {e}")
    finally:
        if pdf_path:
            pdf_path.unlink(missing_ok=True)


def main():
    """Run basic integration tests."""
    print("OCR Detection Basic Integration Tests")
    print("=====================================")

    success = True

    # Test basic functionality
    if not test_basic_functionality():
        print("Basic functionality test FAILED")
        success = False
    else:
        print("Basic functionality test PASSED")

    # Test performance comparison
    if not test_performance_comparison():
        print("Performance comparison test FAILED")
        success = False
    else:
        print("Performance comparison test PASSED")

    print(f"\n{'All tests PASSED' if success else 'Some tests FAILED'}")
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
