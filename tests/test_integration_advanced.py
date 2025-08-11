#!/usr/bin/env python3
"""Advanced integration test for parallel processing with larger PDFs."""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_detection import OCRDetection, PDFAnalyzer


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


def test_large_pdf_parallel():
    """Test parallel processing with a larger PDF."""
    print("=== Large PDF Parallel Processing Test ===")

    # Use the provided PDF file
    pdf_url = "https://files.edgestore.dev/kv3hoirymwcmuuoj/publicFiles/_public/e168fb66-4c8b-4ddb-a807-414cb0ca72fd.pdf"
    pdf_path = download_pdf(pdf_url, "large_test.pdf")

    if not pdf_path:
        print("Could not download test PDF")
        pytest.fail("Could not download test PDF")

    try:
        with PDFAnalyzer(pdf_path) as analyzer:
            total_pages = len(analyzer.doc)
            print(f"PDF has {total_pages} pages")

            if total_pages <= 10:
                print("PDF too small for meaningful parallel test, but testing anyway...")

            # Test sequential processing
            print("Testing sequential processing...")
            start_time = time.time()
            sequential_results = analyzer.analyze_all_pages()
            sequential_time = time.time() - start_time
            print(
                f"Sequential: {sequential_time:.3f}s ({total_pages/sequential_time:.1f} pages/sec)"
            )

            # Test parallel processing
            print("Testing parallel processing...")
            start_time = time.time()
            parallel_results = analyzer.analyze_all_pages_parallel(max_workers=4)
            parallel_time = time.time() - start_time
            print(f"Parallel: {parallel_time:.3f}s ({total_pages/parallel_time:.1f} pages/sec)")

            # Test auto method
            print("Testing auto method selection...")
            start_time = time.time()
            auto_results = analyzer.analyze_all_pages_auto(parallel=True, max_workers=2)
            auto_time = time.time() - start_time
            print(f"Auto: {auto_time:.3f}s ({total_pages/auto_time:.1f} pages/sec)")

            # Verify all methods return same number of results
            assert len(sequential_results) == len(parallel_results) == len(auto_results)
            print("All methods returned consistent page counts")

            # Verify page ordering
            for i in range(total_pages):
                assert sequential_results[i].page_number == i
                assert parallel_results[i].page_number == i
                assert auto_results[i].page_number == i
            print("Page ordering consistent across all methods")

            # Test different worker counts
            if total_pages > 5:
                print("Testing different worker counts...")
                for workers in [2, 4, 8]:
                    start_time = time.time()
                    worker_results = analyzer.analyze_all_pages_parallel(max_workers=workers)
                    worker_time = time.time() - start_time
                    print(f"Workers={workers}: {worker_time:.3f}s")
                    assert len(worker_results) == total_pages

    except Exception as e:
        print(f"Large PDF test failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Large PDF test failed: {e}")
    finally:
        if pdf_path:
            pdf_path.unlink(missing_ok=True)


def test_cli_integration():
    """Test CLI with parallel processing options."""
    print("\n=== CLI Integration Test ===")

    # Use the provided PDF file
    pdf_url = "https://files.edgestore.dev/kv3hoirymwcmuuoj/publicFiles/_public/e168fb66-4c8b-4ddb-a807-414cb0ca72fd.pdf"
    pdf_path = download_pdf(pdf_url, "cli_test.pdf")

    if not pdf_path:
        print("Could not download test PDF for CLI")
        pytest.fail("Could not download test PDF for CLI")

    try:
        # Test basic CLI
        print("Testing basic CLI...")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, r'{Path(__file__).parent.parent / 'src'}'); "
                f"from ocr_detection.cli import main; "
                f"import sys; sys.argv = ['ocr-detect', r'{pdf_path}']; main()",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Basic CLI: SUCCESS")
            print(
                "Output preview:",
                result.stdout[:150] + "..." if len(result.stdout) > 150 else result.stdout,
            )
        else:
            print(f"Basic CLI: FAILED - {result.stderr}")
            pytest.fail(f"Basic CLI failed: {result.stderr}")

        # Test parallel CLI
        print("Testing parallel CLI...")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, r'{Path(__file__).parent.parent / 'src'}'); "
                f"from ocr_detection.cli import main; "
                f"import sys; sys.argv = ['ocr-detect', r'{pdf_path}', '--parallel', '--workers', '2', '--verbose']; main()",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Parallel CLI: SUCCESS")
            print(
                "Output preview:",
                result.stdout[:150] + "..." if len(result.stdout) > 150 else result.stdout,
            )
        else:
            print(f"Parallel CLI: FAILED - {result.stderr}")
            pytest.fail(f"Parallel CLI failed: {result.stderr}")

        # Test JSON output with parallel
        print("Testing JSON output with parallel...")
        json_output = pdf_path.with_suffix(".json")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, r'{Path(__file__).parent.parent / 'src'}'); "
                f"from ocr_detection.cli import main; "
                f"import sys; sys.argv = ['ocr-detect', r'{pdf_path}', '--parallel', '--format', 'json', '--output', r'{json_output}']; main()",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and json_output.exists():
            print("JSON output with parallel: SUCCESS")
            with open(json_output) as f:
                content = f.read()
                print(f"JSON output size: {len(content)} characters")
            json_output.unlink()
        else:
            print(f"JSON output test: FAILED - {result.stderr}")
            pytest.fail(f"JSON output test failed: {result.stderr}")

    except Exception as e:
        print(f"CLI integration test failed: {e}")
        pytest.fail(f"CLI integration test failed: {e}")
    finally:
        if pdf_path:
            pdf_path.unlink(missing_ok=True)


def test_error_handling():
    """Test error handling in parallel processing."""
    print("\n=== Error Handling Test ===")

    try:
        # Test with non-existent file
        try:
            detector = OCRDetection(parallel=True)
            detector.detect("nonexistent.pdf")
            print("ERROR: Should have raised FileNotFoundError")
            pytest.fail("Should have raised FileNotFoundError")
        except FileNotFoundError:
            print("Correctly handled non-existent file")

        # Test with invalid worker count (should work but adjust)
        pdf_url = "https://files.edgestore.dev/kv3hoirymwcmuuoj/publicFiles/_public/e168fb66-4c8b-4ddb-a807-414cb0ca72fd.pdf"
        pdf_path = download_pdf(pdf_url, "error_test.pdf")

        if pdf_path:
            with PDFAnalyzer(pdf_path) as analyzer:
                # Test with excessive worker count
                results = analyzer.analyze_all_pages_parallel(max_workers=1000)
                print(f"Handled excessive worker count, got {len(results)} results")
                assert len(results) > 0, "Should have gotten results"

                # Test with 0 workers (should use default)
                results = analyzer.analyze_all_pages_parallel(max_workers=0)
                print(f"Handled 0 workers, got {len(results)} results")
                assert len(results) > 0, "Should have gotten results with default workers"

            pdf_path.unlink()

    except Exception as e:
        print(f"Error handling test failed: {e}")
        pytest.fail(f"Error handling test failed: {e}")


def main():
    """Run advanced integration tests."""
    print("OCR Detection Advanced Integration Tests")
    print("=======================================")

    success = True

    # Test large PDF processing
    if not test_large_pdf_parallel():
        print("Large PDF test FAILED")
        success = False
    else:
        print("Large PDF test PASSED")

    # Test CLI integration
    if not test_cli_integration():
        print("CLI integration test FAILED")
        success = False
    else:
        print("CLI integration test PASSED")

    # Test error handling
    if not test_error_handling():
        print("Error handling test FAILED")
        success = False
    else:
        print("Error handling test PASSED")

    print(f"\n{'All advanced tests PASSED' if success else 'Some advanced tests FAILED'}")
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
