"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile
import requests


@pytest.fixture(scope="session")
def test_pdf_file():
    """Create a test PDF file for tests that need real PDFs."""
    # Try to download the test PDF
    pdf_url = "https://files.edgestore.dev/kv3hoirymwcmuuoj/publicFiles/_public/e168fb66-4c8b-4ddb-a807-414cb0ca72fd.pdf"
    temp_dir = Path(tempfile.gettempdir())
    test_pdf = temp_dir / "test.pdf"
    
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        with open(test_pdf, 'wb') as f:
            f.write(response.content)
        
        yield test_pdf
        
    except Exception:
        # Fallback: create a minimal test file
        test_pdf.write_bytes(b"%PDF-1.4\n%Test PDF\n")
        yield test_pdf
    
    finally:
        # Cleanup
        test_pdf.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def setup_test_environment(test_pdf_file):
    """Automatically set up test environment for all tests."""
    # Make sure test.pdf exists in the tests directory
    tests_dir = Path(__file__).parent
    test_pdf_local = tests_dir / "test.pdf"
    
    if not test_pdf_local.exists():
        try:
            # Copy from the session-scoped PDF
            test_pdf_local.write_bytes(test_pdf_file.read_bytes())
        except Exception:
            # Create minimal PDF
            test_pdf_local.write_bytes(b"%PDF-1.4\n%Test PDF\n")
    
    yield
    
    # Cleanup after each test
    test_pdf_local.unlink(missing_ok=True)