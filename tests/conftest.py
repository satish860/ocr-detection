"""Pytest configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_pdf_file():
    """Provide path to the committed test PDF file."""
    tests_dir = Path(__file__).parent
    test_pdf = tests_dir / "test.pdf"

    if test_pdf.exists():
        yield test_pdf
    else:
        # Fallback: create a minimal test file if the committed one is missing
        test_pdf.write_bytes(b"%PDF-1.4\n%Test PDF\n")
        yield test_pdf
        # Don't clean up the fallback file in case other tests need it
