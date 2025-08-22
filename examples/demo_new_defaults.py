#!/usr/bin/env python3
"""Quick demo of the optimized OCR detection performance."""

import sys
import time
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_detection.simple import detect_ocr


def quick_demo():
    """Quick demonstration of performance improvements."""
    # Path relative to project root
    test_pdf = Path(__file__).parent.parent / "tests" / "test_data" / "e59a8fba-a718-4065-a68a-90ab194cbfb3.pdf"
    
    if not test_pdf.exists():
        print("Test PDF not found. Please run from the examples directory or project root.")
        return
    
    print("OCR Detection Performance Demo")
    print("=" * 40)
    print(f"PDF: {test_pdf.name} (3.9MB, 1045 pages)")
    print()
    
    # Fast mode
    print("Fast mode (new default):")
    start = time.time()
    result = detect_ocr(str(test_pdf))
    fast_time = time.time() - start
    print(f"  Time: {fast_time:.1f}s | Status: {result['status']} | Pages: {len(result['pages'])}")
    
    # Fast mode with images
    print("Fast mode + OCR images:")
    start = time.time()
    result = detect_ocr(str(test_pdf), include_images=True)
    image_time = time.time() - start
    print(f"  Time: {image_time:.1f}s | Status: {result['status']} | Images: {len(result.get('page_images', {}))}")
    
    print()
    print("Success! The problematic PDF is now processed quickly.")


if __name__ == "__main__":
    quick_demo()