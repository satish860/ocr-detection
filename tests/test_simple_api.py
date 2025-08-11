
"""Test the simple OCR detection API."""

import sys
from pathlib import Path

# Test with the basic functionality test PDF
test_pdf = Path("test_basic.py")  # We'll use an existing file for testing

print("Testing Simple OCR Detection API")
print("=" * 50)

# Import the library
try:
    from ocr_detection import OCRDetection, detect_ocr
    print("[OK] Successfully imported OCRDetection API")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 1: Create detector instance
print("\n1. Creating OCRDetection instance...")
try:
    detector = OCRDetection()
    print("[OK] OCRDetection instance created")
except Exception as e:
    print(f"[FAIL] Failed to create instance: {e}")
    sys.exit(1)

# Test 2: Test with non-existent file (should raise error)
print("\n2. Testing error handling with non-existent file...")
try:
    result = detector.detect("non_existent.pdf")
    print("[FAIL] Should have raised FileNotFoundError")
except FileNotFoundError as e:
    print(f"[OK] Correctly raised FileNotFoundError: {e}")
except Exception as e:
    print(f"[FAIL] Unexpected error: {e}")

# Test 3: If test PDF exists, analyze it
test_pdfs = [
    Path("sample.pdf"),
    Path("test.pdf"),
    Path("document.pdf"),
    Path("example.pdf")
]

pdf_to_test = None
for pdf in test_pdfs:
    if pdf.exists():
        pdf_to_test = pdf
        break

if pdf_to_test:
    print(f"\n3. Testing with actual PDF: {pdf_to_test}")
    
    # Test using class
    print("\n   Using OCRDetection class:")
    try:
        result = detector.detect(pdf_to_test)
        print(f"   [OK] Status: {result['status']}")
        print(f"   [OK] Pages: {result['pages']}")
        
        # Validate result structure
        assert 'status' in result, "Result missing 'status' field"
        assert 'pages' in result, "Result missing 'pages' field"
        assert result['status'] in ['true', 'false', 'partial'], f"Invalid status: {result['status']}"
        assert isinstance(result['pages'], list), "Pages should be a list"
        print("   [OK] Result structure is valid")
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
    
    # Test using convenience function
    print("\n   Using detect_ocr function:")
    try:
        result = detect_ocr(pdf_to_test)
        print(f"   [OK] Status: {result['status']}")
        print(f"   [OK] Pages: {result['pages']}")
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
    
    # Test with different confidence threshold
    print("\n   Using custom confidence threshold:")
    try:
        strict_detector = OCRDetection(confidence_threshold=0.8)
        result = strict_detector.detect(pdf_to_test)
        print(f"   [OK] Status (strict): {result['status']}")
        print(f"   [OK] Pages (strict): {result['pages']}")
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
else:
    print("\n3. No test PDF found. Create a file named 'sample.pdf' to test.")
    print("   Skipping PDF analysis tests.")

print("\n" + "=" * 50)
print("API Test Complete!")
print("\nThe simple API provides:")
print("  - detect() method returning {status, pages}")
print("  - status: 'true', 'false', or 'partial'")
print("  - pages: list of page numbers needing OCR")