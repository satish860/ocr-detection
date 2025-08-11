"""Example usage of the simple OCR detection interface."""

from ocr_detection.simple import OCRDetection, detect_ocr


def main():
    # Example 1: Using the class
    print("Example 1: Using OCRDetection class")
    print("-" * 40)
    
    ocr_detector = OCRDetection()
    result = ocr_detector.detect("sample.pdf")
    
    print(f"Status: {result['status']}")
    print(f"Pages needing OCR: {result['pages']}")
    
    # Interpret the results
    if result['status'] == "false":
        print("✓ No OCR needed - all pages have extractable text")
    elif result['status'] == "true":
        print("! OCR required for all pages")
    else:  # partial
        print(f"! OCR required for {len(result['pages'])} pages: {result['pages']}")
    
    print()
    
    # Example 2: Using the convenience function
    print("Example 2: Using convenience function")
    print("-" * 40)
    
    result = detect_ocr("sample.pdf")
    
    if result['status'] == "partial":
        print(f"Some pages need OCR: {result['pages']}")
    elif result['status'] == "true":
        print("All pages need OCR")
    else:
        print("No OCR needed")
    
    print()
    
    # Example 3: Custom confidence threshold
    print("Example 3: Custom confidence threshold")
    print("-" * 40)
    
    # Stricter detection (higher confidence required)
    strict_detector = OCRDetection(confidence_threshold=0.7)
    result = strict_detector.detect("sample.pdf")
    
    print(f"With stricter settings:")
    print(f"  Status: {result['status']}")
    print(f"  Pages: {result['pages']}")


if __name__ == "__main__":
    print("Simple OCR Detection Examples")
    print("=" * 40)
    print()
    
    # Note: Replace 'sample.pdf' with actual PDF path
    print("NOTE: Replace 'sample.pdf' with your actual PDF file path\n")
    
    # Uncomment to run with actual PDF
    # main()