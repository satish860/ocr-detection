"""Example demonstrating base64 image functionality."""

from pathlib import Path
from ocr_detection import detect_ocr, OCRDetection

def demonstrate_base64_functionality():
    """Demonstrate the new base64 image functionality."""
    # Path to test data
    test_dir = Path(__file__).parent.parent / "tests" / "test_data"
    pdf_files = list(test_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No test PDF files found!")
        return
    
    test_pdf = pdf_files[0]
    print(f"Demonstrating with: {test_pdf.name}")
    
    # Example 1: Simple usage with images
    print("\n=== Example 1: Basic usage with images ===")
    result = detect_ocr(test_pdf, include_images=True)
    
    print(f"Status: {result['status']}")
    print(f"Pages needing OCR: {result['pages']}")
    
    if 'page_images' in result:
        print(f"Images available for pages: {list(result['page_images'].keys())}")
        for page_num, image_b64 in result['page_images'].items():
            print(f"  Page {page_num}: {len(image_b64)} characters of base64 data")
    
    # Example 2: Advanced configuration
    print("\n=== Example 2: Advanced configuration ===")
    detector = OCRDetection(
        confidence_threshold=0.7,
        include_images=True,
        image_format="jpeg",  # Use JPEG instead of PNG
        image_dpi=200         # Higher resolution
    )
    
    result_advanced = detector.detect(test_pdf)
    print(f"Advanced config - Status: {result_advanced['status']}")
    if 'page_images' in result_advanced:
        total_size = sum(len(img) for img in result_advanced['page_images'].values())
        print(f"Total base64 data: {total_size:,} characters")
    
    # Example 3: Backward compatibility (no images)
    print("\n=== Example 3: Backward compatibility ===")
    result_no_images = detect_ocr(test_pdf, include_images=False)
    
    print(f"Without images - Status: {result_no_images['status']}")
    print(f"Page images included: {'page_images' in result_no_images}")
    
    # Example 4: Different DPI settings
    print("\n=== Example 4: DPI comparison ===")
    result_low_dpi = detect_ocr(test_pdf, include_images=True, image_dpi=72)
    result_high_dpi = detect_ocr(test_pdf, include_images=True, image_dpi=300)
    
    if 'page_images' in result_low_dpi and 'page_images' in result_high_dpi:
        for page_num in result_low_dpi['page_images']:
            if page_num in result_high_dpi['page_images']:
                low_size = len(result_low_dpi['page_images'][page_num])
                high_size = len(result_high_dpi['page_images'][page_num])
                print(f"Page {page_num}: 72 DPI = {low_size:,} chars, 300 DPI = {high_size:,} chars")
    
    print("\n=== Base64 Image Functionality Demo Complete ===")
    print("\nKey features:")
    print("- Images are provided only for pages that need OCR")
    print("- Page numbers are 1-indexed (matching PDF page numbers)")
    print("- Supports both PNG and JPEG formats")
    print("- Configurable DPI for quality vs. size trade-offs")
    print("- Fully backward compatible when include_images=False")
    print("- Works with both sequential and parallel processing")

if __name__ == "__main__":
    demonstrate_base64_functionality()