"""Basic integration test to verify OCR detection functionality."""

import sys
sys.path.insert(0, 'src')

from ocr_detection import PDFAnalyzer, ContentAnalyzer
from pathlib import Path

def test_basic_functionality():
    """Test basic functionality without a real PDF."""
    print("Testing OCR Detection Library...")
    
    # Test text metrics analysis
    sample_text = "This is a sample text with some numbers 123 and punctuation!"
    metrics = ContentAnalyzer.analyze_text_quality(sample_text)
    
    print(f"Text Analysis Test:")
    print(f"  Character count: {metrics.char_count}")
    print(f"  Word count: {metrics.word_count}")
    print(f"  Average word length: {metrics.avg_word_length:.2f}")
    print(f"  Has structured content: {metrics.has_structured_content}")
    
    # Test OCR artifact detection
    ocr_artifacts = ContentAnalyzer.detect_ocr_artifacts(sample_text)
    print(f"\nOCR Artifact Detection:")
    print(f"  Artifacts found: {len(ocr_artifacts['artifacts_found'])}")
    print(f"  Confidence: {ocr_artifacts['confidence']:.2f}")
    
    print("\n[SUCCESS] Basic functionality tests passed!")
    
    # Instructions for testing with actual PDFs
    print("\nTo test with actual PDFs:")
    print("1. Place a PDF file in this directory")
    print("2. Run: uv run ocr-detect your_file.pdf")
    print("3. Or run: uv run python test_basic.py")
    print("4. Run unit tests: uv run pytest tests/")

if __name__ == "__main__":
    test_basic_functionality()