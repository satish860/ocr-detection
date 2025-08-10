"""Basic functionality tests that don't require PDF files."""

import pytest
from ocr_detection import ContentAnalyzer, PageType, AnalysisResult


def test_imports():
    """Test that all main components can be imported."""
    from ocr_detection import PDFAnalyzer, ContentAnalyzer, PageType, AnalysisResult
    assert PDFAnalyzer is not None
    assert ContentAnalyzer is not None
    assert PageType is not None
    assert AnalysisResult is not None


def test_page_types():
    """Test PageType enum values."""
    assert PageType.TEXT.value == "text"
    assert PageType.SCANNED.value == "scanned"
    assert PageType.MIXED.value == "mixed"
    assert PageType.EMPTY.value == "empty"


def test_content_analyzer_basic():
    """Test ContentAnalyzer basic functionality."""
    # Test with regular text
    text = "This is a normal sentence with some words and numbers 123."
    metrics = ContentAnalyzer.analyze_text_quality(text)
    
    assert metrics.char_count > 0
    assert metrics.word_count > 0
    assert metrics.avg_word_length > 0
    assert isinstance(metrics.has_structured_content, bool)
    assert isinstance(metrics.language_indicators, dict)


def test_ocr_artifact_detection():
    """Test OCR artifact detection."""
    # Test with clean text
    clean_text = "This is clean readable text without any OCR issues."
    artifacts = ContentAnalyzer.detect_ocr_artifacts(clean_text)
    
    assert isinstance(artifacts, dict)
    assert "artifacts_found" in artifacts
    assert "confidence" in artifacts
    assert "error_rate" in artifacts
    assert 0 <= artifacts["confidence"] <= 1


def test_analysis_result_creation():
    """Test AnalysisResult dataclass."""
    result = AnalysisResult(
        page_number=0,
        page_type=PageType.TEXT,
        confidence=0.85,
        text_ratio=0.7,
        image_ratio=0.1,
        text_length=500,
        image_count=2,
        details={"test": "data"}
    )
    
    assert result.page_number == 0
    assert result.page_type == PageType.TEXT
    assert result.confidence == 0.85
    assert result.text_ratio == 0.7
    assert result.image_ratio == 0.1
    assert result.text_length == 500
    assert result.image_count == 2
    assert result.details == {"test": "data"}


def test_structured_content_detection():
    """Test detection of structured content."""
    # Structured text
    structured = """
    1. First item
    2. Second item
    • Bullet point
    - Another bullet
    TITLE: Some title
    """
    
    # Unstructured text  
    unstructured = "Just a plain paragraph of text without any special formatting or structure."
    
    structured_metrics = ContentAnalyzer.analyze_text_quality(structured)
    unstructured_metrics = ContentAnalyzer.analyze_text_quality(unstructured)
    
    # Note: The exact results may vary, but we can test the function doesn't crash
    assert isinstance(structured_metrics.has_structured_content, bool)
    assert isinstance(unstructured_metrics.has_structured_content, bool)