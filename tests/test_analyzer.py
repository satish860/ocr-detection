"""Tests for the content analyzer module."""

import pytest
from ocr_detection.analyzer import ContentAnalyzer


def test_analyze_text_quality():
    """Test text quality analysis."""
    text = "This is a sample text with some numbers 123 and punctuation!"
    metrics = ContentAnalyzer.analyze_text_quality(text)
    
    assert metrics.char_count == len(text)
    assert metrics.word_count > 0
    assert metrics.avg_word_length > 0
    assert isinstance(metrics.has_structured_content, bool)
    assert isinstance(metrics.language_indicators, dict)


def test_analyze_empty_text():
    """Test analysis of empty text."""
    metrics = ContentAnalyzer.analyze_text_quality("")
    
    assert metrics.char_count == 0
    assert metrics.word_count == 0
    assert metrics.avg_word_length == 0.0
    assert metrics.has_structured_content is False
    assert metrics.language_indicators == {}


def test_detect_structured_content():
    """Test structured content detection."""
    structured_text = """
    1. First item
    2. Second item
    • Bullet point
    | Column 1 | Column 2 |
    """
    
    unstructured_text = "Just a simple paragraph with no structure."
    
    structured_metrics = ContentAnalyzer.analyze_text_quality(structured_text)
    unstructured_metrics = ContentAnalyzer.analyze_text_quality(unstructured_text)
    
    assert structured_metrics.has_structured_content is True
    assert unstructured_metrics.has_structured_content is False


def test_detect_ocr_artifacts():
    """Test OCR artifact detection."""
    clean_text = "This is clean text without OCR errors."
    artifacts = ContentAnalyzer.detect_ocr_artifacts(clean_text)
    
    assert isinstance(artifacts["artifacts_found"], list)
    assert isinstance(artifacts["confidence"], float)
    assert 0 <= artifacts["confidence"] <= 1
    assert isinstance(artifacts["error_rate"], float)


def test_detect_ocr_artifacts_empty():
    """Test OCR artifact detection on empty text."""
    artifacts = ContentAnalyzer.detect_ocr_artifacts("")
    
    assert artifacts["artifacts_found"] == []
    assert artifacts["confidence"] == 1.0


def test_language_indicators():
    """Test language indicator detection."""
    english_text = "The quick brown fox jumps over the lazy dog and runs to the forest."
    metrics = ContentAnalyzer.analyze_text_quality(english_text)
    
    assert "english" in metrics.language_indicators
    assert "numeric" in metrics.language_indicators
    assert "special_chars" in metrics.language_indicators
    
    # English text should have high english score
    assert metrics.language_indicators["english"] > 0