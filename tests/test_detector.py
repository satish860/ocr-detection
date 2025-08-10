"""Tests for the PDF detector module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from ocr_detection.detector import PDFAnalyzer, PageType, AnalysisResult


def test_page_type_enum():
    """Test PageType enumeration."""
    assert PageType.TEXT.value == "text"
    assert PageType.SCANNED.value == "scanned" 
    assert PageType.MIXED.value == "mixed"
    assert PageType.EMPTY.value == "empty"


def test_analysis_result_creation():
    """Test AnalysisResult dataclass creation."""
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


def test_pdf_analyzer_file_not_found():
    """Test PDFAnalyzer with non-existent file."""
    with pytest.raises(FileNotFoundError):
        PDFAnalyzer("nonexistent.pdf")


@patch('ocr_detection.detector.fitz')
@patch('ocr_detection.detector.pdfplumber')
@patch('ocr_detection.detector.Path.exists')
def test_pdf_analyzer_context_manager(mock_exists, mock_pdfplumber, mock_fitz):
    """Test PDFAnalyzer context manager functionality."""
    mock_exists.return_value = True
    mock_doc = Mock()
    mock_pdf = Mock()
    mock_fitz.open.return_value = mock_doc
    mock_pdfplumber.open.return_value = mock_pdf
    
    analyzer = PDFAnalyzer("test.pdf")
    
    with analyzer as a:
        assert a is analyzer
        assert analyzer.doc == mock_doc
        assert analyzer.plumber_pdf == mock_pdf
    
    mock_doc.close.assert_called_once()
    mock_pdf.close.assert_called_once()


class TestPDFAnalyzer:
    """Test class for PDFAnalyzer methods."""
    
    @patch('ocr_detection.detector.fitz')
    @patch('ocr_detection.detector.pdfplumber')  
    @patch('ocr_detection.detector.Path.exists')
    def setup_method(self, method, mock_exists, mock_pdfplumber, mock_fitz):
        """Set up test fixtures."""
        mock_exists.return_value = True
        
        # Mock fitz document and page
        self.mock_fitz_doc = Mock()
        self.mock_fitz_page = Mock()
        self.mock_fitz_page.get_text.return_value = "Sample extracted text"
        self.mock_fitz_page.rect = Mock()
        self.mock_fitz_page.rect.width = 612
        self.mock_fitz_page.rect.height = 792
        self.mock_fitz_page.get_images.return_value = []
        
        self.mock_fitz_doc.configure_mock(**{'__len__.return_value': 1})
        self.mock_fitz_doc.configure_mock(**{'__getitem__.return_value': self.mock_fitz_page})
        mock_fitz.open.return_value = self.mock_fitz_doc
        
        # Mock pdfplumber document and page
        self.mock_plumber_pdf = Mock()
        self.mock_plumber_page = Mock()
        self.mock_plumber_page.extract_text.return_value = "Sample text"
        
        self.mock_plumber_pdf.pages = [self.mock_plumber_page]
        mock_pdfplumber.open.return_value = self.mock_plumber_pdf
        
        self.analyzer = PDFAnalyzer("test.pdf")
    
    def test_classify_page_text_type(self):
        """Test page classification for text-heavy pages."""
        page_type, confidence = self.analyzer._classify_page(0.15, 0.05, 1000, 1)
        assert page_type == PageType.TEXT
        assert confidence > 0.7
    
    def test_classify_page_scanned_type(self):
        """Test page classification for scanned pages.""" 
        page_type, confidence = self.analyzer._classify_page(0.01, 0.8, 20, 5)
        assert page_type == PageType.SCANNED
        assert confidence > 0.6
    
    def test_classify_page_empty_type(self):
        """Test page classification for empty pages."""
        page_type, confidence = self.analyzer._classify_page(0.0, 0.0, 5, 0)
        assert page_type == PageType.EMPTY
        assert confidence > 0.9
    
    def test_classify_page_mixed_type(self):
        """Test page classification for mixed content pages."""
        page_type, confidence = self.analyzer._classify_page(0.08, 0.3, 200, 3)
        assert page_type == PageType.MIXED
        assert confidence > 0.5
    
    def test_calculate_text_ratio(self):
        """Test text ratio calculation."""
        ratio = self.analyzer._calculate_text_ratio("Hello world", 10000)
        assert isinstance(ratio, float)
        assert ratio >= 0.0
        assert ratio <= 1.0
        
        # Test with empty text
        ratio_empty = self.analyzer._calculate_text_ratio("", 10000)
        assert ratio_empty == 0.0
        
        # Test with zero page area
        ratio_zero_area = self.analyzer._calculate_text_ratio("Hello", 0)
        assert ratio_zero_area == 0.0
    
    def test_get_recommendation(self):
        """Test recommendation generation."""
        # High scanned percentage
        rec1 = self.analyzer._get_recommendation({"scanned": 8, "text": 2}, 10)
        assert "Strong recommendation" in rec1
        
        # Moderate mixed content
        rec2 = self.analyzer._get_recommendation({"mixed": 3, "text": 7}, 10)
        assert "Consider OCR" in rec2
        
        # Mostly text
        rec3 = self.analyzer._get_recommendation({"text": 9, "scanned": 1}, 10)
        assert "should work well" in rec3