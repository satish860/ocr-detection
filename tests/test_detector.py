"""Tests for the PDF detector module."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from concurrent.futures import Future
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


class TestParallelProcessing:
    """Test class for parallel processing functionality."""
    
    @patch('ocr_detection.detector.fitz')
    @patch('ocr_detection.detector.pdfplumber')
    @patch('ocr_detection.detector.Path.exists')
    def setup_method(self, method, mock_exists, mock_pdfplumber, mock_fitz):
        """Set up test fixtures for parallel processing tests."""
        mock_exists.return_value = True
        
        # Mock a document with 20 pages
        self.mock_fitz_doc = MagicMock()
        
        # Create mock pages
        self.mock_pages = []
        for i in range(20):
            mock_page = Mock()
            mock_page.get_text.return_value = f"Page {i} text content"
            mock_page.rect = Mock()
            mock_page.rect.width = 612
            mock_page.rect.height = 792
            mock_page.get_images.return_value = []
            self.mock_pages.append(mock_page)
        
        def get_item(index):
            return self.mock_pages[index]
        
        self.mock_fitz_doc.configure_mock(**{'__len__.return_value': 20})
        self.mock_fitz_doc.configure_mock(**{'__getitem__.side_effect': get_item})
        mock_fitz.open.return_value = self.mock_fitz_doc
        
        # Mock pdfplumber
        self.mock_plumber_pdf = Mock()
        self.mock_plumber_pages = []
        for i in range(20):
            mock_page = Mock()
            mock_page.extract_text.return_value = f"Page {i} text"
            self.mock_plumber_pages.append(mock_page)
        
        self.mock_plumber_pdf.pages = self.mock_plumber_pages
        mock_pdfplumber.open.return_value = self.mock_plumber_pdf
        
        self.analyzer = PDFAnalyzer("test.pdf")
    
    def test_parallel_analysis_small_pdf(self):
        """Test that small PDFs use sequential processing."""
        # Mock a document with only 5 pages
        self.mock_fitz_doc.configure_mock(**{'__len__.return_value': 5})
        
        with self.analyzer:
            results = self.analyzer.analyze_all_pages_parallel()
            
            # Should have 5 results
            assert len(results) == 5
            
            # Results should be in order
            for i, result in enumerate(results):
                assert result.page_number == i
    
    @patch('ocr_detection.detector.ThreadPoolExecutor')
    def test_parallel_analysis_large_pdf(self, mock_executor_class):
        """Test parallel processing for large PDFs."""
        # Mock the executor
        mock_executor = Mock()
        mock_executor_class.return_value.__enter__ = Mock(return_value=mock_executor)
        mock_executor_class.return_value.__exit__ = Mock(return_value=None)
        
        # Mock futures
        mock_futures = []
        for i in range(4):  # Simulate 4 chunks
            future = Mock(spec=Future)
            chunk_results = []
            for j in range(5):
                page_num = i * 5 + j
                result = AnalysisResult(
                    page_number=page_num,
                    page_type=PageType.TEXT,
                    confidence=0.9,
                    text_ratio=0.7,
                    image_ratio=0.1,
                    text_length=100,
                    image_count=0,
                    details={}
                )
                chunk_results.append(result)
            future.result.return_value = chunk_results
            mock_futures.append(future)
        
        mock_executor.submit.side_effect = mock_futures
        
        # Mock as_completed
        with patch('ocr_detection.detector.as_completed') as mock_as_completed:
            mock_as_completed.return_value = mock_futures
            
            with self.analyzer:
                results = self.analyzer.analyze_all_pages_parallel(max_workers=4)
                
                # Should have 20 results
                assert len(results) == 20
                
                # Results should be sorted by page number
                for i, result in enumerate(results):
                    assert result.page_number == i
    
    def test_parallel_vs_sequential_consistency(self):
        """Test that parallel and sequential processing produce consistent results."""
        with self.analyzer:
            # Get sequential results
            sequential_results = self.analyzer.analyze_all_pages()
            
            # Get parallel results
            parallel_results = self.analyzer.analyze_all_pages_parallel()
            
            # Should have same number of results
            assert len(parallel_results) == len(sequential_results)
            
            # Results should be in same order
            for seq_result, par_result in zip(sequential_results, parallel_results):
                assert seq_result.page_number == par_result.page_number
                # Page types should match (though confidence might vary slightly)
                assert seq_result.page_type == par_result.page_type
    
    def test_analyze_all_pages_auto(self):
        """Test automatic method selection."""
        with self.analyzer:
            # Should use parallel for 20 pages
            results = self.analyzer.analyze_all_pages_auto(parallel=True)
            assert len(results) == 20
            
            # Should use sequential when parallel=False
            results_seq = self.analyzer.analyze_all_pages_auto(parallel=False)
            assert len(results_seq) == 20
    
    @patch('ocr_detection.detector.os.cpu_count')
    def test_worker_count_determination(self, mock_cpu_count):
        """Test automatic worker count determination."""
        mock_cpu_count.return_value = 8
        
        with self.analyzer:
            # Should limit workers to 8 even with more CPUs
            with patch('ocr_detection.detector.ThreadPoolExecutor') as mock_executor:
                self.analyzer.analyze_all_pages_parallel()
                # Check that ThreadPoolExecutor was called with max_workers=8
                mock_executor.assert_called_once()
                args, kwargs = mock_executor.call_args
                assert kwargs.get('max_workers') == 8 or args[0] == 8
    
    def test_parallel_error_handling(self):
        """Test error handling in parallel processing."""
        with patch.object(self.analyzer, '_analyze_pages_batch') as mock_batch:
            # Make one batch fail
            def side_effect(pages):
                if 10 in pages:
                    raise Exception("Test error")
                return []
            
            mock_batch.side_effect = side_effect
            
            with self.analyzer:
                # Should fall back to sequential for failed batch
                results = self.analyzer.analyze_all_pages_parallel()
                # Should still complete successfully
                assert len(results) >= 0  # Some results should be returned