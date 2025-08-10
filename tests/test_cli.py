"""Tests for the CLI module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
from pathlib import Path

from ocr_detection.cli import main
from ocr_detection.detector import AnalysisResult, PageType


@pytest.fixture
def mock_analysis_results():
    """Create mock analysis results for testing."""
    return [
        AnalysisResult(
            page_number=0,
            page_type=PageType.TEXT,
            confidence=0.85,
            text_ratio=0.7,
            image_ratio=0.1,
            text_length=500,
            image_count=1,
            details={"extracted_text_preview": "Sample text content for testing purposes"}
        ),
        AnalysisResult(
            page_number=1,
            page_type=PageType.SCANNED,
            confidence=0.9,
            text_ratio=0.02,
            image_ratio=0.8,
            text_length=10,
            image_count=3,
            details={"extracted_text_preview": "OCR"}
        )
    ]


@pytest.fixture
def mock_summary():
    """Create mock summary data."""
    return {
        "total_pages": 2,
        "type_counts": {"text": 1, "scanned": 1},
        "type_percentages": {"text": 50.0, "scanned": 50.0},
        "average_confidence": 0.875,
        "recommended_action": "Consider OCR for optimal text extraction"
    }


class TestCLI:
    """Test class for CLI functionality."""
    
    def test_cli_help(self):
        """Test CLI help output."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "Analyze PDF pages to detect text vs scanned content" in result.output
        assert "--output" in result.output
        assert "--format" in result.output
        assert "--verbose" in result.output
    
    @patch('ocr_detection.cli.PDFAnalyzer')
    def test_cli_nonexistent_file(self, mock_analyzer_class):
        """Test CLI with non-existent PDF file."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['nonexistent.pdf'])
            # Click should handle the file existence check
            assert result.exit_code != 0
    
    @patch('ocr_detection.cli.PDFAnalyzer')
    def test_cli_basic_analysis(self, mock_analyzer_class, mock_analysis_results, mock_summary):
        """Test basic CLI analysis functionality."""
        # Setup mocks
        mock_analyzer = Mock()
        mock_analyzer.analyze_all_pages.return_value = mock_analysis_results
        mock_analyzer.get_summary.return_value = mock_summary
        mock_analyzer.configure_mock(**{'__enter__.return_value': mock_analyzer})
        mock_analyzer.configure_mock(**{'__exit__.return_value': None})
        mock_analyzer_class.return_value = mock_analyzer
        
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Create a dummy PDF file
            Path('test.pdf').write_text("dummy content")
            
            result = runner.invoke(main, ['test.pdf'])
            
            assert result.exit_code == 0
            assert "PDF CONTENT ANALYSIS SUMMARY" in result.output
            assert "Total Pages: 2" in result.output
            assert "Consider OCR" in result.output
    
    @patch('ocr_detection.cli.PDFAnalyzer')
    def test_cli_json_output(self, mock_analyzer_class, mock_analysis_results, mock_summary):
        """Test CLI JSON output format."""
        # Setup mocks
        mock_analyzer = Mock()
        mock_analyzer.analyze_all_pages.return_value = mock_analysis_results
        mock_analyzer.get_summary.return_value = mock_summary
        mock_analyzer.configure_mock(**{'__enter__.return_value': mock_analyzer})
        mock_analyzer.configure_mock(**{'__exit__.return_value': None})
        mock_analyzer_class.return_value = mock_analyzer
        
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            Path('test.pdf').write_text("dummy content")
            
            result = runner.invoke(main, ['test.pdf', '--format', 'json'])
            
            assert result.exit_code == 0
            assert '"page_number": 0' in result.output
            assert '"page_type": "text"' in result.output
            assert '"confidence": 0.85' in result.output
    
    @patch('ocr_detection.cli.PDFAnalyzer')
    def test_cli_specific_page(self, mock_analyzer_class, mock_analysis_results, mock_summary):
        """Test CLI analysis of specific page."""
        # Setup mocks
        mock_analyzer = Mock()
        mock_analyzer.analyze_page.return_value = mock_analysis_results[0]
        mock_analyzer.get_summary.return_value = mock_summary
        mock_analyzer.configure_mock(**{'__enter__.return_value': mock_analyzer})
        mock_analyzer.configure_mock(**{'__exit__.return_value': None})
        mock_analyzer_class.return_value = mock_analyzer
        
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            Path('test.pdf').write_text("dummy content")
            
            result = runner.invoke(main, ['test.pdf', '--page', '0'])
            
            assert result.exit_code == 0
            assert "Analyzing page 0" in result.output
    
    @patch('ocr_detection.cli.PDFAnalyzer')
    def test_cli_verbose_output(self, mock_analyzer_class, mock_analysis_results, mock_summary):
        """Test CLI verbose output."""
        # Setup mocks
        mock_analyzer = Mock()
        mock_analyzer.analyze_all_pages.return_value = mock_analysis_results
        mock_analyzer.get_summary.return_value = mock_summary
        mock_analyzer.configure_mock(**{'__enter__.return_value': mock_analyzer})
        mock_analyzer.configure_mock(**{'__exit__.return_value': None})
        mock_analyzer_class.return_value = mock_analyzer
        
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            Path('test.pdf').write_text("dummy content")
            
            result = runner.invoke(main, ['test.pdf', '--verbose'])
            
            assert result.exit_code == 0
            assert "DETAILED PAGE ANALYSIS" in result.output
            assert "Page 0:" in result.output
            assert "Confidence: 0.85" in result.output
    
    @patch('ocr_detection.cli.PDFAnalyzer')  
    def test_cli_csv_output_file(self, mock_analyzer_class, mock_analysis_results, mock_summary):
        """Test CLI CSV output to file."""
        # Setup mocks
        mock_analyzer = Mock()
        mock_analyzer.analyze_all_pages.return_value = mock_analysis_results
        mock_analyzer.get_summary.return_value = mock_summary
        mock_analyzer.configure_mock(**{'__enter__.return_value': mock_analyzer})
        mock_analyzer.configure_mock(**{'__exit__.return_value': None})
        mock_analyzer_class.return_value = mock_analyzer
        
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            Path('test.pdf').write_text("dummy content")
            
            result = runner.invoke(main, ['test.pdf', '--output', 'results.csv'])
            
            assert result.exit_code == 0
            assert "Results saved to results.csv" in result.output
            assert Path('results.csv').exists()
            
            # Check CSV content
            csv_content = Path('results.csv').read_text()
            assert 'page_number,page_type,confidence' in csv_content
            assert '0,text,0.85' in csv_content