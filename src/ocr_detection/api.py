"""Simplified API for OCR detection library."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .detector import PDFAnalyzer, PageType, AnalysisResult
from .analyzer import ContentAnalyzer


class PageRecommendation(Enum):
    """Recommendation for page processing."""
    NEEDS_OCR = "needs_ocr"
    NO_OCR_NEEDED = "no_ocr_needed"
    OCR_OPTIONAL = "ocr_optional"
    EMPTY = "empty"


@dataclass
class PageAnalysis:
    """Simplified page analysis result."""
    page_number: int
    recommendation: PageRecommendation
    confidence: float
    page_type: str
    text_length: int
    has_images: bool
    details: Optional[Dict[str, Any]] = None
    
    def needs_ocr(self) -> bool:
        """Check if page needs OCR processing."""
        return self.recommendation == PageRecommendation.NEEDS_OCR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "page_number": self.page_number,
            "recommendation": self.recommendation.value,
            "confidence": self.confidence,
            "page_type": self.page_type,
            "text_length": self.text_length,
            "has_images": self.has_images,
            "details": self.details
        }


@dataclass
class PDFAnalysisResult:
    """Complete PDF analysis result."""
    file_path: str
    total_pages: int
    pages_needing_ocr: List[int]
    pages_with_text: List[int]
    pages_with_images: List[int]
    empty_pages: List[int]
    overall_recommendation: str
    confidence: float
    page_analyses: List[PageAnalysis]
    
    def get_ocr_pages(self) -> List[int]:
        """Get list of pages that need OCR (1-indexed)."""
        return self.pages_needing_ocr
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "pages_needing_ocr": self.pages_needing_ocr,
            "pages_with_text": self.pages_with_text,
            "pages_with_images": self.pages_with_images,
            "empty_pages": self.empty_pages,
            "overall_recommendation": self.overall_recommendation,
            "confidence": self.confidence,
            "pages": [page.to_dict() for page in self.page_analyses]
        }


class OCRDetector:
    """Simple API for OCR detection."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 text_quality_threshold: float = 0.5,
                 min_text_length: int = 50):
        """Initialize OCR detector with configurable thresholds.
        
        Args:
            confidence_threshold: Minimum confidence for recommendations (0-1)
            text_quality_threshold: Minimum text quality score (0-1)
            min_text_length: Minimum text length to consider page as having text
        """
        self.confidence_threshold = confidence_threshold
        self.text_quality_threshold = text_quality_threshold
        self.min_text_length = min_text_length
    
    def analyze_pdf(self, pdf_path: Union[str, Path], 
                    include_details: bool = False) -> PDFAnalysisResult:
        """Analyze a PDF file for OCR requirements.
        
        Args:
            pdf_path: Path to the PDF file
            include_details: Include detailed analysis information
            
        Returns:
            PDFAnalysisResult with complete analysis
        """
        pdf_path = Path(pdf_path)
        
        with PDFAnalyzer(pdf_path) as analyzer:
            results = analyzer.analyze_all_pages()
            summary = analyzer.get_summary(results)
            
            # Process results into simplified format
            page_analyses = []
            pages_needing_ocr = []
            pages_with_text = []
            pages_with_images = []
            empty_pages = []
            
            for result in results:
                # Determine recommendation
                recommendation = self._get_page_recommendation(result)
                
                # Create simplified page analysis
                page_analysis = PageAnalysis(
                    page_number=result.page_number + 1,  # Convert to 1-indexed
                    recommendation=recommendation,
                    confidence=result.confidence,
                    page_type=result.page_type.value,
                    text_length=result.text_length,
                    has_images=result.image_count > 0,
                    details=result.details if include_details else None
                )
                page_analyses.append(page_analysis)
                
                # Categorize pages
                page_num = result.page_number + 1
                if recommendation == PageRecommendation.NEEDS_OCR:
                    pages_needing_ocr.append(page_num)
                if result.text_length >= self.min_text_length:
                    pages_with_text.append(page_num)
                if result.image_count > 0:
                    pages_with_images.append(page_num)
                if result.page_type == PageType.EMPTY:
                    empty_pages.append(page_num)
            
            return PDFAnalysisResult(
                file_path=str(pdf_path),
                total_pages=summary["total_pages"],
                pages_needing_ocr=pages_needing_ocr,
                pages_with_text=pages_with_text,
                pages_with_images=pages_with_images,
                empty_pages=empty_pages,
                overall_recommendation=summary["recommended_action"],
                confidence=summary["average_confidence"],
                page_analyses=page_analyses
            )
    
    def get_pages_needing_ocr(self, pdf_path: Union[str, Path]) -> List[int]:
        """Get list of page numbers that need OCR processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of 1-indexed page numbers needing OCR
        """
        result = self.analyze_pdf(pdf_path, include_details=False)
        return result.get_ocr_pages()
    
    def analyze_page(self, pdf_path: Union[str, Path], 
                    page_number: int) -> PageAnalysis:
        """Analyze a specific page.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            PageAnalysis for the specified page
        """
        pdf_path = Path(pdf_path)
        
        with PDFAnalyzer(pdf_path) as analyzer:
            # Convert to 0-indexed for internal use
            result = analyzer.analyze_page(page_number - 1)
            
            recommendation = self._get_page_recommendation(result)
            
            return PageAnalysis(
                page_number=page_number,
                recommendation=recommendation,
                confidence=result.confidence,
                page_type=result.page_type.value,
                text_length=result.text_length,
                has_images=result.image_count > 0,
                details=result.details
            )
    
    def quick_check(self, pdf_path: Union[str, Path]) -> str:
        """Quick check for OCR requirements.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            String recommendation: "OCR_REQUIRED", "NO_OCR_NEEDED", or "OCR_RECOMMENDED"
        """
        result = self.analyze_pdf(pdf_path, include_details=False)
        return result.overall_recommendation
    
    def _get_page_recommendation(self, result: AnalysisResult) -> PageRecommendation:
        """Convert analysis result to simplified recommendation."""
        if result.page_type == PageType.EMPTY:
            return PageRecommendation.EMPTY
        
        # Check text quality if available
        text_quality = result.details.get("text_quality", {})
        ocr_quality = text_quality.get("ocr_quality_score", 1.0)
        
        # Determine based on page type and quality
        if result.page_type == PageType.SCANNED:
            return PageRecommendation.NEEDS_OCR
        elif result.page_type == PageType.TEXT:
            if (result.text_length < self.min_text_length or 
                ocr_quality < self.text_quality_threshold or
                result.confidence < self.confidence_threshold):
                return PageRecommendation.NEEDS_OCR
            else:
                return PageRecommendation.NO_OCR_NEEDED
        elif result.page_type == PageType.MIXED:
            if (result.text_length < self.min_text_length * 2 or
                ocr_quality < self.text_quality_threshold or
                result.confidence < self.confidence_threshold):
                return PageRecommendation.NEEDS_OCR
            else:
                return PageRecommendation.OCR_OPTIONAL
        
        return PageRecommendation.OCR_OPTIONAL