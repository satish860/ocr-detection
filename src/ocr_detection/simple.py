"""Simple OCR detection interface."""

from pathlib import Path
from typing import Union, Dict, List, Any
from enum import Enum

from .detector import PDFAnalyzer, PageType


class OCRStatus(Enum):
    """OCR requirement status for document."""
    TRUE = "true"      # All pages need OCR
    FALSE = "false"    # No pages need OCR
    PARTIAL = "partial"  # Some pages need OCR


class OCRDetection:
    """Simple OCR detection interface."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize OCR detection.
        
        Args:
            confidence_threshold: Minimum confidence for determining OCR need (0-1)
        """
        self.confidence_threshold = confidence_threshold
    
    def detect(self, document: Union[str, Path]) -> Dict[str, Any]:
        """Detect which pages need OCR processing.
        
        Args:
            document: Path to PDF document
            
        Returns:
            Dictionary with:
                - status: "true" (all pages), "false" (no pages), or "partial" (some pages)
                - pages: List of page numbers needing OCR (1-indexed)
        """
        document = Path(document)
        
        if not document.exists():
            raise FileNotFoundError(f"Document not found: {document}")
        
        pages_needing_ocr = []
        total_pages = 0
        
        with PDFAnalyzer(document) as analyzer:
            results = analyzer.analyze_all_pages()
            total_pages = len(results)
            
            for result in results:
                # Page needs OCR if it's scanned or has poor text quality
                needs_ocr = self._page_needs_ocr(result)
                
                if needs_ocr:
                    # Add 1-indexed page number
                    pages_needing_ocr.append(result.page_number + 1)
        
        # Determine status
        if len(pages_needing_ocr) == 0:
            status = OCRStatus.FALSE
        elif len(pages_needing_ocr) == total_pages:
            status = OCRStatus.TRUE
        else:
            status = OCRStatus.PARTIAL
        
        return {
            "status": status.value,
            "pages": pages_needing_ocr
        }
    
    def _page_needs_ocr(self, result) -> bool:
        """Determine if a page needs OCR processing.
        
        Args:
            result: Page analysis result
            
        Returns:
            True if page needs OCR, False otherwise
        """
        # Scanned pages always need OCR
        if result.page_type == PageType.SCANNED:
            return True
        
        # Empty pages might have images that need OCR
        if result.page_type == PageType.EMPTY:
            return True
        
        # Check text quality for text and mixed pages
        text_quality = result.details.get("text_quality", {})
        ocr_quality = text_quality.get("ocr_quality_score", 1.0)
        
        # Text pages with poor quality need OCR
        if result.page_type == PageType.TEXT:
            if (result.text_length < 50 or 
                ocr_quality < 0.4 or 
                result.confidence < self.confidence_threshold):
                return True
            return False
        
        # Mixed pages - need OCR if text quality is poor
        if result.page_type == PageType.MIXED:
            if (result.text_length < 100 or
                ocr_quality < 0.5 or
                result.confidence < self.confidence_threshold):
                return True
            return False
        
        return False


# Convenience function for one-line usage
def detect_ocr(document: Union[str, Path]) -> Dict[str, Any]:
    """Quick function to detect OCR requirements.
    
    Args:
        document: Path to PDF document
        
    Returns:
        Dictionary with status and pages needing OCR
    """
    detector = OCRDetection()
    return detector.detect(document)