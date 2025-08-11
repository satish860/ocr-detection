"""OCR Detection Library - Analyze PDF pages to determine if they contain text or are scanned images."""

from .detector import PDFAnalyzer, PageType, AnalysisResult
from .analyzer import ContentAnalyzer
from .api import OCRDetector, PageRecommendation, PageAnalysis, PDFAnalysisResult
from .simple import OCRDetection, OCRStatus, detect_ocr

__version__ = "0.1.0"
__all__ = [
    # Core classes
    "PDFAnalyzer", 
    "PageType", 
    "AnalysisResult", 
    "ContentAnalyzer",
    # Enhanced API
    "OCRDetector",
    "PageRecommendation",
    "PageAnalysis", 
    "PDFAnalysisResult",
    # Simple API
    "OCRDetection",
    "OCRStatus", 
    "detect_ocr"
]
