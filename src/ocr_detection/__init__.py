"""OCR Detection Library - Analyze PDF pages to determine if they contain text or are scanned images."""

from .analyzer import ContentAnalyzer
from .api import OCRDetector, PageAnalysis, PageRecommendation, PDFAnalysisResult
from .detector import AnalysisResult, PageType, PDFAnalyzer
from .simple import OCRDetection, OCRStatus, detect_ocr

__version__ = "0.1.0"
__all__ = [
    "AnalysisResult",
    "ContentAnalyzer",
    # Simple API
    "OCRDetection",
    # Enhanced API
    "OCRDetector",
    "OCRStatus",
    "PDFAnalysisResult",
    # Core classes
    "PDFAnalyzer",
    "PageAnalysis",
    "PageRecommendation",
    "PageType",
    "detect_ocr",
]
