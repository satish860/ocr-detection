"""OCR Detection Library - Analyze PDF pages to determine if they contain text or are scanned images."""

from .detector import PDFAnalyzer, PageType, AnalysisResult
from .analyzer import ContentAnalyzer

__version__ = "0.1.0"
__all__ = ["PDFAnalyzer", "PageType", "AnalysisResult", "ContentAnalyzer"]
