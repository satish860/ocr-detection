"""OCR Detection Library - Analyze PDF pages to determine if they contain text or are scanned images."""

from .simple import OCRDetection, OCRStatus, detect_ocr

__version__ = "0.2.0"
__all__ = [
    "OCRDetection",
    "OCRStatus",
    "detect_ocr",
]
