"""Simple OCR detection interface."""

from enum import Enum
from pathlib import Path
from typing import Any

from ._detector import AnalysisResult, PageType, PDFAnalyzer


class OCRStatus(Enum):
    """OCR requirement status for document."""

    TRUE = "true"  # All pages need OCR
    FALSE = "false"  # No pages need OCR
    PARTIAL = "partial"  # Some pages need OCR


class OCRDetection:
    """Simple OCR detection interface."""

    def __init__(self, confidence_threshold: float = 0.5, parallel: bool = True,
                 include_images: bool = False, image_format: str = "png", image_dpi: int = 72,
                 accuracy_mode: bool = False):
        """Initialize OCR detection.

        Args:
            confidence_threshold: Minimum confidence for determining OCR need (0-1)
            parallel: Enable parallel processing for faster analysis
            include_images: Include base64-encoded page images in results
            image_format: Image format for rendering ("png" or "jpeg")
            image_dpi: Resolution for image rendering (default 72)
            accuracy_mode: Use maximum accuracy mode (slower but more precise)
                          Default False uses fast mode with 40x+ speedup
        """
        self.confidence_threshold = confidence_threshold
        self.parallel = parallel
        self.include_images = include_images
        self.image_format = image_format
        self.image_dpi = image_dpi
        self.accuracy_mode = accuracy_mode

    def detect(
        self, document: str | Path, parallel: bool | None = None, max_workers: int | None = None,
        include_images: bool | None = None
    ) -> dict[str, Any]:
        """Detect which pages need OCR processing.

        Args:
            document: Path to PDF document
            parallel: Override default parallel setting (optional)
            max_workers: Number of worker threads for parallel processing (optional)
            include_images: Override default include_images setting (optional)

        Returns:
            Dictionary with:
                - status: "true" (all pages), "false" (no pages), or "partial" (some pages)
                - pages: List of page numbers needing OCR (1-indexed)
                - page_images: Dict mapping page numbers to base64 images (if include_images=True)
        """
        document = Path(document)

        if not document.exists():
            raise FileNotFoundError(f"Document not found: {document}")

        # Use instance settings if not specified
        if parallel is None:
            parallel = self.parallel
        if include_images is None:
            include_images = self.include_images

        pages_needing_ocr = []
        page_images = {}
        total_pages = 0

        with PDFAnalyzer(document, accuracy_mode=self.accuracy_mode) as analyzer:
            # Use parallel or sequential based on settings
            if parallel:
                results = analyzer.analyze_all_pages_parallel(
                    max_workers=max_workers,
                    include_images=include_images,
                    image_format=self.image_format,
                    image_dpi=self.image_dpi
                )
            else:
                results = analyzer.analyze_all_pages(
                    include_images=include_images,
                    image_format=self.image_format,
                    image_dpi=self.image_dpi
                )
            total_pages = len(results)

            for result in results:
                # Page needs OCR if it's scanned or has poor text quality
                needs_ocr = self._page_needs_ocr(result)

                if needs_ocr:
                    # Convert to 1-indexed page number (PDF pages start at 1, not 0)
                    page_number_1_indexed = result.page_number + 1
                    pages_needing_ocr.append(page_number_1_indexed)

                    # Add image if available and requested (only for pages needing OCR)
                    if include_images and result.page_image:
                        page_images[page_number_1_indexed] = result.page_image

        # Determine status
        if len(pages_needing_ocr) == 0:
            status = OCRStatus.FALSE
        elif len(pages_needing_ocr) == total_pages:
            status = OCRStatus.TRUE
        else:
            status = OCRStatus.PARTIAL

        result_dict = {"status": status.value, "pages": pages_needing_ocr}

        # Include page images if they were requested and collected
        if include_images:
            result_dict["page_images"] = page_images

        return result_dict

    def _page_needs_ocr(self, result: AnalysisResult) -> bool:
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
            return bool(
                result.text_length < 50
                or ocr_quality < 0.4
                or result.confidence < self.confidence_threshold
            )

        # Mixed pages - need OCR if text quality is poor
        if result.page_type == PageType.MIXED:
            return bool(
                result.text_length < 100
                or ocr_quality < 0.5
                or result.confidence < self.confidence_threshold
            )

        return False


# Convenience function for one-line usage
def detect_ocr(document: str | Path, parallel: bool = True, include_images: bool = False,
              image_format: str = "png", image_dpi: int = 72, accuracy_mode: bool = False) -> dict[str, Any]:
    """Quick function to detect OCR requirements.

    Args:
        document: Path to PDF document
        parallel: Use parallel processing for faster analysis
        include_images: Include base64-encoded page images in results
        image_format: Image format for rendering ("png" or "jpeg")
        image_dpi: Resolution for image rendering (default 72)
        accuracy_mode: Use maximum accuracy mode (slower but more precise)
                      Default False uses fast mode with 40x+ speedup

    Returns:
        Dictionary with status, pages needing OCR, and optionally page images
    """
    detector = OCRDetection(
        parallel=parallel,
        include_images=include_images,
        image_format=image_format,
        image_dpi=image_dpi,
        accuracy_mode=accuracy_mode
    )
    return detector.detect(document)
