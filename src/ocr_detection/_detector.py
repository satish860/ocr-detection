"""Core PDF detection and analysis functionality."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber

from ._analyzer import TextMetrics


class PageType(Enum):
    """Enumeration of PDF page types."""

    TEXT = "text"
    SCANNED = "scanned"
    MIXED = "mixed"
    EMPTY = "empty"


@dataclass
class AnalysisResult:
    """Result of page analysis."""

    page_number: int
    page_type: PageType
    confidence: float
    text_ratio: float
    image_ratio: float
    text_length: int
    image_count: int
    details: dict[str, Any]


class PDFAnalyzer:
    """Main class for analyzing PDF content."""

    def __init__(self, pdf_path: str | Path):
        """Initialize the analyzer with a PDF file."""
        self.pdf_path = Path(pdf_path)
        self.doc: fitz.Document | None = None
        self.plumber_pdf: pdfplumber.PDF | None = None

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    def __enter__(self) -> "PDFAnalyzer":
        """Context manager entry."""
        self.doc = fitz.open(str(self.pdf_path))
        self.plumber_pdf = pdfplumber.open(str(self.pdf_path))
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: object | None
    ) -> None:
        """Context manager exit."""
        if self.doc:
            self.doc.close()
        if self.plumber_pdf:
            self.plumber_pdf.close()

    def analyze_page(self, page_num: int) -> AnalysisResult:
        """Analyze a single page to determine its type."""
        if not self.doc or not self.plumber_pdf:
            raise RuntimeError("PDF not opened. Use within context manager.")

        if page_num >= len(self.doc):
            raise IndexError(f"Page {page_num} does not exist. PDF has {len(self.doc)} pages.")

        # Get pages from both libraries
        fitz_page = self.doc[page_num]
        plumber_page = self.plumber_pdf.pages[page_num]

        # Extract text using both methods
        fitz_text = fitz_page.get_text().strip()
        plumber_text = plumber_page.extract_text() or ""
        plumber_text = plumber_text.strip()

        # Use the longer text extraction
        extracted_text = fitz_text if len(fitz_text) > len(plumber_text) else plumber_text
        text_length = len(extracted_text)

        # Get page dimensions
        page_rect = fitz_page.rect
        page_area = page_rect.width * page_rect.height

        # Analyze images with improved background detection
        image_info = self._analyze_images(fitz_page)
        total_image_area = image_info["total_area"]
        meaningful_image_area = image_info["meaningful_image_area"]
        content_image_count = len(image_info["content_images"])

        # Calculate ratios - use meaningful images instead of total
        text_ratio = self._calculate_text_ratio(extracted_text, page_area)
        image_ratio = meaningful_image_area / page_area if page_area > 0 else 0
        background_ratio = image_info["background_coverage_ratio"]

        # Get text quality metrics
        from ._analyzer import ContentAnalyzer

        text_metrics = ContentAnalyzer.analyze_text_quality(extracted_text)

        # Classify page type and calculate confidence with enhanced logic
        try:
            page_type, confidence = self._classify_page_enhanced(
                text_ratio,
                image_ratio,
                text_length,
                content_image_count,
                text_metrics,
                background_ratio,
            )
        except Exception:
            # Fallback to original classification if enhanced fails
            page_type, confidence = self._classify_page(
                text_ratio, image_ratio, text_length, content_image_count
            )

        # Prepare detailed information
        details = {
            "extracted_text_preview": (
                extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
            ),
            "page_dimensions": {"width": page_rect.width, "height": page_rect.height},
            "total_image_area": total_image_area,
            "meaningful_image_area": meaningful_image_area,
            "background_coverage_ratio": background_ratio,
            "content_image_count": content_image_count,
            "image_details": image_info["content_images"][:5],  # Show content images only
            "text_extraction_method": (
                "fitz" if len(fitz_text) > len(plumber_text) else "pdfplumber"
            ),
            "text_quality": {
                "ocr_quality_score": text_metrics.ocr_quality_score,
                "text_density": text_metrics.text_density,
                "formatting_consistency": text_metrics.formatting_consistency,
            },
        }

        return AnalysisResult(
            page_number=page_num,
            page_type=page_type,
            confidence=confidence,
            text_ratio=text_ratio,
            image_ratio=image_ratio,
            text_length=text_length,
            image_count=content_image_count,
            details=details,
        )

    def analyze_all_pages(self) -> list[AnalysisResult]:
        """Analyze all pages in the PDF."""
        if not self.doc:
            raise RuntimeError("PDF not opened. Use within context manager.")

        results = []
        for page_num in range(len(self.doc)):
            results.append(self.analyze_page(page_num))

        return results

    def analyze_all_pages_parallel(self, max_workers: int | None = None) -> list[AnalysisResult]:
        """Analyze all pages in the PDF using parallel processing.

        Args:
            max_workers: Maximum number of worker threads. Defaults to CPU count.

        Returns:
            List of AnalysisResult objects, ordered by page number.
        """
        if not self.doc:
            raise RuntimeError("PDF not opened. Use within context manager.")

        total_pages = len(self.doc)

        # For small PDFs, use sequential processing
        if total_pages <= 10:
            return self.analyze_all_pages()

        # Determine number of workers
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, total_pages, 8)
        else:
            max_workers = min(max_workers, total_pages)

        results = []

        # Create page chunks for better load distribution
        pages_per_worker = max(1, total_pages // max_workers)
        page_chunks = []
        for i in range(0, total_pages, pages_per_worker):
            chunk = list(range(i, min(i + pages_per_worker, total_pages)))
            page_chunks.append(chunk)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each chunk
            future_to_chunk = {}
            for chunk in page_chunks:
                future = executor.submit(self._analyze_pages_batch, chunk)
                future_to_chunk[future] = chunk

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception:
                    # If a chunk fails, fall back to sequential processing for those pages
                    chunk = future_to_chunk[future]
                    for page_num in chunk:
                        try:
                            result = self.analyze_page(page_num)
                            results.append(result)
                        except Exception as page_error:
                            # Log error but continue with other pages
                            print(f"Error analyzing page {page_num}: {page_error}")

        # Sort results by page number to maintain order
        results.sort(key=lambda r: r.page_number)
        return results

    def _analyze_pages_batch(self, page_numbers: list[int]) -> list[AnalysisResult]:
        """Analyze a batch of pages in a separate thread.

        This method creates its own PDF document instances to ensure thread safety.

        Args:
            page_numbers: List of page numbers to analyze.

        Returns:
            List of AnalysisResult objects for the batch.
        """
        results = []

        # Open separate PDF instances for thread safety
        with (
            fitz.open(str(self.pdf_path)) as thread_doc,
            pdfplumber.open(str(self.pdf_path)) as thread_plumber,
        ):
            for page_num in page_numbers:
                # Get pages from both libraries
                fitz_page = thread_doc[page_num]
                plumber_page = thread_plumber.pages[page_num]

                # Extract text using both methods
                fitz_text = fitz_page.get_text().strip()
                plumber_text = plumber_page.extract_text() or ""
                plumber_text = plumber_text.strip()

                # Use the longer text extraction
                extracted_text = fitz_text if len(fitz_text) > len(plumber_text) else plumber_text
                text_length = len(extracted_text)

                # Get page dimensions
                page_rect = fitz_page.rect
                page_area = page_rect.width * page_rect.height

                # Analyze images with improved background detection
                image_info = self._analyze_images(fitz_page)
                total_image_area = image_info["total_area"]
                meaningful_image_area = image_info["meaningful_image_area"]
                content_image_count = len(image_info["content_images"])

                # Calculate ratios - use meaningful images instead of total
                text_ratio = self._calculate_text_ratio(extracted_text, page_area)
                image_ratio = meaningful_image_area / page_area if page_area > 0 else 0
                background_ratio = image_info["background_coverage_ratio"]

                # Get text quality metrics
                from ._analyzer import ContentAnalyzer

                text_metrics = ContentAnalyzer.analyze_text_quality(extracted_text)

                # Classify page type and calculate confidence with enhanced logic
                try:
                    page_type, confidence = self._classify_page_enhanced(
                        text_ratio,
                        image_ratio,
                        text_length,
                        content_image_count,
                        text_metrics,
                        background_ratio,
                    )
                except Exception:
                    # Fallback to original classification if enhanced fails
                    page_type, confidence = self._classify_page(
                        text_ratio, image_ratio, text_length, content_image_count
                    )

                # Prepare detailed information
                details = {
                    "extracted_text_preview": (
                        extracted_text[:200] + "..."
                        if len(extracted_text) > 200
                        else extracted_text
                    ),
                    "page_dimensions": {"width": page_rect.width, "height": page_rect.height},
                    "total_image_area": total_image_area,
                    "meaningful_image_area": meaningful_image_area,
                    "background_coverage_ratio": background_ratio,
                    "content_image_count": content_image_count,
                    "image_details": image_info["content_images"][:5],  # Show content images only
                    "text_extraction_method": (
                        "fitz" if len(fitz_text) > len(plumber_text) else "pdfplumber"
                    ),
                    "text_quality": {
                        "ocr_quality_score": text_metrics.ocr_quality_score,
                        "text_density": text_metrics.text_density,
                        "formatting_consistency": text_metrics.formatting_consistency,
                    },
                }

                result = AnalysisResult(
                    page_number=page_num,
                    page_type=page_type,
                    confidence=confidence,
                    text_ratio=text_ratio,
                    image_ratio=image_ratio,
                    text_length=text_length,
                    image_count=content_image_count,
                    details=details,
                )

                results.append(result)

        return results

    def analyze_all_pages_auto(
        self, parallel: bool = True, max_workers: int | None = None
    ) -> list[AnalysisResult]:
        """Analyze all pages with automatic method selection.

        Args:
            parallel: If True, use parallel processing for large PDFs.
            max_workers: Maximum number of worker threads (only used if parallel=True).

        Returns:
            List of AnalysisResult objects, ordered by page number.
        """
        if parallel and self.doc and len(self.doc) > 10:
            return self.analyze_all_pages_parallel(max_workers)
        else:
            return self.analyze_all_pages()

    def get_summary(
        self, results: list[AnalysisResult] | None = None, parallel: bool = False
    ) -> dict[str, Any]:
        """Get a summary of the analysis results."""
        if results is None:
            results = self.analyze_all_pages_parallel() if parallel else self.analyze_all_pages()

        total_pages = len(results)
        type_counts: dict[str, int] = {}
        confidence_sum = 0.0
        text_quality_sum = 0
        text_quality_count = 0

        for result in results:
            page_type = result.page_type.value
            type_counts[page_type] = type_counts.get(page_type, 0) + 1
            confidence_sum += result.confidence

            # Track text quality for pages with text
            if result.text_length > 50 and "text_quality" in result.details:
                quality_score = result.details["text_quality"]["ocr_quality_score"]
                text_quality_sum += quality_score
                text_quality_count += 1

        type_percentages = {
            page_type: (count / total_pages) * 100 for page_type, count in type_counts.items()
        }

        avg_text_quality = text_quality_sum / text_quality_count if text_quality_count > 0 else 0

        # Identify problematic pages
        problematic_pages = self._identify_problematic_pages(results)

        return {
            "total_pages": total_pages,
            "type_counts": type_counts,
            "type_percentages": type_percentages,
            "average_confidence": confidence_sum / total_pages if total_pages > 0 else 0,
            "average_text_quality": avg_text_quality,
            "recommended_action": self._get_recommendation_enhanced(
                type_counts, total_pages, avg_text_quality, results
            ),
            "problematic_pages": problematic_pages,
        }

    def _analyze_images(self, page: fitz.Page) -> dict[str, Any]:
        """Analyze images on a page, distinguishing between content and OCR backgrounds."""
        images = []
        total_area = 0
        background_area = 0
        content_images = []

        # Get page dimensions for comparison
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        try:
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    # Get image info
                    xref = img[0]
                    image_dict = page.get_image_rects(xref)

                    if image_dict:
                        for rect in image_dict:
                            area = rect.width * rect.height
                            total_area += area

                            # Calculate coverage ratio
                            coverage_ratio = area / page_area if page_area > 0 else 0

                            image_info = {
                                "index": img_index,
                                "area": area,
                                "width": rect.width,
                                "height": rect.height,
                                "xref": xref,
                                "coverage_ratio": coverage_ratio,
                                "is_likely_background": False,
                            }

                            # Detect likely OCR background images
                            # High coverage (>80%) suggests background
                            # Images close to page size are likely OCR backgrounds
                            if coverage_ratio > 0.8 or (
                                rect.width > page_rect.width * 0.9
                                and rect.height > page_rect.height * 0.9
                            ):
                                image_info["is_likely_background"] = True
                                background_area += area
                            else:
                                content_images.append(image_info)

                            images.append(image_info)

                except Exception:
                    # Skip problematic images
                    continue

        except Exception:
            # If image analysis fails, continue with empty results
            pass

        # Calculate meaningful image area (excluding likely backgrounds)
        meaningful_image_area = total_area - background_area

        return {
            "images": images,
            "total_area": total_area,
            "background_area": background_area,
            "meaningful_image_area": meaningful_image_area,
            "content_images": content_images,
            "has_background_images": background_area > 0,
            "background_coverage_ratio": background_area / page_area if page_area > 0 else 0,
        }

    def _calculate_text_ratio(self, text: str, page_area: float) -> float:
        """Calculate text density ratio."""
        if not text or page_area <= 0:
            return 0.0

        # Simple heuristic: assume average character takes ~8x12 pixels
        char_area = len(text) * 96  # 8x12 pixels per character
        ratio = min(char_area / page_area, 1.0)  # Cap at 1.0

        return ratio

    def _classify_page_enhanced(
        self,
        text_ratio: float,
        image_ratio: float,
        text_length: int,
        content_image_count: int,
        text_metrics: TextMetrics,
        background_ratio: float,
    ) -> tuple[PageType, float]:
        """Enhanced page classification using text quality and background detection."""

        # Empty page check
        if text_length < 10 and content_image_count == 0:
            return PageType.EMPTY, 0.95

        # Strong text indicators with quality assessment
        if text_length > 200:  # Reasonable amount of text
            # Check text quality scores
            quality_score = text_metrics.ocr_quality_score
            text_density = text_metrics.text_density

            # High-quality text extraction (likely good OCR or native text)
            if quality_score > 0.6 and text_density > 0.4:
                # Even with background images, if text quality is good, it's usable
                if background_ratio > 0.5:  # Has OCR background
                    confidence = min(0.9, 0.7 + quality_score * 0.3)
                    return PageType.TEXT, confidence
                elif content_image_count <= 2:  # Few meaningful images
                    confidence = min(0.95, 0.8 + quality_score * 0.2)
                    return PageType.TEXT, confidence
                else:  # Some meaningful images but good text
                    confidence = min(0.85, 0.7 + quality_score * 0.2)
                    return PageType.MIXED, confidence

            # Moderate text quality
            elif quality_score > 0.4 or text_length > 500:
                if content_image_count <= 1:
                    confidence = 0.6 + quality_score * 0.2
                    return PageType.TEXT, confidence
                else:
                    confidence = 0.6 + quality_score * 0.15
                    return PageType.MIXED, confidence

        # Moderate text content
        elif 50 <= text_length <= 200:
            quality_score = text_metrics.ocr_quality_score

            if quality_score > 0.5 and content_image_count <= 1:
                return PageType.TEXT, 0.7
            elif content_image_count > 2 or image_ratio > 0.4:
                return PageType.MIXED, 0.65
            else:
                return PageType.TEXT, 0.6

        # Low text content - focus on images
        else:  # text_length < 50
            if content_image_count > 0 or image_ratio > 0.2:
                confidence = min(0.9, 0.6 + image_ratio * 0.4)
                return PageType.SCANNED, confidence
            else:
                return PageType.EMPTY, 0.8

        # Fallback for any edge cases
        return PageType.MIXED, 0.5

    def _classify_page(
        self, text_ratio: float, image_ratio: float, text_length: int, image_count: int
    ) -> tuple[PageType, float]:
        """Classify page type and calculate confidence."""

        # Empty page
        if text_length < 10 and image_count == 0:
            return PageType.EMPTY, 0.95

        # Strong text indicators
        if text_length > 500 and text_ratio > 0.1:
            if image_ratio < 0.2:
                confidence = min(0.9, 0.7 + text_ratio * 0.5)
                return PageType.TEXT, confidence
            else:
                confidence = min(0.85, 0.6 + text_ratio * 0.3)
                return PageType.MIXED, confidence

        # Strong scanned indicators
        if text_length < 50 and image_ratio > 0.3:
            confidence = min(0.9, 0.6 + image_ratio * 0.4)
            return PageType.SCANNED, confidence

        # Moderate text with images
        if 50 <= text_length <= 500:
            if image_ratio > 0.4:
                return PageType.MIXED, 0.7
            elif text_ratio > 0.05:
                return PageType.TEXT, 0.75
            else:
                return PageType.SCANNED, 0.6

        # Default classification based on ratios
        if text_ratio > image_ratio:
            confidence = 0.5 + min(text_ratio, 0.3)
            return PageType.TEXT, confidence
        else:
            confidence = 0.5 + min(image_ratio, 0.3)
            return PageType.SCANNED, confidence

    def _get_recommendation_enhanced(
        self,
        type_counts: dict[str, int],
        total_pages: int,
        avg_text_quality: float,
        results: list[AnalysisResult],
    ) -> str:
        """Enhanced recommendation based on text quality and content analysis."""
        scanned_count = type_counts.get("scanned", 0)
        mixed_count = type_counts.get("mixed", 0)
        text_count = type_counts.get("text", 0)

        text_percentage = (text_count / total_pages) * 100
        mixed_percentage = (mixed_count / total_pages) * 100
        scanned_percentage = (scanned_count / total_pages) * 100

        # Count pages with good text extraction
        good_text_pages = 0
        total_text_length = 0

        for result in results:
            total_text_length += result.text_length
            if (
                result.text_length > 200
                and result.details.get("text_quality", {}).get("ocr_quality_score", 0) > 0.6
            ):
                good_text_pages += 1

        good_text_percentage = (good_text_pages / total_pages) * 100
        avg_text_per_page = total_text_length / total_pages

        # Decision logic prioritizing text quality
        if (
            (good_text_percentage > 60 and avg_text_quality > 0.6)
            or (text_percentage > 70 and avg_text_quality > 0.5)
            or (good_text_percentage > 30 and avg_text_per_page > 300)
        ):
            return "NO OCR NEEDED"
        elif (
            scanned_percentage > 70
            or (mixed_percentage > 80 and avg_text_quality < 0.4)
            or (text_percentage < 10 and scanned_percentage + mixed_percentage > 80)
        ):
            return "OCR REQUIRED"
        else:
            return "OCR RECOMMENDED"

    def _identify_problematic_pages(self, results: list[AnalysisResult]) -> dict[str, Any]:
        """Identify specific pages that may need OCR processing."""

        scanned_pages = []
        mixed_pages_needing_ocr = []
        low_quality_text_pages = []
        empty_pages = []

        for result in results:
            page_num = result.page_number + 1  # Convert to 1-based indexing

            if result.page_type == PageType.EMPTY:
                empty_pages.append(page_num)
            elif result.page_type == PageType.SCANNED:
                scanned_pages.append(page_num)
            elif result.page_type == PageType.MIXED:
                # Check if mixed page would benefit from OCR
                text_quality = result.details.get("text_quality", {})
                ocr_quality = text_quality.get("ocr_quality_score", 0)

                if result.text_length < 100 or ocr_quality < 0.4 or result.confidence < 0.7:
                    mixed_pages_needing_ocr.append(
                        {
                            "page": page_num,
                            "reason": "Low text quality or quantity",
                            "text_length": result.text_length,
                            "quality_score": ocr_quality,
                            "confidence": result.confidence,
                        }
                    )
            elif result.page_type == PageType.TEXT:
                # Even text pages might need OCR if quality is very poor
                text_quality = result.details.get("text_quality", {})
                ocr_quality = text_quality.get("ocr_quality_score", 1.0)

                if result.text_length < 50 or ocr_quality < 0.3 or result.confidence < 0.6:
                    low_quality_text_pages.append(
                        {
                            "page": page_num,
                            "reason": "Very poor text extraction quality",
                            "text_length": result.text_length,
                            "quality_score": ocr_quality,
                            "confidence": result.confidence,
                        }
                    )

        # Create summary
        total_needing_ocr = (
            len(scanned_pages) + len(mixed_pages_needing_ocr) + len(low_quality_text_pages)
        )

        return {
            "scanned_pages": scanned_pages,
            "mixed_pages_needing_ocr": mixed_pages_needing_ocr,
            "low_quality_text_pages": low_quality_text_pages,
            "empty_pages": empty_pages,
            "total_pages_needing_ocr": total_needing_ocr,
            "summary": self._create_page_summary(
                scanned_pages, mixed_pages_needing_ocr, low_quality_text_pages, empty_pages
            ),
        }

    def _create_page_summary(
        self,
        scanned_pages: list[int],
        mixed_pages: list[dict],
        low_quality_pages: list[dict],
        empty_pages: list[int],
    ) -> str:
        """Create a human-readable summary of problematic pages."""
        if not any([scanned_pages, mixed_pages, low_quality_pages, empty_pages]):
            return "All pages have good text extraction quality."

        summary_parts = []

        if scanned_pages:
            pages_str = self._format_page_list(scanned_pages)
            summary_parts.append(f"Scanned pages needing OCR: {pages_str}")

        if mixed_pages:
            page_nums = [p["page"] for p in mixed_pages]
            pages_str = self._format_page_list(page_nums)
            summary_parts.append(f"Mixed content pages that would benefit from OCR: {pages_str}")

        if low_quality_pages:
            page_nums = [p["page"] for p in low_quality_pages]
            pages_str = self._format_page_list(page_nums)
            summary_parts.append(f"Pages with poor text extraction quality: {pages_str}")

        if empty_pages:
            pages_str = self._format_page_list(empty_pages)
            summary_parts.append(f"Empty pages (no processing needed): {pages_str}")

        return " | ".join(summary_parts)

    def _format_page_list(self, pages: list[int]) -> str:
        """Format a list of page numbers for display."""
        if not pages:
            return "None"

        if len(pages) <= 5:
            return ", ".join(str(p) for p in pages)
        else:
            return f"{', '.join(str(p) for p in pages[:3])}, ... and {len(pages) - 3} more"

    def _get_recommendation(self, type_counts: dict[str, int], total_pages: int) -> str:
        """Get recommendation based on analysis results."""
        scanned_count = type_counts.get("scanned", 0)
        mixed_count = type_counts.get("mixed", 0)
        text_count = type_counts.get("text", 0)

        ocr_needed_count = scanned_count + mixed_count
        ocr_percentage = (ocr_needed_count / total_pages) * 100

        if ocr_percentage > 50:
            return "OCR REQUIRED"
        elif text_count > 0 and ocr_percentage < 25:
            return "NO OCR NEEDED"
        else:
            return "OCR RECOMMENDED"
