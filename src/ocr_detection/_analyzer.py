"""Additional content analysis utilities."""

import re
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .detector import AnalysisResult


@dataclass
class TextMetrics:
    """Metrics about extracted text."""

    char_count: int
    word_count: int
    line_count: int
    avg_word_length: float
    has_structured_content: bool
    language_indicators: dict[str, float]
    ocr_quality_score: float
    text_density: float
    formatting_consistency: float


class ContentAnalyzer:
    """Advanced content analysis for PDF pages."""

    @staticmethod
    def analyze_text_quality(text: str) -> TextMetrics:
        """Analyze the quality and characteristics of extracted text."""
        if not text:
            return TextMetrics(0, 0, 0, 0.0, False, {}, 0.0, 0.0, 0.0)

        # Basic counts
        char_count = len(text)
        lines = text.split("\n")
        line_count = len(lines)

        # Word analysis
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        avg_word_length = statistics.mean(len(word) for word in words) if words else 0.0

        # Check for structured content
        has_structured_content = ContentAnalyzer._detect_structured_content(text)

        # Basic language detection
        language_indicators = ContentAnalyzer._analyze_language_indicators(text)

        # New metrics for OCR quality assessment
        ocr_quality_score = ContentAnalyzer._calculate_ocr_quality_score(text, words, lines)
        text_density = ContentAnalyzer._calculate_text_density(text, lines)
        formatting_consistency = ContentAnalyzer._analyze_formatting_consistency(text, lines)

        return TextMetrics(
            char_count=char_count,
            word_count=word_count,
            line_count=line_count,
            avg_word_length=avg_word_length,
            has_structured_content=has_structured_content,
            language_indicators=language_indicators,
            ocr_quality_score=ocr_quality_score,
            text_density=text_density,
            formatting_consistency=formatting_consistency,
        )

    @staticmethod
    def _detect_structured_content(text: str) -> bool:
        """Detect if text has structured content like tables, lists, etc."""
        # Check for common structured patterns
        patterns = [
            r"\d+\.\s+",  # Numbered lists
            r"[•\-\*]\s+",  # Bullet lists
            r"\|\s*\w+\s*\|",  # Table-like structures
            r"\n\s*\n\s*\n",  # Multiple blank lines (paragraphs)
            r":\s*$",  # Colon at end of line (labels)
            r"^\s*[A-Z][A-Z\s]+:",  # ALL CAPS labels
        ]

        structure_score = sum(1 for pattern in patterns if re.search(pattern, text, re.MULTILINE))
        return structure_score >= 2

    @staticmethod
    def _analyze_language_indicators(text: str) -> dict[str, float]:
        """Simple language analysis based on character patterns."""
        if not text:
            return {}

        total_chars = len(text)
        indicators = {}

        # English indicators
        english_common = len(
            re.findall(r"\b(the|and|or|is|are|was|were|to|of|in|for|with)\b", text, re.IGNORECASE)
        )
        indicators["english"] = min(english_common / (total_chars / 100), 1.0)

        # Numeric content
        numeric_chars = len(re.findall(r"[0-9]", text))
        indicators["numeric"] = numeric_chars / total_chars if total_chars > 0 else 0

        # Special characters (might indicate OCR errors)
        special_chars = len(re.findall(r"[^\w\s\.\,\!\?\:\;\-\(\)]", text))
        indicators["special_chars"] = special_chars / total_chars if total_chars > 0 else 0

        return indicators

    @staticmethod
    def detect_ocr_artifacts(text: str) -> dict[str, Any]:
        """Detect common OCR artifacts that might indicate poor text extraction."""
        if not text:
            return {"artifacts_found": [], "confidence": 1.0}

        artifacts = []

        # Common OCR mistakes
        ocr_patterns = [
            (r"[Il1|]{2,}", "vertical_line_confusion"),
            (r"rn(?=\w)", "rn_m_confusion"),
            (r"cl(?=\w)", "cl_d_confusion"),
            (r"(?<!\w)[O0](?=\w)", "o_zero_confusion"),
            (r"\b\w*[^\w\s]{2,}\w*\b", "excessive_special_chars"),
            (r"\b[A-Z]{1}[a-z]*[A-Z]{1}[a-z]*\b", "inconsistent_case"),
        ]

        total_matches = 0
        for pattern, artifact_type in ocr_patterns:
            matches = len(re.findall(pattern, text))
            if matches > 0:
                artifacts.append(
                    {
                        "type": artifact_type,
                        "count": matches,
                        "examples": re.findall(pattern, text)[:3],
                    }
                )
                total_matches += matches

        # Calculate confidence (lower is worse)
        text_length = len(text.split())
        error_rate = total_matches / text_length if text_length > 0 else 0
        confidence = max(0.1, 1.0 - error_rate * 2)

        return {"artifacts_found": artifacts, "confidence": confidence, "error_rate": error_rate}

    @staticmethod
    def suggest_processing_method(
        analysis_result: "AnalysisResult", text_metrics: TextMetrics | None = None
    ) -> dict[str, str]:
        """Suggest the best processing method based on analysis."""
        page_type = analysis_result.page_type
        confidence = analysis_result.confidence

        suggestions = {"method": "unknown", "reason": "", "confidence": "low"}

        if page_type.value == "text" and confidence > 0.8:
            suggestions.update(
                {
                    "method": "direct_extraction",
                    "reason": "High-quality text content detected",
                    "confidence": "high",
                }
            )
        elif page_type.value == "scanned" and confidence > 0.7:
            suggestions.update(
                {
                    "method": "ocr_required",
                    "reason": "Scanned content detected, OCR processing needed",
                    "confidence": "high",
                }
            )
        elif page_type.value == "mixed":
            suggestions.update(
                {
                    "method": "hybrid_processing",
                    "reason": "Mixed content - combine text extraction with OCR for images",
                    "confidence": "medium",
                }
            )
        elif confidence < 0.5:
            suggestions.update(
                {
                    "method": "manual_review",
                    "reason": "Uncertain classification - manual review recommended",
                    "confidence": "low",
                }
            )

        # Adjust based on text metrics if provided
        if (
            text_metrics
            and text_metrics.language_indicators.get("special_chars", 0) > 0.1
            and suggestions["method"] == "direct_extraction"
        ):
            suggestions.update(
                {
                    "method": "verify_with_ocr",
                    "reason": "High special character rate suggests possible extraction issues",
                    "confidence": "medium",
                }
            )

        return suggestions

    @staticmethod
    def _calculate_ocr_quality_score(text: str, words: list[str], lines: list[str]) -> float:
        """Calculate a quality score indicating if text appears to be from good OCR."""
        if not text or not words:
            return 0.0

        score = 0.0

        # 1. Dictionary word ratio (approximate)
        common_english_words = {
            "the",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "in",
            "for",
            "with",
            "that",
            "this",
            "have",
            "has",
            "had",
            "will",
            "would",
            "could",
            "should",
            "not",
            "but",
            "from",
            "they",
            "we",
            "he",
            "she",
            "it",
            "you",
            "me",
            "us",
            "all",
            "any",
            "some",
            "each",
            "every",
            "other",
            "than",
            "only",
            "such",
            "court",
            "case",
            "law",
            "order",
            "appeal",
            "petition",
            "plaintiff",
            "defendant",
        }

        dictionary_matches = sum(1 for word in words if word.lower() in common_english_words)
        dictionary_ratio = dictionary_matches / len(words) if words else 0
        score += min(dictionary_ratio * 0.3, 0.3)  # Max 0.3 points

        # 2. Consistent capitalization patterns
        capitalized_words = sum(1 for word in words if word[0].isupper() and len(word) > 1)
        all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        cap_consistency = (capitalized_words + all_caps_words) / len(words) if words else 0
        if 0.05 <= cap_consistency <= 0.4:  # Reasonable capitalization
            score += 0.2

        # 3. Proper sentence structure
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 5 <= avg_sentence_length <= 30:  # Reasonable sentence length
                score += 0.2

        # 4. Low special character noise
        special_char_ratio = len(re.findall(r"[^\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'\/]", text)) / len(
            text
        )
        if special_char_ratio < 0.05:  # Low noise
            score += 0.2

        # 5. Consistent line spacing/formatting
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            line_lengths = [len(line) for line in non_empty_lines]
            if line_lengths:
                line_length_std = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0
                avg_line_length = statistics.mean(line_lengths)
                cv = line_length_std / avg_line_length if avg_line_length > 0 else 0
                if cv < 1.0:  # Reasonably consistent line lengths
                    score += 0.1

        return min(score, 1.0)

    @staticmethod
    def _calculate_text_density(text: str, lines: list[str]) -> float:
        """Calculate text density - how much meaningful text per line."""
        if not text or not lines:
            return 0.0

        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            return 0.0

        total_words = len(re.findall(r"\b\w+\b", text))
        avg_words_per_line = total_words / len(non_empty_lines)

        # Normalize: 5-15 words per line is good density
        if avg_words_per_line >= 5:
            return min(avg_words_per_line / 15, 1.0)
        else:
            return avg_words_per_line / 5

    @staticmethod
    def _analyze_formatting_consistency(text: str, lines: list[str]) -> float:
        """Analyze how consistent the formatting appears (indicates good OCR)."""
        if not lines or len(lines) < 3:
            return 0.0

        score = 0.0

        # 1. Consistent indentation patterns
        indents = []
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                indents.append(leading_spaces)

        if indents:
            unique_indents = set(indents)
            if len(unique_indents) <= 5:  # Few distinct indentation levels
                score += 0.3

        # 2. Consistent paragraph breaks
        empty_line_count = sum(1 for line in lines if not line.strip())
        non_empty_count = len(lines) - empty_line_count

        if non_empty_count > 0:
            empty_ratio = empty_line_count / len(lines)
            if 0.1 <= empty_ratio <= 0.4:  # Reasonable paragraph breaks
                score += 0.3

        # 3. Consistent punctuation usage
        sentences_ending_properly = len(re.findall(r"[.!?]\s*$", text, re.MULTILINE))
        total_sentences = len(re.split(r"[.!?]+", text)) - 1

        if total_sentences > 0:
            proper_ending_ratio = sentences_ending_properly / total_sentences
            if proper_ending_ratio > 0.7:
                score += 0.4

        return min(score, 1.0)
