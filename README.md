# OCR Detection Library

A Python library to analyze PDF pages and determine whether they contain extractable text or are scanned images requiring OCR processing.

## Features

- **Page Type Detection**: Automatically classifies PDF pages as:
  - `text`: Pages with extractable text content
  - `scanned`: Pages that are primarily scanned images
  - `mixed`: Pages with both text and significant image content
  - `empty`: Pages with minimal content

- **Content Analysis**: Advanced text quality metrics and OCR artifact detection
- **CLI Interface**: Easy-to-use command-line tool
- **Multiple Output Formats**: JSON, CSV, and text summary formats
- **Confidence Scoring**: Reliability indicators for classifications

## Installation

```bash
# Clone or download the project
cd ocr-detection

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

### Command Line Usage

```bash
# Basic analysis
uv run ocr-detect document.pdf

# Analyze specific page
uv run ocr-detect document.pdf --page 0

# Generate JSON output
uv run ocr-detect document.pdf --format json --output results.json

# Verbose analysis with text preview
uv run ocr-detect document.pdf --verbose --include-text

# CSV export with custom confidence threshold
uv run ocr-detect document.pdf --format csv --confidence-threshold 0.8
```

### Python API Usage

```python
from ocr_detection import PDFAnalyzer, ContentAnalyzer

# Analyze a PDF file
with PDFAnalyzer("document.pdf") as analyzer:
    # Analyze all pages
    results = analyzer.analyze_all_pages()
    
    # Get summary statistics
    summary = analyzer.get_summary(results)
    
    # Analyze specific page
    page_result = analyzer.analyze_page(0)
    
    print(f"Page type: {page_result.page_type.value}")
    print(f"Confidence: {page_result.confidence:.2f}")
    print(f"Recommendation: {summary['recommended_action']}")

# Advanced content analysis
text = "Sample extracted text..."
metrics = ContentAnalyzer.analyze_text_quality(text)
artifacts = ContentAnalyzer.detect_ocr_artifacts(text)
```

## Example Output

```
PDF CONTENT ANALYSIS SUMMARY
============================================================

Total Pages: 10
Average Confidence: 0.85

Page Type Distribution:
  Text    :   6 pages ( 60.0%)
  Scanned :   3 pages ( 30.0%)
  Mixed   :   1 pages ( 10.0%)

Recommendation: Consider OCR for optimal text extraction

   Pages with low confidence (< 0.5):
  Page 7: mixed (confidence: 0.45)
```

## Testing

```bash
# Run unit tests
uv run pytest tests/

# Run basic functionality test
uv run python test_basic.py

# Run with coverage (if pytest-cov is installed)
uv run pytest tests/ --cov=ocr_detection
```

## Use Cases

- **Document Processing Pipelines**: Automatically determine optimal text extraction method
- **OCR Pre-processing**: Identify which pages need OCR vs direct text extraction
- **Content Quality Assessment**: Evaluate PDF text extraction reliability
- **Batch Document Analysis**: Process large collections of PDF files efficiently

## Technical Details

The library uses multiple detection methods:

1. **Text Extraction**: Uses both PyMuPDF and pdfplumber for robust text extraction
2. **Image Analysis**: Detects and measures embedded images using PyMuPDF
3. **Content Ratios**: Calculates text-to-image ratios for classification
4. **Quality Metrics**: Analyzes text characteristics and OCR artifacts
5. **Confidence Scoring**: Provides reliability indicators based on multiple factors

## Dependencies

- Python 3.13+
- PyMuPDF (fitz) - PDF processing and image extraction
- pdfplumber - Alternative text extraction
- click - CLI interface

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]