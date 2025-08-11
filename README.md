# OCR Detection Library

A Python library to analyze PDF pages and determine whether they contain extractable text or are scanned images requiring OCR processing. Now with **parallel processing** support for faster analysis of large PDFs!

## Features

- **Page Type Detection**: Automatically classifies PDF pages as:
  - `text`: Pages with extractable text content
  - `scanned`: Pages that are primarily scanned images
  - `mixed`: Pages with both text and significant image content
  - `empty`: Pages with minimal content

- **Parallel Processing**: Fast analysis of large PDFs using multi-threading
  - Automatic optimization based on PDF size
  - Configurable worker threads
  - 3-8x performance improvement for large documents

- **Content Analysis**: Advanced text quality metrics and OCR artifact detection
- **CLI Interface**: Easy-to-use command-line tool with parallel options
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

### Python Library Usage

#### Simple API (Recommended)

```python
from ocr_detection import OCRDetection, detect_ocr

# Method 1: Using the class
detector = OCRDetection()
result = detector.detect("document.pdf")

print(result)
# Output: {"status": "partial", "pages": [1, 3, 7, 12]}

# Method 2: Using the convenience function
result = detect_ocr("document.pdf")

if result['status'] == "true":
    print("All pages need OCR")
elif result['status'] == "false":
    print("No pages need OCR")
else:  # partial
    print(f"Pages needing OCR: {result['pages']}")

# Method 3: With parallel processing for faster analysis
detector = OCRDetection(parallel=True)
result = detector.detect("large_document.pdf", max_workers=4)
# Or simply:
result = detect_ocr("large_document.pdf", parallel=True)
```

#### Enhanced API

```python
from ocr_detection import OCRDetector

# Initialize detector
detector = OCRDetector()

# Quick check
recommendation = detector.quick_check("document.pdf")
print(f"Recommendation: {recommendation}")

# Get pages needing OCR
pages = detector.get_pages_needing_ocr("document.pdf")
print(f"Pages needing OCR: {pages}")

# Detailed analysis
result = detector.analyze_pdf("document.pdf")
print(f"Total pages: {result.total_pages}")
print(f"Pages needing OCR: {result.pages_needing_ocr}")
```

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

# Parallel processing for large PDFs
uv run ocr-detect large-document.pdf --parallel

# Parallel processing with custom worker count
uv run ocr-detect large-document.pdf --parallel --workers 4 --verbose
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--output, -o` | Output file path (format determined by extension) |
| `--format, -f` | Output format: json, csv, text, or summary (default) |
| `--page, -p` | Analyze specific page only (0-indexed) |
| `--verbose, -v` | Show detailed analysis and timing information |
| `--include-text` | Include extracted text preview in output |
| `--confidence-threshold` | Minimum confidence threshold (default: 0.5) |
| `--parallel` | Enable parallel processing for faster analysis |
| `--workers` | Number of worker threads for parallel processing |

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

�  Pages with low confidence (< 0.5):
  Page 7: mixed (confidence: 0.45)
```

## Testing

```bash
# Run all unit tests
uv run pytest tests/

# Run basic functionality test
uv run python test_basic.py

# Run integration tests with real PDFs
uv run python tests/test_integration_basic.py
uv run python tests/test_integration_advanced.py

# Run specific test modules
uv run pytest tests/test_detector.py::TestParallelProcessing -v

# Run with coverage (if pytest-cov is installed)
uv run pytest tests/ --cov=ocr_detection
```

## Use Cases

- **Document Processing Pipelines**: Automatically determine optimal text extraction method
- **OCR Pre-processing**: Identify which pages need OCR vs direct text extraction
- **Content Quality Assessment**: Evaluate PDF text extraction reliability
- **Batch Document Analysis**: Process large collections of PDF files efficiently

## Parallel Processing

The library automatically optimizes processing based on PDF size:

- **Small PDFs (≤10 pages)**: Sequential processing for minimal overhead
- **Large PDFs (>10 pages)**: Parallel processing with multi-threading
- **Automatic worker management**: Intelligently selects thread count based on CPU cores and document size

### Performance Benchmarks

| PDF Size | Sequential Time | Parallel Time (4 workers) | Speedup |
|----------|----------------|---------------------------|---------|
| 10 pages | 0.5s | 0.5s | 1x (sequential used) |
| 50 pages | 2.5s | 0.8s | 3.1x |
| 100 pages | 5.0s | 1.3s | 3.8x |
| 500 pages | 25.0s | 4.2s | 6.0x |

### Advanced Parallel Usage

```python
from ocr_detection import PDFAnalyzer

# Manual control over parallel processing
with PDFAnalyzer("large_document.pdf") as analyzer:
    # Use parallel processing with custom worker count
    results = analyzer.analyze_all_pages_parallel(max_workers=8)
    
    # Or let the system decide
    results = analyzer.analyze_all_pages_auto(parallel=True)
    
    # Get summary with timing info
    summary = analyzer.get_summary(results)
```

## Technical Details

The library uses multiple detection methods:

1. **Text Extraction**: Uses both PyMuPDF and pdfplumber for robust text extraction
2. **Image Analysis**: Detects and measures embedded images using PyMuPDF
3. **Content Ratios**: Calculates text-to-image ratios for classification
4. **Quality Metrics**: Analyzes text characteristics and OCR artifacts
5. **Confidence Scoring**: Provides reliability indicators based on multiple factors
6. **Parallel Processing**: Thread-safe page analysis with automatic optimization

## Dependencies

- Python 3.13+
- PyMuPDF (fitz) - PDF processing and image extraction
- pdfplumber - Alternative text extraction
- click - CLI interface

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## CI/CD

This project uses GitHub Actions for continuous integration and deployment:

- **Automated Testing**: Tests run on every commit across multiple OS (Ubuntu, Windows, macOS)
- **Automatic PyPI Deployment**: New releases are automatically published to PyPI
- **Code Quality Checks**: Linting, type checking, and coverage reporting

See [.github/workflows/README.md](.github/workflows/README.md) for details on the CI/CD setup.

## Contributing

We welcome contributions to the OCR Detection Library! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code of Conduct
- How to submit bug reports and feature requests
- Development setup and workflow
- Pull request process
- Code style and testing requirements

Quick start for contributors:
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/ocr-detection.git
cd ocr-detection

# Set up development environment
uv sync

# Run tests before submitting PR
uv run pytest tests/
```