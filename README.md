# OCR Detection Library

A Python library to analyze PDF pages and determine whether they contain extractable text or are scanned images requiring OCR processing.

## Features

- **Page Type Detection**: Automatically classifies PDF pages as text, scanned, mixed, or empty
- **Parallel Processing**: Fast analysis of large PDFs using multi-threading
- **Confidence Scoring**: Reliability indicators for classifications
- **Simple API**: Easy-to-use interface with minimal complexity

## Installation

```bash
# Clone or download the project
cd ocr-detection

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Usage

### Quick Start

```python
from ocr_detection import detect_ocr

# Analyze a PDF document
result = detect_ocr("document.pdf")

print(result)
# Output: {"status": "partial", "pages": [1, 3, 7, 12]}

# Check the status
if result['status'] == "true":
    print("All pages need OCR")
elif result['status'] == "false":
    print("No pages need OCR")
else:  # partial
    print(f"Pages needing OCR: {result['pages']}")
```

### Using the OCRDetection Class

```python
from ocr_detection import OCRDetection

# Initialize detector with options
detector = OCRDetection(
    confidence_threshold=0.5,  # Minimum confidence for OCR detection
    parallel=True              # Enable parallel processing
)

# Analyze a document
result = detector.detect("document.pdf")

# With custom parallel settings
result = detector.detect("large_document.pdf", max_workers=4)
```

### Understanding Results

The library returns a simple dictionary with two fields:

- **status**: Indicates the OCR requirement
  - `"true"` - All pages need OCR processing
  - `"false"` - No pages need OCR processing  
  - `"partial"` - Some pages need OCR processing

- **pages**: List of page numbers (1-indexed) that need OCR processing
  - Empty list when status is `"false"`
  - Contains all page numbers when status is `"true"`
  - Contains specific page numbers when status is `"partial"`

### Examples

```python
from ocr_detection import detect_ocr

# Example 1: Fully text-based PDF
result = detect_ocr("text_document.pdf")
# {"status": "false", "pages": []}

# Example 2: Scanned PDF
result = detect_ocr("scanned_document.pdf")
# {"status": "true", "pages": [1, 2, 3, 4, 5]}

# Example 3: Mixed content PDF
result = detect_ocr("mixed_document.pdf")
# {"status": "partial", "pages": [2, 5, 8]}

# Example 4: With parallel processing for large PDFs
result = detect_ocr("large_document.pdf", parallel=True)
```

## Performance

The library automatically optimizes performance based on document size:
- Documents with ≤10 pages use sequential processing
- Larger documents use parallel processing with configurable worker threads
- Parallel processing provides 3-8x performance improvement for large documents

## License

MIT License - see LICENSE file for details