# OCR Detection Library

A Python library to analyze PDF pages and determine whether they contain extractable text or are scanned images requiring OCR processing.

**NEW in v0.3.0**: Smart Image Extraction provides **5x faster** performance for scanned PDFs with **33% less memory usage**!

## Features

- **Page Type Detection**: Automatically classifies PDF pages as text, scanned, mixed, or empty
- **Smart Image Extraction**: 5x faster image processing for scanned PDFs using embedded images
- **Base64 Image Output**: Get page images as base64-encoded strings for visualization
- **Parallel Processing**: Fast analysis of large PDFs using multi-threading
- **Confidence Scoring**: Reliability indicators for classifications
- **Memory Efficient**: 33% reduction in memory usage with optimized image handling
- **Simple API**: Easy-to-use interface with minimal complexity

## Installation

```bash
# Clone or download the project
cd ocr-detection

# Install with uv (recommended)
uv sync

# Or install with pip
pip install ocr-detection
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
    parallel=True,             # Enable parallel processing
    include_images=True,       # Include base64 page images
    image_format="png",        # Image format: "png" or "jpeg"
    image_dpi=150             # Image resolution (DPI)
)

# Analyze a document
result = detector.detect("document.pdf")

# With custom parallel settings
result = detector.detect("large_document.pdf", max_workers=4)
```

### Understanding Results

The library returns a dictionary with the following fields:

- **status**: Indicates the OCR requirement
  - `"true"` - All pages need OCR processing
  - `"false"` - No pages need OCR processing  
  - `"partial"` - Some pages need OCR processing

- **pages**: List of page numbers (1-indexed) that need OCR processing
  - Empty list when status is `"false"`
  - Contains all page numbers when status is `"true"`
  - Contains specific page numbers when status is `"partial"`

- **page_images**: Dictionary mapping page numbers to base64-encoded images (when `include_images=True`)
  - Only included for pages that need OCR processing
  - Page numbers are 1-indexed to match PDF page numbering
  - Images are base64-encoded PNG or JPEG strings

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

# Example 4: With base64 images
result = detect_ocr("document.pdf", include_images=True)
# {
#   "status": "partial", 
#   "pages": [2, 5], 
#   "page_images": {
#     2: "iVBORw0KGgoAAAANSUhEUgAA...",  # base64 PNG data
#     5: "iVBORw0KGgoAAAANSUhEUgAA..."   # base64 PNG data
#   }
# }

# Example 5: Custom image settings
result = detect_ocr(
    "document.pdf", 
    include_images=True,
    image_format="jpeg",  # Use JPEG instead of PNG
    image_dpi=200        # Higher resolution
)

# Example 6: With parallel processing for large PDFs
result = detect_ocr("large_document.pdf", parallel=True)
```

## Image Output Options

The library can generate base64-encoded images of pages that need OCR processing:

### Parameters
- **include_images**: `bool` - Enable base64 image output (default: `False`)
- **image_format**: `str` - Output format: `"png"` or `"jpeg"` (default: `"png"`)
- **image_dpi**: `int` - Resolution in DPI (default: `150`)

### Usage Notes
- Images are only generated for pages that need OCR processing
- **Smart extraction**: Scanned pages use embedded images for 5x faster processing
- Higher DPI values produce larger but clearer images (only affects rendered pages)
- PNG format preserves quality but has larger file sizes
- JPEG format is more compact but may have compression artifacts
- Page numbers in `page_images` match those in the `pages` list (1-indexed)

## Performance

### Version 0.3.0 Optimization

The library now features **Smart Image Extraction** for dramatically improved performance:

- **5x faster** processing for scanned PDFs (2.5s → 0.54s)
- **33% memory reduction** (116MB → 79MB)
- **8x smaller** image data (15.9MB → 2.0MB)
- **20x faster** per-image processing (1.2s → 0.06s per image)

### How It Works

- **Scanned PDFs**: Extracts original embedded JPEG images directly (no re-rendering)
- **Text PDFs**: Uses traditional rendering for vector content
- **Quality Preservation**: Maintains original image compression and quality
- **Thread Safety**: Works seamlessly with parallel processing

### Automatic Optimization

The library automatically optimizes performance based on document size and content:
- Documents with ≤10 pages use sequential processing
- Larger documents use parallel processing with configurable worker threads
- Parallel processing provides 3-8x performance improvement for large documents
- Smart image extraction eliminates unnecessary rendering overhead

## License

MIT License - see LICENSE file for details