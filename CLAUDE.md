# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an OCR Detection Library that analyzes PDF pages to determine whether they contain extractable text or are scanned images requiring OCR processing. The library provides multiple APIs and interfaces for different use cases.

## Development Commands

**Package Management (using UV)**:
- `uv sync` - Install dependencies and set up development environment
- `uv add <package>` - Add new dependency
- `uv run <command>` - Run commands in the virtual environment

**Testing**:
- `uv run pytest tests/` - Run all tests
- `uv run pytest tests/test_detector.py` - Run specific test file
- `uv run pytest tests/test_detector.py::TestParallelProcessing` - Run specific test class
- `uv run pytest tests/ --cov=ocr_detection` - Run with coverage
- `uv run python test_basic.py` - Run basic functionality test

**CLI Usage**:
- `uv run ocr-detect document.pdf` - Basic analysis
- `uv run ocr-detect document.pdf --parallel --workers 4` - Parallel processing
- `uv run ocr-detect document.pdf --verbose --include-text` - Detailed output
- `uv run ocr-detect document.pdf --format json --output results.json` - Export results

## Architecture

The codebase is structured around three main API layers:

### Core Layer (`detector.py`)
- **PDFAnalyzer**: Core class that handles PDF document analysis
  - `analyze_page()`: Single page analysis
  - `analyze_all_pages()`: Sequential processing of all pages
  - `analyze_all_pages_parallel()`: Parallel processing for large PDFs (>10 pages)
  - `analyze_all_pages_auto()`: Automatic method selection
- **AnalysisResult**: Dataclass containing page analysis results
- **PageType**: Enum defining page types (TEXT, SCANNED, MIXED, EMPTY)

### Content Analysis Layer (`analyzer.py`)  
- **ContentAnalyzer**: Advanced text quality analysis
- **TextMetrics**: Detailed text quality metrics including OCR artifact detection
- Provides text density, formatting consistency, and OCR quality scoring

### API Layers
1. **Simple API** (`simple.py`): One-line usage with `detect_ocr()` function and `OCRDetection` class
2. **Enhanced API** (`api.py`): Rich objects with `OCRDetector`, `PageAnalysis`, and `PDFAnalysisResult`
3. **CLI Interface** (`cli.py`): Command-line tool with multiple output formats

## Parallel Processing

The library supports parallel processing for faster analysis of large PDFs:
- Automatically uses sequential processing for PDFs ≤10 pages
- Uses ThreadPoolExecutor with configurable worker threads
- Thread-safe by opening separate PDF instances per thread
- Maintains page order and handles errors gracefully
- Provides 3-8x performance improvement for large documents

## Key Design Patterns

**Context Manager Pattern**: PDFAnalyzer uses context managers to ensure proper resource cleanup of PDF documents.

**Multi-Library Approach**: Uses both PyMuPDF (fitz) and pdfplumber for robust text extraction, selecting the better result.

**Progressive Enhancement**: Three API layers provide increasing complexity - simple one-liners to detailed analysis objects.

**Confidence Scoring**: All results include confidence scores based on multiple detection methods and text quality metrics.

**Error Resilience**: Graceful fallbacks when advanced analysis fails, ensuring the library always produces results.

## Testing Strategy

Tests are organized by functionality:
- `test_detector.py`: Core PDFAnalyzer functionality and parallel processing
- `test_analyzer.py`: Content analysis and text metrics
- `test_simple_api.py`: Simple API interface
- `test_cli.py`: Command-line interface

The `TestParallelProcessing` class specifically tests thread safety, worker management, and result consistency between parallel and sequential processing.

## Dependencies

- **PyMuPDF (fitz)**: Primary PDF processing and image analysis
- **pdfplumber**: Alternative text extraction method  
- **click**: CLI framework
- **concurrent.futures**: Parallel processing (Python stdlib)
- Dont use Emoji in code. Its failing in Windows system.