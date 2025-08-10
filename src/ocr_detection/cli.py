"""Command-line interface for OCR detection."""

import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any

import click
from .detector import PDFAnalyzer, AnalysisResult
from .analyzer import ContentAnalyzer


@click.command()
@click.argument('pdf_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help='Output file path (format determined by extension: .json, .csv, or .txt)')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'text', 'summary']), 
              default='summary', help='Output format (default: summary)')
@click.option('--page', '-p', type=int, help='Analyze specific page only (0-indexed)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed analysis information')
@click.option('--include-text', is_flag=True, help='Include extracted text preview in output')
@click.option('--confidence-threshold', type=float, default=0.5,
              help='Minimum confidence threshold for recommendations (default: 0.5)')
def main(pdf_path: Path, output: Path, format: str, page: int, verbose: bool, 
         include_text: bool, confidence_threshold: float):
    """Analyze PDF pages to detect text vs scanned content.
    
    This tool analyzes PDF files to determine whether pages contain extractable
    text or are scanned images that would benefit from OCR processing.
    
    Examples:
    
        ocr-detect document.pdf
        
        ocr-detect document.pdf --format json --output results.json
        
        ocr-detect document.pdf --page 0 --verbose
        
        ocr-detect document.pdf --format csv --confidence-threshold 0.8
    """
    
    try:
        with PDFAnalyzer(pdf_path) as analyzer:
            # Analyze specific page or all pages
            if page is not None:
                results = [analyzer.analyze_page(page)]
                click.echo(f"Analyzing page {page} of {pdf_path.name}...")
            else:
                results = analyzer.analyze_all_pages()
                click.echo(f"Analyzing {len(results)} pages of {pdf_path.name}...")
            
            # Get summary
            summary = analyzer.get_summary(results)
            
            # Process results based on format
            if format == 'summary' and not output:
                _print_summary(results, summary, verbose, include_text, confidence_threshold)
            elif format == 'text' or (output and output.suffix == '.txt'):
                content = _format_text_output(results, summary, verbose, include_text)
                _write_or_print(content, output)
            elif format == 'json' or (output and output.suffix == '.json'):
                content = _format_json_output(results, summary, include_text)
                _write_or_print(content, output)
            elif format == 'csv' or (output and output.suffix == '.csv'):
                content = _format_csv_output(results, include_text)
                _write_or_print(content, output, mode='csv')
            else:
                # Default to summary
                _print_summary(results, summary, verbose, include_text, confidence_threshold)
                
    except Exception as e:
        click.echo(f"Error analyzing PDF: {e}", err=True)
        sys.exit(1)


def _print_summary(results: List[AnalysisResult], summary: Dict[str, Any], 
                  verbose: bool, include_text: bool, confidence_threshold: float):
    """Print a formatted summary to console."""
    
    click.echo("\n" + "="*60)
    click.echo("PDF CONTENT ANALYSIS SUMMARY")
    click.echo("="*60)
    
    # Overall statistics
    click.echo(f"\nTotal Pages: {summary['total_pages']}")
    click.echo(f"Average Confidence: {summary['average_confidence']:.2f}")
    click.echo(f"\nPage Type Distribution:")
    
    for page_type, count in summary['type_counts'].items():
        percentage = summary['type_percentages'][page_type]
        click.echo(f"  {page_type.title():8}: {count:3d} pages ({percentage:5.1f}%)")
    
    click.echo(f"\n{'='*60}")
    click.echo(f"FINAL RECOMMENDATION: {summary['recommended_action']}")
    click.echo(f"{'='*60}")
    
    # Show problematic pages if any
    problematic = summary.get('problematic_pages', {})
    if problematic.get('total_pages_needing_ocr', 0) > 0:
        click.echo(f"\n[PAGES NEEDING ATTENTION]")
        click.echo(f"Total pages that would benefit from OCR: {problematic['total_pages_needing_ocr']}")
        
        if problematic.get('scanned_pages'):
            pages_str = ', '.join(str(p) for p in problematic['scanned_pages'][:10])
            if len(problematic['scanned_pages']) > 10:
                pages_str += f" (and {len(problematic['scanned_pages'])-10} more)"
            click.echo(f"  * Scanned pages: {pages_str}")
        
        if problematic.get('mixed_pages_needing_ocr'):
            pages = [str(p['page']) for p in problematic['mixed_pages_needing_ocr'][:10]]
            pages_str = ', '.join(pages)
            if len(problematic['mixed_pages_needing_ocr']) > 10:
                pages_str += f" (and {len(problematic['mixed_pages_needing_ocr'])-10} more)"
            click.echo(f"  * Mixed content pages with poor text quality: {pages_str}")
        
        if problematic.get('low_quality_text_pages'):
            pages = [str(p['page']) for p in problematic['low_quality_text_pages'][:10]]
            pages_str = ', '.join(pages)
            if len(problematic['low_quality_text_pages']) > 10:
                pages_str += f" (and {len(problematic['low_quality_text_pages'])-10} more)"
            click.echo(f"  * Pages with very poor text extraction: {pages_str}")
    
    elif summary['recommended_action'] == "NO OCR NEEDED":
        click.echo(f"\n[SUCCESS] All pages have good text extraction quality!")
    
    # Show empty pages if any (for information)
    if problematic.get('empty_pages'):
        pages_str = ', '.join(str(p) for p in problematic['empty_pages'][:10])
        if len(problematic['empty_pages']) > 10:
            pages_str += f" (and {len(problematic['empty_pages'])-10} more)"
        click.echo(f"\n[INFO] Empty pages (no processing needed): {pages_str}")
    
    # Low confidence pages
    low_confidence_pages = [r for r in results if r.confidence < confidence_threshold]
    if low_confidence_pages:
        click.echo(f"\n[WARNING] Pages with low confidence (< {confidence_threshold}):")
        for result in low_confidence_pages:
            click.echo(f"  Page {result.page_number + 1}: {result.page_type.value} "
                      f"(confidence: {result.confidence:.2f})")
    
    # Detailed page information if verbose
    if verbose:
        click.echo("\n" + "-"*60)
        click.echo("DETAILED PAGE ANALYSIS")
        click.echo("-"*60)
        
        for result in results:
            click.echo(f"\nPage {result.page_number + 1}:")
            click.echo(f"  Type: {result.page_type.value.title()}")
            click.echo(f"  Confidence: {result.confidence:.2f}")
            click.echo(f"  Text Length: {result.text_length} characters")
            click.echo(f"  Text Ratio: {result.text_ratio:.3f}")
            click.echo(f"  Image Count: {result.image_count}")
            click.echo(f"  Image Ratio: {result.image_ratio:.3f}")
            
            # Processing suggestion
            text_metrics = ContentAnalyzer.analyze_text_quality(
                result.details.get('extracted_text_preview', '')
            )
            suggestion = ContentAnalyzer.suggest_processing_method(result, text_metrics)
            click.echo(f"  Suggested Method: {suggestion['method']}")
            click.echo(f"  Reason: {suggestion['reason']}")
            
            if include_text and result.text_length > 0:
                preview = result.details.get('extracted_text_preview', '')
                click.echo(f"  Text Preview: {preview[:100]}...")


def _format_text_output(results: List[AnalysisResult], summary: Dict[str, Any], 
                       verbose: bool, include_text: bool) -> str:
    """Format results as plain text."""
    output = []
    
    output.append("PDF Content Analysis Report")
    output.append("=" * 50)
    output.append(f"Total Pages: {summary['total_pages']}")
    output.append(f"Average Confidence: {summary['average_confidence']:.2f}")
    output.append("")
    
    output.append("Page Type Distribution:")
    for page_type, count in summary['type_counts'].items():
        percentage = summary['type_percentages'][page_type]
        output.append(f"  {page_type.title()}: {count} pages ({percentage:.1f}%)")
    
    output.append("")
    output.append("=" * 50)
    output.append(f"FINAL RECOMMENDATION: {summary['recommended_action']}")
    output.append("=" * 50)
    output.append("")
    
    if verbose:
        output.append("Page Details:")
        output.append("-" * 30)
        for result in results:
            output.append(f"Page {result.page_number + 1}: {result.page_type.value} "
                         f"(confidence: {result.confidence:.2f}, "
                         f"text: {result.text_length} chars, "
                         f"images: {result.image_count})")
            
            if include_text and result.text_length > 0:
                preview = result.details.get('extracted_text_preview', '')
                output.append(f"  Text: {preview[:100]}...")
            output.append("")
    
    return "\n".join(output)


def _format_json_output(results: List[AnalysisResult], summary: Dict[str, Any], 
                       include_text: bool) -> str:
    """Format results as JSON."""
    
    # Convert results to dictionaries
    results_data = []
    for result in results:
        result_dict = {
            "page_number": result.page_number + 1,  # Convert to 1-based indexing
            "page_type": result.page_type.value,
            "confidence": result.confidence,
            "text_ratio": result.text_ratio,
            "image_ratio": result.image_ratio,
            "text_length": result.text_length,
            "image_count": result.image_count
        }
        
        if include_text:
            result_dict["text_preview"] = result.details.get('extracted_text_preview', '')
            result_dict["details"] = result.details
        
        results_data.append(result_dict)
    
    output_data = {
        "summary": summary,
        "pages": results_data,
        "problematic_pages": summary.get("problematic_pages", {})
    }
    
    return json.dumps(output_data, indent=2)


def _format_csv_output(results: List[AnalysisResult], include_text: bool) -> str:
    """Format results as CSV."""
    import io
    
    output = io.StringIO()
    
    fieldnames = [
        'page_number', 'page_type', 'confidence', 'text_ratio', 
        'image_ratio', 'text_length', 'image_count'
    ]
    
    if include_text:
        fieldnames.append('text_preview')
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for result in results:
        row = {
            'page_number': result.page_number + 1,  # Convert to 1-based indexing
            'page_type': result.page_type.value,
            'confidence': round(result.confidence, 3),
            'text_ratio': round(result.text_ratio, 3),
            'image_ratio': round(result.image_ratio, 3),
            'text_length': result.text_length,
            'image_count': result.image_count
        }
        
        if include_text:
            row['text_preview'] = result.details.get('extracted_text_preview', '')
        
        writer.writerow(row)
    
    return output.getvalue()


def _write_or_print(content: str, output_path: Path = None, mode: str = 'text'):
    """Write content to file or print to console."""
    if output_path:
        if mode == 'csv':
            output_path.write_text(content, encoding='utf-8')
        else:
            output_path.write_text(content, encoding='utf-8')
        click.echo(f"Results saved to {output_path}")
    else:
        click.echo(content)


if __name__ == '__main__':
    main()