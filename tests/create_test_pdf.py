#!/usr/bin/env python3
"""Create a simple test PDF for testing purposes."""

import tempfile
import requests
from pathlib import Path


def create_simple_test_pdf(filename="test.pdf"):
    """Create or download a simple test PDF."""
    # Try to download a simple PDF first
    pdf_url = "https://files.edgestore.dev/kv3hoirymwcmuuoj/publicFiles/_public/e168fb66-4c8b-4ddb-a807-414cb0ca72fd.pdf"
    
    try:
        print(f"Downloading test PDF from {pdf_url}...")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        pdf_path = Path(filename)
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Created test PDF: {pdf_path} ({len(response.content)} bytes)")
        return pdf_path
        
    except Exception as e:
        print(f"Failed to download PDF: {e}")
        
        # Fallback: create a minimal PDF using reportlab if available
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            pdf_path = Path(filename)
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            
            # Add some content to multiple pages
            for page_num in range(5):
                c.drawString(100, 750, f"Test PDF - Page {page_num + 1}")
                c.drawString(100, 700, "This is a test PDF file for OCR detection testing.")
                c.drawString(100, 650, f"Page {page_num + 1} contains extractable text content.")
                c.drawString(100, 600, "Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
                c.showPage()
            
            c.save()
            print(f"Created simple test PDF using reportlab: {pdf_path}")
            return pdf_path
            
        except ImportError:
            print("reportlab not available, creating minimal file")
            # Create a minimal file that exists but isn't a real PDF
            pdf_path = Path(filename)
            pdf_path.write_bytes(b"%PDF-1.4\n%Fake PDF for testing\n")
            print(f"Created minimal test file: {pdf_path}")
            return pdf_path


if __name__ == "__main__":
    create_simple_test_pdf()