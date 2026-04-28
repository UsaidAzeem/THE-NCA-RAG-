"""
Process 155 NCA PDFs into chunks for RAG and GraphRAG pipelines.
Extracts text, chunks it, and saves to data/chunks/chunks.json
"""

import os
import json
import re
from pathlib import Path
from pypdf import PdfReader

# Configuration
PDF_DIR = Path("D:/assortments/GraphRAG/data/PDFs")
CHUNKS_OUTPUT = Path("D:/assortments/GraphRAG/data/chunks/chunks.json")
CHUNK_SIZE = 500  # tokens/words per chunk
CHUNK_OVERLAP = 50  # overlap between chunks

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    try:
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path.name}: {e}")
        return ""

def extract_year_from_filename(filename):
    """Extract year from PDF filename if present."""
    # Look for 4-digit year in filename
    match = re.search(r'(20\d{2})', filename)
    if match:
        return match.group(1)
    return "unknown"

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks

def process_all_pdfs():
    """Process all PDFs and create chunks."""
    print("Starting PDF processing...")
    print(f"PDF directory: {PDF_DIR}")
    
    if not PDF_DIR.exists():
        print(f"Error: PDF directory not found: {PDF_DIR}")
        return
    
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    all_chunks = []
    chunk_id = 0
    
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"  No text extracted from {pdf_path.name}")
            continue
        
        # Extract metadata
        year = extract_year_from_filename(pdf_path.name)
        document_name = pdf_path.stem  # filename without extension
        
        # Chunk the text
        chunks_from_pdf = chunk_text(text)
        print(f"  Extracted {len(chunks_from_pdf)} chunks from {len(text.split())} words")
        
        # Create chunk objects
        for i, chunk_content in enumerate(chunks_from_pdf):
            chunk = {
                "id": chunk_id,
                "text": chunk_content,
                "metadata": {
                    "document_name": document_name,
                    "year": year,
                    "section": f"chunk_{i}",
                    "source_file": pdf_path.name
                }
            }
            all_chunks.append(chunk)
            chunk_id += 1
    
    # Save chunks
    CHUNKS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "chunks": all_chunks,
        "total_chunks": len(all_chunks),
        "total_pdfs": len(pdf_files),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }
    
    with open(CHUNKS_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Chunks saved to: {CHUNKS_OUTPUT}")
    print(f"{'='*50}")
    
    return output_data

if __name__ == "__main__":
    process_all_pdfs()
