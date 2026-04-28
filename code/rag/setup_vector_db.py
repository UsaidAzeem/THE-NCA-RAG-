"""
Setup vector database for RAG pipeline.
Creates ChromaDB collection and loads document chunks.
"""

import json
import chromadb
from pathlib import Path

def setup_vector_db(chunks_file="data/chunks/chunks.json", collection_name="crime_docs_v2"):
    """
    Setup ChromaDB vector database with document chunks.
    
    Args:
        chunks_file: Path to chunks JSON
        collection_name: Name for ChromaDB collection
    """
    print("Setting up vector database...")
    
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    chunks = chunks_data.get('chunks', chunks_data)
    print(f"Loaded {len(chunks)} chunks")
    
    # Create ChromaDB client
    chroma_path = Path("data/chunks/chromadb")
    chroma_path.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    # Get or create collection
    try:
        collection = client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists")
        client.delete_collection(collection_name)
        print(f"Deleted existing collection")
    except:
        pass
    
    # Create new collection with embedding function
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "UK Crime Agency documents for RAG"}
    )
    
    # Add chunks to collection
    documents = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        
        documents.append(text)
        metadatas.append(metadata)
        ids.append(f"chunk_{i}")
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"\nVector DB setup complete!")
    print(f"Collection: {collection_name}")
    print(f"Documents: {len(documents)}")
    print(f"Stored at: {chroma_path}")

if __name__ == "__main__":
    setup_vector_db()