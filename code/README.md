# Crime Data RAG/GraphRAG Project

A comparative analysis of RAG and GraphRAG approaches for answering questions about UK crime and statistics.

## Setup

1. Install dependencies:
\`\`\`
pip install -r requirements.txt
\`\`\`

2. Download and setup Ollama:
- Download from https://ollama.com
- Run: \`ollama pull mistral\`

3. Setup ChromaDB:
\`\`\`
python code/rag/setup_vector_db.py
\`\`\`

## Running RAG Pipeline

\`\`\`
python code/rag/rag_pipeline.py "Your question here"
\`\`\`

## Running GraphRAG Pipeline

1. Extract entities:
\`\`\`
python code/rag/graphrag/extract_entities.py
\`\`\`

2. Build knowledge graph:
\`\`\`
python code/rag/graphrag/build_graph.py
\`\`\`

3. Query:
\`\`\`
python code/rag/graphrag/graphrag_pipeline.py "Your question here"
\`\`\`

## Files

- rag_pipeline.py: Simple RAG implementation
- graphrag/extract_entities.py: Entity extraction using LLM
- graphrag/build_graph.py: Knowledge graph builder
- graphrag/graphrag_pipeline.py: GraphRAG query pipeline

## Requirements

- Python 3.x
- Ollama with mistral model
- ChromaDB
- NetworkX
- pdfplumber