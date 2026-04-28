import json
import requests
from pathlib import Path

CHROMA_PATH = Path("D:/assortments/GraphRAG/data/chunks/chromadb")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral"

def query_chromadb(query_text, n_results=3):
    """
    Query the ChromaDB vector database for relevant document chunks.
    
    Args:
        query_text: The user's question/query
        n_results: Number of top results to retrieve (default: 3)
    
    Returns:
        List of context dictionaries with text, source, year, section
    """
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection("crime_docs_v2")
    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    contexts = []
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        contexts.append({
            'text': doc,
            'source': meta.get('document_name', 'unknown'),
            'year': meta.get('year', 'unknown'),
            'section': meta.get('section', 'unknown')
        })
    return contexts

def build_prompt(query, contexts):
    """
    Build the prompt for the LLM with retrieved context.
    
    Args:
        query: User's question
        contexts: List of context dictionaries from ChromaDB
    
    Returns:
        Formatted prompt string
    """
    context_text = "\n\n".join([
        f"[{c['year']} - {c['section']}]\n{c['text']}" 
        for c in contexts
    ])
    
    prompt = f"""You are a helpful assistant answering questions about UK crime and statistics based on National Crime Agency documents.

Context information:
{context_text}

Question: {query}

Answer based only on the context provided. If the answer is not in the context, say so."""
    return prompt

def query_ollama(prompt, model="mistral"):
    """
    Query the Ollama local LLM for text generation.
    
    Args:
        prompt: Formatted prompt with context and question
        model: Model name to use (default: mistral)
    
    Returns:
        Generated text response
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()['message']['content']
    except requests.exceptions.ConnectionError:
        return "Error: Ollama not running. Start Ollama and run 'ollama run mistral' first."
    except Exception as e:
        return f"Error: {str(e)}"

def rag_query(query, model="mistral"):
    """
    Main RAG query function that combines retrieval and generation.
    
    Args:
        query: User's question
        model: LLM model name
    
    Returns:
        Tuple of (answer, contexts)
    """
    print(f"Querying: {query}")
    
    print("Retrieving relevant context from ChromaDB...")
    contexts = query_chromadb(query)
    
    print(f"Found {len(contexts)} relevant chunks")
    for c in contexts:
        print(f"  - {c['source']} ({c['year']})")
    
    prompt = build_prompt(query, contexts)
    
    print("Sending to Ollama...")
    answer = query_ollama(prompt, model)
    
    return answer, contexts

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your question: ")
    if query:
        answer, contexts = rag_query(query)
        print("\n" + "="*50)
        print("ANSWER:")
        print(answer)