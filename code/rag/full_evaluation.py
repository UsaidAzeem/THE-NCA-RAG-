"""
Full Evaluation: RAG vs GraphRAG
Runs 10 questions through both pipelines and calculates metrics.
"""

import json
import time
import requests
import chromadb

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"
CHROMA_PATH = "D:/assortments/GraphRAG/data/chunks/chromadb"
GRAPH_FILE = "D:/assortments/GraphRAG/code/rag/graphrag/data/graphrag_with_bridges.json"

def query_ollama(prompt, max_tokens=200):
    """Query Ollama."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1}
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        return f"Error: {str(e)}"

def rag_query(question):
    """RAG query."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("crime_docs_v2")
    
    results = collection.query(query_texts=[question], n_results=3)
    
    context = ""
    sources = []
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        context += f"\n[{meta.get('year', '?')}] {doc[:400]}\n"
        sources.append(meta.get('document_name', 'unknown'))
    
    prompt = f"""Context: {context}

Question: {question}

Answer briefly and specifically based only on the context:"""
    
    answer = query_ollama(prompt, max_tokens=200)
    return answer, sources

def graphrag_query(question):
    """GraphRAG query."""
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    
    # Extract question entities
    import re
    years = re.findall(r'20[12][0-9]{2}', question)
    keywords = ['NCA', 'crime', 'drug', 'trafficking', 'fraud', 'money laundering', 'human']
    question_lower = question.lower()
    
    entities = years[:]
    for kw in keywords:
        if kw.lower() in question_lower and kw not in entities:
            entities.append(kw)
    
    # Get relevant entities from graph
    relevant_entities = []
    node_map = {n['id']: n for n in nodes}
    
    for ent in entities:
        if ent in node_map:
            relevant_entities.append(ent)
    
    # Get neighbors (BFS depth=1)
    if relevant_entities:
        for edge in edges[:2000]:
            for re_ent in relevant_entities[:5]:
                if edge.get('source') == re_ent and len(relevant_entities) < 15:
                    relevant_entities.append(edge.get('target'))
                elif edge.get('target') == re_ent and len(relevant_entities) < 15:
                    relevant_entities.append(edge.get('source'))
    
    # Get chunks from ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("crime_docs_v2")
    results = collection.query(query_texts=[question], n_results=2)
    
    chunk_text = ""
    sources = []
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        chunk_text += f"\n[{meta.get('year', '?')}] {doc[:300]}\n"
        sources.append(meta.get('document_name', 'unknown'))
    
    prompt = f"""Knowledge Graph Entities: {', '.join(relevant_entities[:10])}

Document Context: {chunk_text}

Question: {question}

Answer briefly based on the context and entities:"""
    
    answer = query_ollama(prompt, max_tokens=200)
    return answer, sources

def calculate_metrics(question, rag_answer, graphrag_answer):
    """Calculate evaluation metrics using LLM-as-judge."""
    prompt = f"""You are evaluating two answers to the same question about crime data.

Question: {question}

Answer A (RAG): {rag_answer[:300]}

Answer B (GraphRAG): {graphrag_answer[:300]}

Rate each answer on a scale of 1-5 for:
1. Relevance (does it answer the question?)
2. Completeness (does it cover key aspects?)
3. Grounding (is it supported by context?)

Format your response as:
RAG: Relevance=X, Completeness=Y, Grounding=Z
GraphRAG: Relevance=X, Completeness=Y, Grounding=Z
Winner: [RAG/GraphRAG/Both]

Be objective and brief."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 150, "temperature": 0.1}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()['response']
    except:
        return "Metrics calculation failed"

def run_evaluation():
    """Run full evaluation."""
    questions = [
        "What are the main crime trends 2016-2023?",
        "How does NCA collaborate with other organizations?",
        "What is the statistical trend in human trafficking cases?",
        "Which organizations are most active in drug trafficking?",
        "Compare crime patterns in 2016 vs 2023",
        "What are the key entities in NCA's operations?",
        "How do crime types cluster together?",
        "What are the top crime priorities for NCA?",
        "Which entities bridge different crime domains?",
        "Summarize NCA's strategic focus areas"
    ]
    
    print("="*70)
    print("FULL EVALUATION: RAG vs GraphRAG (10 Questions)")
    print("="*70)
    
    results = []
    
    for i, question in enumerate(questions):
        print(f"\n{'='*70}")
        print(f"Question {i+1}/10: {question}")
        print(f"{'='*70}")
        
        # RAG Query
        print("RAG: ", end="", flush=True)
        start = time.time()
        rag_answer, rag_sources = rag_query(question)
        rag_time = time.time() - start
        print(f"Done ({rag_time:.1f}s)")
        
        # GraphRAG Query
        print("GraphRAG: ", end="", flush=True)
        start = time.time()
        graphrag_answer, graphrag_sources = graphrag_query(question)
        graphrag_time = time.time() - start
        print(f"Done ({graphrag_time:.1f}s)")
        
        # Calculate metrics
        print("Metrics: ", end="", flush=True)
        metrics = calculate_metrics(question, rag_answer, graphrag_answer)
        print("Done")
        
        results.append({
            'question': question,
            'rag_answer': rag_answer,
            'rag_time': rag_time,
            'rag_sources': rag_sources,
            'graphrag_answer': graphrag_answer,
            'graphrag_time': graphrag_time,
            'graphrag_sources': graphrag_sources,
            'metrics': metrics
        })
        
        print(f"\nRAG Answer (first 150 chars): {rag_answer[:150]}...")
        print(f"GraphRAG Answer (first 150 chars): {graphrag_answer[:150]}...")
        print(f"\nMetrics:\n{metrics}")
    
    # Save results
    output = {
        'results': results,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'summary': {
            'total_questions': len(questions),
            'avg_rag_time': sum(r['rag_time'] for r in results) / len(results) if results else 0,
            'avg_graphrag_time': sum(r['graphrag_time'] for r in results) / len(results) if results else 0
        }
    }
    
    with open('D:/assortments/GraphRAG/code/rag/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: evaluation_results.json")
    print(f"Average RAG time: {output['summary']['avg_rag_time']:.1f}s")
    print(f"Average GraphRAG time: {output['summary']['avg_graphrag_time']:.1f}s")
    print("="*70)

if __name__ == "__main__":
    run_evaluation()
