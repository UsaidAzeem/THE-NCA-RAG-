"""
GraphRAG query function v2 - Enhanced with chunk text context.
Traverses the knowledge graph and retrieves actual text from chunks.
"""

import json
import time
import requests

# Configuration
GRAPH_FILE = "D:/assortments/GraphRAG/code/rag/graphrag/data/graphrag_with_bridges.json"
CHUNKS_FILE = "D:/assortments/GraphRAG/data/chunks/chunks.json"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral"

def load_graph():
    """Load the GraphRAG graph."""
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_chunks():
    """Load chunks data."""
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Handle both formats: {"chunks": [...]} or [...]
        if isinstance(data, dict) and 'chunks' in data:
            return data['chunks']
        return data

def extract_question_entities(question):
    """Extract entities from question (rule-based)."""
    import re
    
    entities = []
    question_lower = question.lower()
    
    # Extract years
    years = re.findall(r'20[12][0-9]{2}', question)
    entities.extend(years)
    
    # Extract keywords
    keywords = ['NCA', 'crime', 'drug', 'trafficking', 'fraud', 'money laundering', 'human trafficking']
    for kw in keywords:
        if kw.lower() in question_lower:
            entities.append(kw)
    
    return list(set(entities))

def graphrag_query(question, graph_data, chunks, top_k=5):
    """Query the GraphRAG graph with chunk text context."""
    print(f"GraphRAG Query: {question}")
    
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    # Extract question entities
    question_entities = extract_question_entities(question)
    print(f"Question entities: {question_entities}")
    
    # Find these entities in graph
    relevant_nodes = []
    node_map = {n['id']: n for n in nodes}
    
    for qe in question_entities:
        if qe in node_map:
            relevant_nodes.append(qe)
    
    if not relevant_nodes:
        print("No relevant entities found in graph")
        return None, []
    
    # Traverse graph (BFS, depth=2)
    visited = set(relevant_nodes)
    frontier = list(relevant_nodes)
    
    for depth in range(2):
        new_frontier = []
        for node in frontier:
            # Get neighbors
            for edge in edges:
                neighbor = None
                if edge.get('source') == node:
                    neighbor = edge.get('target')
                elif edge.get('target') == node:
                    neighbor = edge.get('source')
                
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    new_frontier.append(neighbor)
        
        frontier = new_frontier
        if depth == 0:
            # Add top-k neighbors
            visited_list = list(visited)
            relevant_nodes.extend(visited_list[:top_k])
    
    # Build context from visited nodes - get unique entity types
    context_entities = []
    for node_id in visited:
        if node_id in node_map:
            node = node_map[node_id]
            context_entities.append({
                'name': node['id'],
                'type': node.get('group', 'Unknown')
            })
    
    # Get relevant chunks based on entities
    # For now, use ChromaDB to get relevant chunks
    import chromadb
    client = chromadb.PersistentClient(path="D:/assortments/GraphRAG/data/chunks/chromadb")
    collection = client.get_collection("crime_docs_v2")
    
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    
    relevant_chunks = []
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        relevant_chunks.append({
            'text': doc[:500],  # Limit chunk size
            'source': meta.get('document_name', 'unknown'),
            'year': meta.get('year', 'unknown')
        })
    
    print(f"Found {len(context_entities)} entities, {len(relevant_chunks)} chunks")
    
    # Build prompt with both graph context and text chunks
    context_text = "Knowledge Graph Entities:\n"
    for e in context_entities[:50]:  # Limit context
        context_text += f"  - {e['name']} (Type: {e['type']})\n"
    
    context_text += "\n\nRelevant Document Chunks:\n"
    for i, chunk in enumerate(relevant_chunks):
        context_text += f"\n[Chunk {i+1}] {chunk['source']} ({chunk['year']}):\n"
        context_text += chunk['text'] + "\n"
    
    prompt = f"""You are a helpful assistant answering questions about UK crime and NCA data.

Context Information:
{context_text}

Question: {question}

Answer based ONLY on the context provided. If the answer cannot be derived from the context, say so. Be specific and cite sources when possible."""

    # Query LLM
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        answer = response.json()['message']['content']
        return answer, context_entities, relevant_chunks
    
    except Exception as e:
        return f"Error querying LLM: {e}", context_entities, relevant_chunks

def run_test_questions():
    """Run test questions comparing RAG vs GraphRAG."""
    questions = [
        "What are the main crime trends 2016-2023?",
        "How does NCA collaborate with other organizations?",
        "What is the statistical trend in human trafficking cases?",
        "Which organizations are most active in drug trafficking?",
        "Compare crime patterns in 2016 vs 2023"
    ]
    
    print("="*70)
    print("GRAPHRAG vs RAG COMPARISON")
    print("="*70)
    
    graph_data = load_graph()
    chunks = load_chunks()
    
    print(f"Graph loaded: {len(graph_data.get('nodes', []))} nodes, "
          f"{len(graph_data.get('edges', []))} edges")
    print(f"Chunks loaded: {len(chunks)} chunks")
    
    results = []
    
    for i, question in enumerate(questions):
        print(f"\n{'='*70}")
        print(f"Question {i+1}/{len(questions)}: {question}")
        print(f"{'='*70}")
        
        # GraphRAG query
        start = time.time()
        answer, entities, chunks_used = graphrag_query(question, graph_data, chunks, top_k=3)
        elapsed = time.time() - start
        
        results.append({
            'question': question,
            'answer': answer,
            'entities_count': len(entities),
            'chunks_count': len(chunks_used),
            'time': elapsed,
            'method': 'GraphRAG'
        })
        
        print(f"\nGraphRAG Answer (first 300 chars):")
        print(answer[:300] if answer else "No answer")
        print(f"Time: {elapsed:.1f}s")
        print(f"Entities: {len(entities)}, Chunks: {len(chunks_used)}")
    
    # Save results
    output = {
        'results': results,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'method': 'GraphRAG with chunk context'
    }
    
    with open('D:/assortments/GraphRAG/code/rag/graphrag/data/graphrag_v2_results.json', 
             'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("TESTING COMPLETE!")
    print(f"Results saved to: graphrag_v2_results.json")
    print(f"{'='*70}")

if __name__ == "__main__":
    run_test_questions()
