"""
Simple Gradio interface for RAG vs GraphRAG comparison.
Dark theme, minimal design, National Crime Agency branding.
"""

import gradio as gr
import json
import time
import requests
from pathlib import Path
import chromadb

# Configuration
CHROMA_PATH = Path("D:/assortments/GraphRAG/data/chunks/chromadb")
GRAPH_FILE = "D:/assortments/GraphRAG/code/rag/graphrag/data/graphrag_with_bridges.json"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral"

# Custom CSS for dark theme
css = """
.gradio-container {
    background: #0a0a0a !important;
    color: #00ff00 !important;
}
.gradio-header {
    background: linear-gradient(180deg, #1a1a1a 0%, #0a0a0a 100%) !important;
    border-bottom: 2px solid #00ff00 !important;
    padding: 20px !important;
}
h1 {
    color: #00ff00 !important;
    font-family: 'Courier New', monospace !important;
    text-shadow: 0 0 10px rgba(0, 255, 0, 0.8) !important;
}
p {
    color: #00cc00 !important;
    font-family: 'Courier New', monospace !important;
}
.gradio-button {
    background: #1a1a1a !important;
    color: #00ff00 !important;
    border: 1px solid #00ff00 !important;
}
.gradio-button:hover {
    background: #00ff00 !important;
    color: #0a0a0a !important;
}
"""

def query_chromadb(query_text, n_results=3):
    """Query ChromaDB for relevant chunks."""
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

def query_ollama(prompt, model=MODEL_NAME, timeout=120):
    """Query Ollama LLM."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def rag_query(query):
    """Standard RAG query."""
    contexts = query_chromadb(query, n_results=3)
    
    context_text = "\n\n".join([
        f"[{c['year']} - {c['section']}]\n{c['text'][:500]}" 
        for c in contexts[:3]
    ])
    
    prompt = f"""You are a helpful assistant answering questions about UK crime and statistics based on National Crime Agency documents.

Context information:
{context_text}

Question: {query}

Answer based only on the context provided. If the answer is not in the context, say so."""
    
    answer = query_ollama(prompt)
    return answer, contexts

def graphrag_query(query):
    """GraphRAG query with entity graph."""
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    
    # Extract question entities
    import re
    question_lower = query.lower()
    entities = []
    
    years = re.findall(r'20[12][0-9]{2}', query)
    entities.extend(years)
    
    keywords = ['NCA', 'crime', 'drug', 'trafficking', 'fraud', 'money', 'human', 'organization']
    for kw in keywords:
        if kw.lower() in question_lower and kw not in entities:
            entities.append(kw)
    
    # Find entities in graph
    relevant_entities = []
    node_map = {n['id']: n for n in nodes}
    
    for ent in entities:
        if ent in node_map:
            relevant_entities.append(ent)
    
    # Get chunks from ChromaDB
    contexts = query_chromadb(query, n_results=2)
    
    # Build prompt
    entity_context = "Knowledge Graph Entities: " + ", ".join(relevant_entities[:10])
    
    context_text = entity_context + "\n\n" + "\n\n".join([
        f"[{c['year']} - {c['section']}]\n{c['text'][:500]}" 
        for c in contexts[:2]
    ])
    
    prompt = f"""You are a helpful assistant answering questions about UK crime and NCA data.

Context Information:
{context_text}

Question: {query}

Answer based on the context and entities provided. If the answer is not in the context, say so."""
    
    answer = query_ollama(prompt)
    
    # Format entities for display
    entity_list = []
    for ent in relevant_entities[:10]:
        if ent in node_map:
            node = node_map[ent]
            entity_list.append({
                'entity': ent,
                'type': node.get('group', 'Unknown')
            })
    
    return answer, contexts, entity_list

def process_query(query, mode):
    """Process query through selected mode."""
    start_time = time.time()
    
    if mode == "RAG":
        answer, contexts, _ = rag_query(query)
        entities = []
    else:
        answer, contexts, entities = graphrag_query(query)
    
    elapsed = time.time() - start_time
    
    # Format output
    output = f"⏱️ Response Time: {elapsed:.1f}s\n\n"
    output += f"📝 Answer:\n{answer}\n\n"
    output += f"📄 Retrieved {len(contexts)} chunks:\n"
    for i, c in enumerate(contexts, 1):
        output += f"  {i}. {c['source']} ({c['year']})\n"
    
    if entities:
        output += f"\n🕸️ Found {len(entities)} entities:\n"
        for e in entities[:5]:
            output += f"  • {e['entity']} ({e['type']})\n"
    
    return output

# Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("""
    # National Crime Agency
    ## Protecting the UK from serious and organised crime
    ---
    """)
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Enter your question",
            placeholder="e.g., What are the main crime trends 2016-2023?",
            lines=2
        )
    
    with gr.Row():
        mode_select = gr.Radio(
            choices=["RAG", "GraphRAG"],
            label="Select Mode",
            value="RAG",
            horizontal=True
        )
    
    submit_btn = gr.Button("Submit", variant="primary")
    clear_btn = gr.Button("Clear")
    
    output = gr.Textbox(
        label="Result",
        lines=20,
        interactive=False
    )
    
    def clear_all():
        return "", ""
    
    submit_btn.click(
        fn=process_query,
        inputs=[query_input, mode_select],
        outputs=output
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[query_input, output]
    )

if __name__ == "__main__":
    demo.launch(server_port=8501, share=False)
