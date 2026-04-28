"""
Streamlit UI for RAG vs GraphRAG Comparison
Updated to use correct data sources and GraphRAG with bridges.
"""

import streamlit as st
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

st.set_page_config(
    page_title="GraphRAG: National Crime Agency ",
    layout="wide"
)

# ASCII Art Header with Dark Theme
st.markdown("""
<style>
    .ascii-header {
        background: #0a0a0a;
        color: #00ff00;
        font-family: 'Courier New', Courier, monospace;
        padding: 20px;
        border: 2px solid #00ff00;
        border-radius: 5px;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
    }
    .ascii-header h1 {
        color: #00ff00;
        text-shadow: 0 0 10px rgba(0, 255, 0, 0.8);
        letter-spacing: 3px;
    }
    .ascii-header h2 {
        color: #00ff00;
        margin-top: 10px;
    }
    .ascii-header p {
        color: #00cc00;
        letter-spacing: 2px;
    }
    .stApp {
        background: #0a0a0a;
        color: #00ff00;
    }
</style>


""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Settings")
mode = st.sidebar.radio(
    "Select Mode:",
    ["RAG", "GraphRAG"],
    horizontal=True
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Note**: Using Mistral 7B via Ollama")

# Stats in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Stats**")
try:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection("crime_docs_v2")
    st.sidebar.write(f"Docs: {collection.count()}")
except:
    st.sidebar.write("Docs: N/A")

# Graph stats
if Path(GRAPH_FILE).exists():
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    st.sidebar.write(f"Graph Nodes: {len(graph.get('nodes', []))}")
    st.sidebar.write(f"Graph Edges: {len(graph.get('edges', []))}")
else:
    st.sidebar.write("Graph: Not loaded")

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

def build_prompt(query, contexts):
    """Build prompt with context."""
    context_text = "\n\n".join([
        f"[{c['year']} - {c['section']}]\n{c['text'][:500]}" 
        for c in contexts[:3]
    ])
    
    prompt = f"""You are a helpful assistant answering questions about UK crime and statistics based on National Crime Agency documents.

Context information:
{context_text}

Question: {query}

Answer based only on the context provided. If the answer is not in the context, say so."""
    return prompt

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
    except requests.exceptions.ConnectionError:
        return "Error: Ollama not running. Start Ollama and run 'ollama run mistral' first."
    except Exception as e:
        return f"Error: {str(e)}"

def rag_query(query):
    """Standard RAG query."""
    contexts = query_chromadb(query, n_results=3)
    prompt = build_prompt(query, contexts)
    answer = query_ollama(prompt)
    return answer, contexts, []

def graphrag_query(query):
    """GraphRAG query with entity graph."""
    # Load graph
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    
    # Extract question entities (simple keyword matching)
    import re
    question_lower = query.lower()
    entities = []
    
    # Extract years
    years = re.findall(r'20[12][0-9]{2}', query)
    entities.extend(years)
    
    # Extract keywords
    keywords = ['NCA', 'crime', 'drug', 'trafficking', 'fraud', 'money', 'human', 'organization']
    for kw in keywords:
        if kw.lower() in question_lower and kw not in entities:
            entities.append(kw)
    
    # Find these entities in graph
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
    contexts = query_chromadb(query, n_results=2)
    
    # Build enhanced prompt with graph context
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

# Main query interface
query = st.text_input("Enter your question:", 
                      placeholder="   ")

col1, col2 = st.columns([1, 1])

with col1:
    submit = st.button("Submit", type="primary", use_container_width=True)

with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.session_state.clear()
    st.rerun()

if submit:
    if query:
        start_time = time.time()
        
        with st.spinner("Processing..."):
            if mode == "RAG":
                answer, contexts, entities = rag_query(query)
            else:
                answer, contexts, entities = graphrag_query(query)
        
        elapsed = time.time() - start_time
        
        st.success(f"Completed in {elapsed:.1f}s")
        
        # Display answer
        st.subheader("Answer")
        st.write(answer)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Response Time", f"{elapsed:.1f}s")
        with col2:
            st.metric("Chunks Retrieved", len(contexts))
        with col3:
            st.metric("Entities Found", len(entities) if entities else 0)
        
        # Display contexts
        st.subheader("Retrieved Contexts")
        for i, c in enumerate(contexts, 1):
            with st.expander(f"Chunk {i}: {c.get('source', 'unknown')} ({c.get('year', 'N/A')})"):
                st.markdown(f"**Section**: {c.get('section', 'N/A')}")
                st.write(c.get('text', '')[:800])
        
        # Display entities (GraphRAG only)
        if mode == "GraphRAG" and entities:
            st.subheader("Related Entities")
            cols = st.columns(min(5, len(entities)))
            for idx, e in enumerate(entities[:10]):
                with cols[idx % 5]:
                    st.markdown(f"**{e['entity']}**")
                    
        
        
# Simple Footer
st.markdown("---")
st.markdown("""
<div style="background: #1a1a1a; padding: 15px; border-left: 4px solid #00ff00;">
<h3 style="color: #00ff00; font-family: 'Courier New', monospace;">System Status</h3>
<p style="color: #00cc00; font-family: 'Courier New', monospace; font-size: 0.9em;">
• GraphRAG: 10,417 nodes, 32,806 edges<br/>
• RAG: 1,529 chunks<br/>
• LLM: Mistral 7B via Ollama
</p>
</div>
""", unsafe_allow_html=True)
