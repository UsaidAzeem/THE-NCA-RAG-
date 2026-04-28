"""
Add SEMANTIC_SIMILAR edges using ChromaDB embeddings.
Connects entities across distant clusters via cosine similarity > 0.75.
"""

import json
import time
import random
from pathlib import Path

# Configuration
CHUNKS_FILE = "D:/assortments/GraphRAG/data/chunks/chunks.json"
GRAPH_FILE = "D:/assortments/GraphRAG/code/rag/graphrag/data/graphrag_high_quality.json"
OUTPUT_FILE = "D:/assortments/GraphRAG/code/rag/graphrag/data/graphrag_with_semantic.json"

SIMILARITY_THRESHOLD = 0.75
MAX_EDGES = 5000
SAMPLE_SIZE = 300  # Entities to check pairwise

def get_entity_chunk_mapping():
    """Build entity -> chunk_ids mapping from chunks file."""
    print("Building entity -> chunk mapping...")
    
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data.get('chunks', [])
    
    import re
    entity_chunks = {}  # normalized_name -> set of chunk_ids
    
    for chunk in chunks:
        chunk_id = f"chunk_{chunk.get('id', 0)}"
        text = chunk.get('text', '')
        
        # Extract entities (same logic as build_hq_graph.py)
        # Dates
        dates = re.findall(r'20[12][0-9]{2}', text)
        for date in set(dates):
            if date not in entity_chunks:
                entity_chunks[date] = set()
            entity_chunks[date].add(chunk_id)
        
        # Organizations (ALL CAPS)
        orgs = re.findall(r'\b([A-Z]{2,}(?:\s+[A-Z]+)*)\b', text)
        for org in set(orgs):
            if len(org) > 2:
                if org not in entity_chunks:
                    entity_chunks[org] = set()
                entity_chunks[org].add(chunk_id)
    
    print(f"Found {len(entity_chunks)} unique entities")
    return entity_chunks

def get_chromadb_embeddings(chunk_ids):
    """Get embeddings from ChromaDB for given chunk IDs."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="D:/assortments/GraphRAG/data/chunks/chromadb")
        collection = client.get_collection("crime_docs_v2")
        
        # Limit to 1000 chunks
        chunk_list = list(chunk_ids)[:1000]
        
        results = collection.get(
            ids=chunk_list,
            include=['embeddings']
        )
        
        embeddings = {}
        for i, chunk_id in enumerate(results['ids']):
            if i < len(results['embeddings']):
                embeddings[chunk_id] = results['embeddings'][i]
        
        print(f"Got {len(embeddings)} chunk embeddings")
        return embeddings
    
    except Exception as e:
        print(f"Error accessing ChromaDB: {e}")
        return {}

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity."""
    dot = sum(a * b for a, b in zip(emb1, emb2))
    mag1 = sum(a * a for a in emb1) ** 0.5
    mag2 = sum(b * b for b in emb2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

def add_semantic_edges():
    """Add SEMANTIC_SIMILAR edges."""
    print("="*70)
    print("ADDING SEMANTIC SIMILAR EDGES")
    print("="*70)
    
    # Load graph (D3.js format)
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    print(f"Current graph: {len(nodes)} nodes, {len(edges)} edges")
    
    # Get entity -> chunk mapping
    entity_chunks = get_entity_chunk_mapping()
    
    # Get embeddings for chunks
    all_chunk_ids = set()
    for chunk_set in entity_chunks.values():
        all_chunk_ids.update(chunk_set)
    
    chunk_embeddings = get_chromadb_embeddings(all_chunk_ids)
    
    if not chunk_embeddings:
        print("No embeddings found! Skipping semantic edges.")
        return
    
    # Compute average embedding for each entity
    entity_embeddings = {}
    for entity_name, chunk_ids in entity_chunks.items():
        embs = [chunk_embeddings[cid] for cid in chunk_ids if cid in chunk_embeddings]
        if embs:
            avg_emb = [sum(x) / len(embs) for x in zip(*embs)]
            entity_embeddings[entity_name] = avg_emb
    
    print(f"Computed embeddings for {len(entity_embeddings)} entities")
    
    # Sample entities for pairwise comparison
    entity_names = list(entity_embeddings.keys())
    if len(entity_names) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} entities for speed...")
        random.seed(42)
        entity_names = random.sample(entity_names, SAMPLE_SIZE)
    
    # Find semantic similarities
    print(f"\nComputing pairwise similarities (threshold > {SIMILARITY_THRESHOLD})...")
    
    semantic_edges = []
    seen_pairs = set()
    
    start_time = time.time()
    
    for i in range(len(entity_names)):
        if len(semantic_edges) >= MAX_EDGES:
            print(f"Reached max edges limit ({MAX_EDGES})")
            break
        
        e1 = entity_names[i]
        emb1 = entity_embeddings[e1]
        
        for j in range(i+1, len(entity_names)):
            e2 = entity_names[j]
            emb2 = entity_embeddings[e2]
            
            pair_key = tuple(sorted([e1, e2]))
            if pair_key in seen_pairs:
                continue
            
            sim = cosine_similarity(emb1, emb2)
            
            if sim > SIMILARITY_THRESHOLD:
                semantic_edges.append({
                    "source": e1,
                    "target": e2,
                    "label": "SEMANTIC_SIMILAR",
                    "similarity": round(sim, 3)
                })
                seen_pairs.add(pair_key)
            
            if len(semantic_edges) % 100 == 0 and len(semantic_edges) > 0:
                elapsed = time.time() - start_time
                print(f"  Found {len(semantic_edges)} semantic edges... ({elapsed:.1f}s)")
    
    print(f"\nFound {len(semantic_edges)} semantic edges")
    
    # Add to graph
    for edge in semantic_edges:
        edges.append(edge)
    
    # Update graph data
    graph_data['edges'] = edges
    graph_data['total_relationships'] = len(edges)
    graph_data['semantic_edges_added'] = len(semantic_edges)
    
    # Save
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SEMANTIC EDGES ADDED!")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)} (+{len(semantic_edges)} semantic)")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"{'='*70}")

if __name__ == "__main__":
    add_semantic_edges()
