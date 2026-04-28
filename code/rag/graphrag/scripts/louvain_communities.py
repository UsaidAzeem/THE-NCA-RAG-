"""
Run Louvain community detection on the graph with semantic edges.
This groups nodes into communities based on edge density.
"""

import json
import time
from pathlib import Path

# Configuration
GRAPH_FILE = "D:/assortments/GraphRAG/code/rag/graphrag/data/graphrag_with_semantic.json"
OUTPUT_FILE = "D:/assortments/GraphRAG/code/rag/graphrag/data/communities.json"

def run_louvain():
    """Run Louvain community detection."""
    print("="*70)
    print("LOUVAIN COMMUNITY DETECTION")
    print("="*70)
    
    # Load graph (D3.js format)
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
    
    # Build NetworkX graph
    import networkx as nx
    G = nx.Graph()
    
    # Add nodes
    for node in nodes:
        G.add_node(
            node['id'],
            group=node.get('group', 'Unknown')
        )
    
    # Add edges
    for edge in edges:
        source = edge.get('source', '')
        target = edge.get('target', '')
        if source and target:
            G.add_edge(source, target, type=edge.get('label', 'unknown'))
    
    print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Run Louvain
    try:
        import community  # python-louvain
        
        print("\nRunning Louvain algorithm...")
        partition = community.best_partition(G)
        
        print(f"Found {len(set(partition.values()))} communities")
        
        # Build communities data
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = {
                    'id': comm_id,
                    'nodes': [],
                    'size': 0,
                    'types': set()
                }
            communities[comm_id]['nodes'].append(node)
            communities[comm_id]['size'] += 1
            node_type = G.nodes[node].get('group', 'Unknown')
            communities[comm_id]['types'].add(node_type)
        
        # Convert sets to lists for JSON serialization
        result = []
        for comm_id, data in communities.items():
            result.append({
                'community_id': comm_id,
                'nodes': data['nodes'],
                'size': data['size'],
                'types': list(data['types']),
                'sample_nodes': data['nodes'][:10]  # First 10 for preview
            })
        
        # Sort by size
        result.sort(key=lambda x: x['size'], reverse=True)
        
        output = {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'num_communities': len(result),
            'communities': result
        }
        
        # Save
        Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"COMMUNITIES FOUND!")
        print(f"Total communities: {len(result)}")
        for i, comm in enumerate(result[:5]):
            print(f"  {i+1}. Community {comm['community_id']}: {comm['size']} nodes, types: {comm['types'][:3]}")
        print(f"Saved to: {OUTPUT_FILE}")
        print(f"{'='*70}")
        
        return output
        
    except ImportError:
        print("python-louvain not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'python-louvain'])
        print("Please re-run the script.")
        return None

if __name__ == "__main__":
    run_louvain()
