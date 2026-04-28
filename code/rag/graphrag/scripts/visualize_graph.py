"""
Build NetworkX graph from extracted entities and create PyVis visualization.
"""

import json
from pathlib import Path
import networkx as nx

KNOWLEDGE_FILE = "D:/assortments/GraphRAG/code/rag/graphrag/graphrag_500.json"
OUTPUT_DIR = Path("D:/assortments/GraphRAG/code/rag/graphrag")

def build_networkx_graph(knowledge_file):
    """Build NetworkX graph from knowledge base."""
    print(f"Building NetworkX graph from {knowledge_file}...")
    
    with open(knowledge_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    entities = data.get('entities', [])
    relationships = data.get('relationships', [])
    
    print(f"Loaded {len(entities)} entities and {len(relationships)} relationships")
    
    # Create graph
    G = nx.Graph()
    
    # Add entities as nodes
    for entity in entities:
        node_id = entity['name']
        G.add_node(
            node_id,
            type=entity.get('type', 'Unknown'),
            description=entity.get('description', ''),
            document=entity.get('document', ''),
            year=entity.get('year', '')
        )
    
    # Add relationships as edges
    print(f"Processing {len(relationships)} relationships...")
    edges_added = 0
    for rel in relationships:
        source = rel.get('source', '')
        target = rel.get('target', '')
        if source and target and source in G.nodes and target in G.nodes:
            G.add_edge(
                source,
                target,
                type=rel.get('type', 'related'),
                description=rel.get('description', '')
            )
            edges_added += 1
        else:
            if source and target:
                print(f"  Skipped: '{source}' -> '{target}' (nodes not in graph)")
    
    print(f"Added {edges_added} edges to the graph")
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def save_graph_for_visualization(G, output_dir):
    """Save graph in formats suitable for PyVis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as GML for NetworkX
    gml_file = output_dir / "graph_demo.gml"
    nx.write_gml(G, gml_file)
    
    # Save nodes and edges as JSON for PyVis
    nodes_data = []
    for node_id, attrs in G.nodes(data=True):
        nodes_data.append({
            "id": node_id,
            "label": node_id,
            "group": attrs.get('type', 'Unknown'),
            "title": f"{node_id}\nType: {attrs.get('type', 'Unknown')}\n{attrs.get('description', '')}"
        })
    
    edges_data = []
    for source, target, attrs in G.edges(data=True):
        edges_data.append({
            "from": source,
            "to": target,
            "label": attrs.get('type', 'related')
        })
    
    graph_data = {
        "nodes": nodes_data,
        "edges": edges_data
    }
    
    json_file = output_dir / "graph_demo.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"Graph saved!")
    print(f"GML format: {gml_file}")
    print(f"JSON format (for PyVis): {json_file}")
    print(f"Nodes: {len(nodes_data)}")
    print(f"Edges: {len(edges_data)}")
    print(f"{'='*50}")
    
    return json_file

def create_pyvis_html():
    """Create PyVis HTML visualization."""
    try:
        from pyvis.network import Network
        
        print("\nCreating PyVis visualization...")
        
        # Load graph data
        json_file = OUTPUT_DIR / "graph_demo.json"
        with open(json_file, 'r') as f:
            graph_data = json.load(f)
        
        # Create PyVis network
        net = Network(height="750px", width="100%", directed=False)
        
        # Add nodes
        for node in graph_data['nodes']:
            net.add_node(
                node['id'],
                label=node['label'],
                group=node['group'],
                title=node['title']
            )
        
        # Add edges
        for edge in graph_data['edges']:
            net.add_edge(edge['from'], edge['to'], label=edge['label'])
        
        # Save
        html_file = OUTPUT_DIR / "graph_visualization.html"
        net.show(str(html_file))
        
        print(f"PyVis visualization saved: {html_file}")
        print(f"Open this file in a browser to view the graph!")
        
    except ImportError:
        print("PyVis not installed. Install with: pip install pyvis")
        print("Skipping visualization creation...")

if __name__ == "__main__":
    # Build graph
    G = build_networkx_graph(KNOWLEDGE_FILE)
    
    # Save graph
    save_graph_for_visualization(G, OUTPUT_DIR)
    
    # Create PyVis visualization
    create_pyvis_html()
