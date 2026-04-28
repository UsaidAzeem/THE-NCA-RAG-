"""
Build a high-quality GraphRAG using:
1. Rule-based entity extraction (deterministic)
2. Co-occurrence relationships (proven to work)
3. Inference-based semantic relations (rule-based)
4. Cross-document linking
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# Input/Output
CHUNKS_FILE = "D:/assortments/GraphRAG/data/chunks/chunks.json"
OUTPUT_FILE = "D:/assortments/GraphRAG/code/rag/graphrag_high_quality.json"
GML_FILE = "D:/assortments/GraphRAG/code/rag/graphrag/graph_hq.gml"

# Entity types and patterns
ENTITY_PATTERNS = {
    "Date": r'\b(20[12][0-9]{2})\b',
    "Statistic": r'(\d+(?:,\d+)?(?:\.\d+)?\s*(?:%|million|billion|thousand))',
    "Organization": r'\b([A-Z]{2,}(?:\s+[A-Z]+)*)\b',
    "Person": r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
}

# Crime keywords for relationship inference
CRIME_KEYWORDS = [
    'drug', 'trafficking', 'launder', 'fraud', 'cyber', 'crime',
    'bribery', 'corruption', 'kidnap', 'extortion', 'slavery'
]

def extract_entities_rule_based(text):
    """Extract entities using deterministic rules."""
    entities = []
    seen = set()
    
    for entity_type, pattern in ENTITY_PATTERNS.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            name = match.group(1) if match.group(1) else match.group()
            if name not in seen and len(name) > 2:
                entities.append({
                    "name": name,
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end()
                })
                seen.add(name)
    
    # Extract crime-related terms
    text_lower = text.lower()
    for keyword in CRIME_KEYWORDS:
        if keyword in text_lower:
            # Find actual case
            match = re.search(re.escape(keyword), text, re.IGNORECASE)
            if match:
                name = match.group()
                if name not in seen:
                    entities.append({
                        "name": name,
                        "type": "Crime",
                        "start": match.start(),
                        "end": match.end()
                    })
                    seen.add(name)
    
    return entities

def infer_relationships(entities, text):
    """Infer relationships based on proximity and rules."""
    relationships = []
    seen_pairs = set()
    
    # Sort entities by position in text
    sorted_entities = sorted(entities, key=lambda x: x.get('start', 0))
    
    # Create relationships between adjacent entities only (not all pairs)
    for i in range(len(sorted_entities) - 1):
        e1 = sorted_entities[i]
        e2 = sorted_entities[i + 1]
        
        # Skip if same entity
        if e1['name'] == e2['name']:
            continue
        
        # Check proximity (within 200 chars)
        distance = abs(e1.get('start', 0) - e2.get('start', 0))
        if distance > 200:
            continue  # Too far apart
        
        # Determine relationship type based on types
        rel_type = infer_relation_type(e1, e2, text)
        
        pair_key = tuple(sorted([e1['name'], e2['name']]))
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            relationships.append({
                "source": e1['name'],
                "target": e2['name'],
                "type": rel_type,
                "distance": distance
            })
    
    return relationships

def infer_relation_type(e1, e2, text):
    """Infer relationship type based on entity types and context."""
    type1 = e1.get('type', '')
    type2 = e2.get('type', '')
    
    # Organization + Date → "active_in"
    if (type1 == "Organization" and type2 == "Date") or \
       (type2 == "Organization" and type1 == "Date"):
        return "active_in"
    
    # Person + Organization → "affiliated_with"
    if (type1 == "Person" and type2 == "Organization") or \
       (type2 == "Person" and type1 == "Organization"):
        return "affiliated_with"
    
    # Crime + Location → "occurs_in"
    if (type1 == "Crime" and type2 == "Location") or \
       (type2 == "Crime" and type1 == "Location"):
        return "occurs_in"
    
    # Statistic + Date → "reported_in"
    if (type1 == "Statistic" and type2 == "Date") or \
       (type2 == "Statistic" and type1 == "Date"):
        return "reported_in"
    
    # Default: co-occurrence
    return "co-occurs_with"

def normalize_entity(name):
    """Normalize entity names."""
    # Strip common suffixes
    name = re.sub(r'\s*\(.*?\)\s*', '', name)
    return name.strip()

def build_high_quality_graph():
    """Build high-quality GraphRAG from all chunks."""
    print("="*70)
    print("BUILDING HIGH-QUALITY GRAPHRAG")
    print("="*70)
    
    # Load chunks
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data.get('chunks', [])
    print(f"Processing {len(chunks)} chunks...")
    
    all_entities = []
    all_relationships = []
    entity_chunks = defaultdict(set)  # entity → set of chunk_ids
    
    start_time = time.time()
    
    for i, chunk in enumerate(chunks):
        if i % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {i}/{len(chunks)} chunks | "
                  f"Entities: {len(all_entities)} | "
                  f"Rels: {len(all_relationships)}")
        
        chunk_id = f"chunk_{chunk.get('id', i)}"
        metadata = chunk.get('metadata', {})
        text = chunk.get('text', '')
        
        # Stage 1: Extract entities
        entities = extract_entities_rule_based(text)
        
        # Add metadata
        for entity in entities:
            entity['chunk_id'] = chunk_id
            entity['document'] = metadata.get('document_name', '')
            entity['year'] = metadata.get('year', '')
            entity['normalized_name'] = normalize_entity(entity['name'])
            all_entities.append(entity)
            entity_chunks[entity['normalized_name']].add(chunk_id)
        
        # Stage 2: Infer relationships
        if len(entities) >= 2:
            relationships = infer_relationships(entities, text)
            for rel in relationships:
                rel['chunk_id'] = chunk_id
                rel['document'] = metadata.get('document_name', '')
                all_relationships.append(rel)
    
    # Build summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("EXTRACTION COMPLETE!")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total entities: {len(all_entities)}")
    print(f"Total relationships: {len(all_relationships)}")
    print(f"Time: {elapsed/60:.1f} minutes")
    
    # Normalize and deduplicate
    print(f"\nNormalizing entities...")
    entity_groups = defaultdict(list)
    for entity in all_entities:
        norm_name = entity.get('normalized_name', entity['name'])
        entity_groups[norm_name].append(entity)
    
    normalized_entities = []
    entity_id_map = {}
    
    for norm_name, group in entity_groups.items():
        merged = group[0].copy()
        merged['name'] = norm_name
        merged['chunk_ids'] = list(set([e.get('chunk_id', '') for e in group]))
        merged['documents'] = list(set([e.get('document', '') for e in group]))
        normalized_entities.append(merged)
        for e in group:
            entity_id_map[e['name']] = norm_name
    
    print(f"Normalized: {len(all_entities)} -> {len(normalized_entities)} entities")
    
    # Normalize relationships
    print("Normalizing relationships...")
    normalized_rels = []
    seen = set()
    
    for rel in all_relationships:
        source = entity_id_map.get(rel.get('source', ''), rel.get('source', ''))
        target = entity_id_map.get(rel.get('target', ''), rel.get('target', ''))
        
        if source == target:
            continue
        
        pair_key = tuple(sorted([source, target, rel.get('type', 'co-occurs_with')]))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        
        normalized_rels.append({
            "source": source,
            "target": target,
            "type": rel.get('type', 'co-occurs_with'),
            "chunk_id": rel.get('chunk_id', '')
        })
    
    print(f"Normalized: {len(all_relationships)} -> {len(normalized_rels)} relationships")
    
    # Save
    output = {
        "entities": normalized_entities,
        "relationships": normalized_rels,
        "total_entities": len(normalized_entities),
        "total_relationships": len(normalized_rels),
        "method": "rule_based_inference",
        "chunks_processed": len(chunks)
    }
    
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"SAVED: {OUTPUT_FILE}")
    print(f"Final: {len(normalized_entities)} entities, {len(normalized_rels)} relationships")
    print(f"{'='*70}")

if __name__ == "__main__":
    import time
    build_high_quality_graph()
