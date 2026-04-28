"""
LLM-as-judge: Score RAG vs GraphRAG answers neutrally.
Metrics: relevance (1-5), completeness (1-5), grounding (1-5), retrieval_quality (1-5)
"""
import json
import requests
import time

EVAL_FILE = "D:/assortments/GraphRAG/code/rag/evaluation_results.json"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral"

def judge_with_llm(question, rag_answer, graphrag_answer, rag_sources, graphrag_sources):
    """Use Mistral to judge both answers neutrally."""
    prompt = f"""You are a neutral evaluator. Score BOTH answers fairly on a 1-5 scale (5=best).

Question: {question}

RAG Answer: {rag_answer[:400]}
RAG Sources: {', '.join(rag_sources)}

GraphRAG Answer: {graphrag_answer[:400]}
GraphRAG Sources: {', '.join(graphrag_sources)}

Score each independently on:
- relevance: Does answer address question? (1-5)
- completeness: Covers key aspects? (1-5)
- grounding: Supported by sources? (1-5)
- retrieval: Sources relevant? (1-5)

Return JSON only:
{{"rag": {{"relevance": 3, "completeness": 3, "grounding": 3, "retrieval_quality": 3}}, "graphrag": {{"relevance": 3, "completeness": 3, "grounding": 3, "retrieval_quality": 3}}}}"""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1}
        }, timeout=120)
        
        content = resp.json().get("message", {}).get("content", "").strip()
        
        # Try to extract JSON - look for the outermost braces
        import re
        # Find JSON object
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            return json.loads(json_str)
    except Exception as e:
        print(f"Judge error: {e}")
    return None

def main():
    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, result in enumerate(data["results"]):
        print(f"Judging question {i+1}/10: {result['question'][:50]}...")
        
        scores = judge_with_llm(
            result["question"],
            result["rag_answer"],
            result["graphrag_answer"],
            result["rag_sources"],
            result["graphrag_sources"]
        )
        
        if scores:
            result["metrics"] = scores
            print(f"  RAG: {scores['rag']}")
            print(f"  GraphRAG: {scores['graphrag']}")
        else:
            print("  Failed to judge")
        
        time.sleep(1)
    
    # Calculate summary
    rag_scores = {"relevance": [], "completeness": [], "grounding": [], "retrieval_quality": []}
    gr_scores = {"relevance": [], "completeness": [], "grounding": [], "retrieval_quality": []}
    
    for r in data["results"]:
        if r.get("metrics"):
            for k in rag_scores.keys():
                rag_scores[k].append(r["metrics"]["rag"][k])
                gr_scores[k].append(r["metrics"]["graphrag"][k])
    
    data["judge_summary"] = {
        "rag_averages": {k: round(sum(v)/len(v), 2) for k, v in rag_scores.items() if v},
        "graphrag_averages": {k: round(sum(v)/len(v), 2) for k, v in gr_scores.items() if v}
    }
    
    with open(EVAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("\n=== FINAL SCORES ===")
    print(f"RAG avg: {data['judge_summary']['rag_averages']}")
    print(f"GraphRAG avg: {data['judge_summary']['graphrag_averages']}")

if __name__ == "__main__":
    main()
