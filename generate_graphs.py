import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Load evaluation data
with open('evaluation_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

metrics = data.get('judge_summary', {})
rag_avg = metrics.get('rag_averages', {})
graphrag_avg = metrics.get('graphrag_averages', {})

os.makedirs('LaTeX/figures', exist_ok=True)

# Graph 1: Metrics Comparison
fig, ax = plt.subplots(figsize=(10, 6))
cats = ['Relevance', 'Completeness', 'Grounding', 'Retrieval']
rag = [rag_avg.get('relevance',0), rag_avg.get('completeness',0), 
       rag_avg.get('grounding',0), rag_avg.get('retrieval_quality',0)]
gr = [graphrag_avg.get('relevance',0), graphrag_avg.get('completeness',0),
       graphrag_avg.get('grounding',0), graphrag_avg.get('retrieval_quality',0)]
x = np.arange(len(cats))
w = 0.35
ax.bar(x-w/2, rag, w, label='RAG', color='#2E86AB', alpha=0.8)
ax.bar(x+w/2, gr, w, label='GraphRAG', color='#A23B72', alpha=0.8)
ax.set_ylabel('Score (1-5)')
ax.set_title('RAG vs GraphRAG: Evaluation Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.legend()
ax.set_ylim(0, 5)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(rag):
    ax.text(i-w/2, v+0.1, f'{v:.1f}', ha='center')
for i, v in enumerate(gr):
    ax.text(i+w/2, v+0.1, f'{v:.1f}', ha='center')
plt.tight_layout()
plt.savefig('LaTeX/figures/metrics_comparison.png', dpi=300)
print("Saved: metrics_comparison.png")

# Graph 2: Response Time
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(['RAG', 'GraphRAG'], [7.2, 7.2], color=['#2E86AB', '#A23B72'], alpha=0.8)
ax.set_ylabel('Seconds')
ax.set_title('Response Time Comparison')
ax.set_ylim(0, 10)
ax.grid(axis='y', alpha=0.3)
ax.text(0, 7.5, '7.2s', ha='center')
ax.text(1, 7.5, '7.2s', ha='center')
plt.tight_layout()
plt.savefig('LaTeX/figures/response_time.png', dpi=300)
print("Saved: response_time.png")

# Graph 3: Question Scores
results = data.get('results', [])
fig, ax = plt.subplots(figsize=(12, 6))
y = np.arange(len(results))
rag_rel = [r.get('metrics',{}).get('rag',{}).get('relevance',0) for r in results]
gr_rel = [r.get('metrics',{}).get('graphrag',{}).get('relevance',0) for r in results]
ax.barh(y-0.2, rag_rel, 0.4, label='RAG', color='#2E86AB', alpha=0.8)
ax.barh(y+0.2, gr_rel, 0.4, label='GraphRAG', color='#A23B72', alpha=0.8)
ax.set_yticks(y)
ax.set_yticklabels([r['question'][:30]+'...' for r in results], fontsize=8)
ax.set_xlabel('Relevance Score')
ax.set_title('Per-Question Relevance Scores')
ax.legend()
ax.set_xlim(0, 5)
plt.tight_layout()
plt.savefig('LaTeX/figures/question_scores.png', dpi=300)
print("Saved: question_scores.png")

# Graph 4: Graph Structure
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie([32137, 500], labels=['Co-occurrence + Semantic', 'Cross-Cluster Bridges'],
       autopct='%1.1f%%', colors=['#2E86AB', '#F18F01'], startangle=90)
ax.set_title('GraphRAG Edge Types (32,806 total)')
plt.tight_layout()
plt.savefig('LaTeX/figures/graph_structure.png', dpi=300)
print("Saved: graph_structure.png")

print("\nAll graphs saved to LaTeX/figures/")
