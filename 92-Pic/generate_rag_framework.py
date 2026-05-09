"""
Generate English version of the RAG Technology Framework diagram.
Usage (from project root): python 92-Pic/generate_rag_framework.py
Output: 92-Pic/RAG-tech-framework-english.png
Requires: matplotlib (pip install matplotlib)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RAG-tech-framework-english.png')

C = dict(
    query_c   = '#fff8e1',
    pre_ret   = '#ede7f6',
    post_ret  = '#fce4ec',
    eval_     = '#e3f2fd',
    routing   = '#e8f5e9',
    vdb       = '#f3e5f5',
    gen       = '#fff3e0',
    adv       = '#f1f8e9',
    chunking  = '#e0f7fa',
    embedding = '#fffde7',
    indexing  = '#fbe9e7',
    loading   = '#f5f5f5',
)

def draw_section(ax, x, y, w, h, title, lines, bg,
                 tc='#1a237e', fst=8.5, fsb=7.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.08',
        facecolor=bg, edgecolor='#9e9e9e', linewidth=0.8, zorder=2))
    ax.text(x + w / 2, y + h - 0.14, title,
            ha='center', va='top', fontsize=fst, fontweight='bold',
            color=tc, zorder=3)
    sep_y = y + h - 0.29
    ax.plot([x + 0.08, x + w - 0.08], [sep_y, sep_y],
            color='#bdbdbd', linewidth=0.5, zorder=3)
    ty = sep_y - 0.06
    for line in lines:
        if line == '':
            ty -= 0.10
            continue
        ax.text(x + 0.12, ty, line,
                ha='left', va='top', fontsize=fsb, color='#333333', zorder=3)
        ty -= 0.195


fig = plt.figure(figsize=(22, 17), facecolor='white')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 22)
ax.set_ylim(0, 17)
ax.axis('off')

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(11, 16.78, 'RAG Technology Framework',
        ha='center', va='top', fontsize=17, fontweight='bold', color='#0d47a1')
ax.text(11, 16.28, 'Complete Architecture Landscape for Retrieval-Augmented Generation Systems',
        ha='center', va='top', fontsize=9, color='#546e7a')

# ── Layout constants ───────────────────────────────────────────────────────────
x0, x1, x2, x3 = 0.15, 5.65, 11.00, 16.35
wc = 5.35        # standard column width
wl = 5.50        # last column (Advanced RAG) slightly wider

r4_bot, r4_top = 0.12, 2.05   # Document Loading
r3_bot, r3_top = 2.20, 5.90   # Chunking / Embedding / Indexing
r2_bot, r2_top = 6.05, 10.15  # Routing / VDB / Generation
r1_bot, r1_top = 10.30, 15.90 # Top row

# ── Row 1 ─────────────────────────────────────────────────────────────────────
draw_section(ax, x0, r1_bot, wc, r1_top - r1_bot, 'Query Construction',
    ['Text-to-SQL',
     '  Natural language → SQL',
     '  (SQL2Pinecone)',
     '',
     'Text-to-Cypher',
     '  Natural language → Cypher',
     '  (GraphDB / GPT)',
     '',
     'Self-Query Retriever',
     '  Automatically constructs',
     '  structured metadata filters',
     '  directly from the query'],
    C['query_c'])

draw_section(ax, x1, r1_bot, wc, r1_top - r1_bot, 'Pre-Retrieval Processing',
    ['Query Rewriting',
     '  Reformulates user query',
     '',
     'Query Decomposition',
     '  Breaks complex question',
     '  into sub-questions',
     '',
     'Query Filtering',
     '  Removes irrelevant terms',
     '',
     'Multi-Query / RAG-Fusion',
     '  Generates multiple sub-queries',
     '  merges results via RRF',
     '',
     'HyDE',
     '  Generates hypothetical doc',
     '  to improve dense retrieval'],
    C['pre_ret'])

draw_section(ax, x2, r1_bot, wc, r1_top - r1_bot, 'Post-Retrieval Processing',
    ['Reranking',
     '  RRF (Reciprocal Rank Fusion)',
     '  CrossEncoder',
     '  ColBERT',
     '  RankGPT / RankLLM',
     '  RAG-Fusion',
     '  Keyword + semantic scoring',
     '  with text compression',
     '',
     'Active Checking — CRAG',
     '  If results are unsatisfactory,',
     '  re-searches multiple sources',
     '  with corrective retrieval'],
    C['post_ret'])

draw_section(ax, x3, r1_bot, wl, r1_top - r1_bot, 'Evaluation',
    ['Retrieval Evaluation',
     '  Precision',
     '  F1 Score',
     '  Recall',
     '  MAP  (Mean Avg Precision)',
     '  MRR  (Mean Reciprocal Rank)',
     '  P@K  (Precision at K)',
     '',
     'Generation Evaluation',
     '  BLEU',
     '  ROUGE',
     '  METEOR',
     '  Faithfulness',
     '  Factuality',
     '  Safety'],
    C['eval_'])

# ── Row 2 ─────────────────────────────────────────────────────────────────────
draw_section(ax, x0, r2_bot, wc, r2_top - r2_bot, 'Query Routing',
    ['Logic Routing',
     '  Rule-based routing to',
     '  different data sources',
     '',
     'Semantic Routing',
     '  LLM-based routing using',
     '  Prompt #1 / Prompt #2',
     '  to select the appropriate',
     '  retrieval method'],
    C['routing'])

draw_section(ax, x1, r2_bot, wc, r2_top - r2_bot, 'Vector Database',
    ['Dense Vector Indexes',
     '  FLAT  (exact search)',
     '  IVF   (inverted file index)',
     '  HNSW / LSH  (approx. NN)',
     '',
     'Relational Database',
     '  Structured metadata storage',
     '',
     'Platforms',
     '  Milvus · Chroma · FAISS',
     '  Pinecone · Weaviate · Qdrant'],
    C['vdb'])

draw_section(ax, x2, r2_bot, wc, r2_top - r2_bot, 'Generation',
    ['Prompt Engineering',
     '  Select LLM:',
     '  DeepSeek / Claude / GPT-4',
     '',
     'Output Parsing',
     '  Pydantic parser',
     '  JSON parser',
     '  Tool / Function calling',
     '',
     'Active Generation',
     '  Self-RAG',
     '  RRR (Rewrite-Retrieve-Read)',
     '  Decides: regenerate or retry'],
    C['gen'])

# Advanced RAG spans rows 1 + 2 + 3
draw_section(ax, x3, r3_bot, wl, r1_top - r3_bot, 'Advanced RAG',
    ['GraphRAG',
     '  Graph-based multi-hop',
     '  reasoning across entities',
     '',
     'Contextual Retrieval',
     '  Augments each chunk with',
     '  document-level context',
     '  before indexing',
     '',
     'Modular RAG',
     '  Plug-and-play modules,',
     '  flexible pipeline design',
     '',
     'Agentic RAG',
     '  Autonomous agent loops,',
     '  self-directed retrieval',
     '',
     'Multi-Modal RAG',
     '  Retrieval over images,',
     '  audio, and video'],
    C['adv'])

# ── Row 3 ─────────────────────────────────────────────────────────────────────
draw_section(ax, x0, r3_bot, wc, r3_top - r3_bot, 'Document Chunking',
    ['Chunking Strategies',
     '  By character',
     '  By paragraph',
     '  By heading',
     '  Intelligent / recursive split',
     '',
     'Semantic Chunking',
     '  Optimizes chunk boundaries',
     '  for retrieval quality and',
     '  generation coherence'],
    C['chunking'])

draw_section(ax, x1, r3_bot, wc, r3_top - r3_bot, 'Embedding',
    ['Dense Embedding Models',
     '  embed-multilingual-v3.0',
     '  jina-embeddings-v3',
     '  text-embedding-3-small',
     '  bge-large-en',
     '',
     'Sparse / Hybrid Embedding',
     '  TF-IDF, BM25, One-hot',
     '',
     'Specialized Embedding',
     '  Fine-Tuning, ColBERT'],
    C['embedding'])

draw_section(ax, x2, r3_bot, wc, r3_top - r3_bot, 'Indexing',
    ['Small-to-Large Context',
     '  Node sentence window',
     '  Sliding window retriever',
     '',
     'Hierarchical Indexing',
     '  Cluster-based indexing',
     '  RAPTOR',
     '',
     'Multi-Representation',
     '  Ensemble retriever',
     '  Multi-vector retriever'],
    C['indexing'])

# ── Row 4: Document Loading ────────────────────────────────────────────────────
draw_section(ax, x0, r4_bot, 16.05, r4_top - r4_bot, 'Document Loading',
    ['LangChain Documents  /  LlamaIndex Nodes     |     '
     'Parsers: PyPDF · PyMuPDF · Unstructured · JSON/XML Loaders · Web Loaders · Camelot'],
    C['loading'], fst=9.5, fsb=8.0)

# ── Data-flow note ─────────────────────────────────────────────────────────────
ax.text(0.15, 0.06,
        'Indexing pipeline: Document Loading → Chunking → Embedding → Vector DB  '
        '│  Query pipeline: Query → Pre-Retrieval → Vector DB → Post-Retrieval → Generation → Answer',
        ha='left', va='bottom', fontsize=6.5, color='#78909c', style='italic')

plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved: {OUT}')
plt.close()
