"""
Generate English version of the RAG Bootcamp course banner (replaces RAG.PNG).
Usage (from project root): python 92-Pic/generate_rag_banner.py
Output: 92-Pic/RAG-english.png
Requires: matplotlib (pip install matplotlib)
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RAG-english.png')

fig = plt.figure(figsize=(16, 10), facecolor='#0a1628')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('#0a1628')

# Subtle background gradient
for i in range(40):
    r = 0.05 + i * 0.001
    g = 0.09 + i * 0.001
    b = 0.18 + i * 0.002
    ax.add_patch(Rectangle((0, i * 0.25), 16, 0.25,
                            facecolor=(r, g, b), zorder=0))

# Top accent bar
ax.add_patch(Rectangle((0, 9.55), 16, 0.45, facecolor='#1565c0', zorder=1))
ax.text(8, 9.78, 'DeepSeek-based RAG System Development Course',
        ha='center', va='center', fontsize=9, color='#90caf9', zorder=2)

# Bottom accent bar
ax.add_patch(Rectangle((0, 0), 16, 0.45, facecolor='#1565c0', zorder=1))
ax.text(8, 0.22, 'RAG is the best on-ramp for developers entering AI  '
        '—  covering 10 core technologies across the full RAG development workflow',
        ha='center', va='center', fontsize=7.5, color='#e3f2fd', style='italic', zorder=2)

# ── Main title ──────────────────────────────────────────────────────────────────
ax.text(7.2, 9.25, 'Advanced LLM  RAG  Bootcamp',
        ha='center', va='top', fontsize=25, fontweight='bold', color='white', zorder=5)
ax.text(7.2, 8.62, 'Hands-on RAG System Development with DeepSeek',
        ha='center', va='top', fontsize=12, color='#90caf9', zorder=5)

# RAG badge (top-right)
ax.add_patch(FancyBboxPatch((13.8, 8.55), 2.0, 0.95,
                            boxstyle='round,pad=0.12',
                            facecolor='#1565c0', edgecolor='#42a5f5',
                            linewidth=2, zorder=4))
ax.text(14.8, 9.02, 'RAG', ha='center', va='center',
        fontsize=22, fontweight='bold', color='white', zorder=5)

# Divider
ax.plot([0.4, 15.6], [8.22, 8.22], color='#1565c0', linewidth=1.5, zorder=5)

# ── Stats boxes ─────────────────────────────────────────────────────────────────
stats = [
    ('40 hrs',          'Course Content'),
    ('10',              'Core RAG Skills'),
    ('4',               'Real-World Projects'),
    ('DeepSeek+Cursor', 'Trending Topics'),
    ('Lifetime',        'Access to Materials'),
]
xs = [1.2, 4.0, 6.8, 9.6, 12.5]
for (val, label), xp in zip(stats, xs):
    ax.add_patch(FancyBboxPatch((xp - 1.2, 6.95), 2.4, 1.05,
                                boxstyle='round,pad=0.12',
                                facecolor='#0d47a1', edgecolor='#42a5f5',
                                linewidth=1.2, alpha=0.85, zorder=4))
    ax.text(xp, 7.73, val, ha='center', va='top',
            fontsize=12 if len(val) <= 8 else 9,
            fontweight='bold', color='white', zorder=5)
    ax.text(xp, 7.05, label, ha='center', va='bottom',
            fontsize=7.5, color='#90caf9', zorder=5)

# ── Tagline ─────────────────────────────────────────────────────────────────────
ax.text(8, 6.58, '  ★  RAG — The Best On-Ramp for Developers Entering AI  ★  ',
        ha='center', va='top', fontsize=11, fontweight='bold', color='#ffcc02',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#0d47a1',
                  edgecolor='#ffcc02', linewidth=1.5),
        zorder=5)

# Divider
ax.plot([0.4, 15.6], [6.05, 6.05], color='#1565c0', linewidth=1,
        linestyle='--', zorder=5)

# ── 10 core technologies ─────────────────────────────────────────────────────────
ax.text(8, 5.85, '10 Core Technologies  —  Full RAG Development Workflow',
        ha='center', va='top', fontsize=11, fontweight='bold',
        color='#42a5f5', zorder=5)

left_techs = [
    '▸  Document Loading & Chunking',
    '▸  Vector Embedding  (Dense + Sparse)',
    '▸  Vector Database & Indexing',
    '▸  Pre-Retrieval: Query Rewriting, HyDE, RAG-Fusion',
    '▸  Query Construction: Text-to-SQL, Text-to-Cypher',
]
right_techs = [
    '▸  Query Routing: Logic & Semantic',
    '▸  Post-Retrieval: Reranking, CRAG',
    '▸  Generation: Prompt Engineering, Output Parsing',
    '▸  Evaluation: RAGAS, TruLens, DeepEval',
    '▸  Advanced RAG: GraphRAG, Agentic, Multi-Modal',
]

for i, t in enumerate(left_techs):
    ax.text(0.5, 5.30 - i * 0.54, t,
            ha='left', va='top', fontsize=9, color='#e1f5fe', zorder=5)

for i, t in enumerate(right_techs):
    ax.text(8.2, 5.30 - i * 0.54, t,
            ha='left', va='top', fontsize=9, color='#e1f5fe', zorder=5)

plt.savefig(OUT, dpi=150, bbox_inches='tight',
            facecolor='#0a1628', edgecolor='none')
print(f'Saved: {OUT}')
plt.close()
