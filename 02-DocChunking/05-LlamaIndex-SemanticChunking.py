from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
documents = SimpleDirectoryReader(input_files=["90-Data/black-myth-wukong/black-myth-wukong-wiki.txt"]).load_data()

# Create the semantic chunker
splitter = SemanticSplitterNodeParser(
    buffer_size=3,  # Buffer size
    breakpoint_percentile_threshold=90, # Breakpoint percentile threshold
    embed_model=OpenAIEmbedding()     # Embedding model to use
)
# Create a basic sentence chunker (for comparison)
base_splitter = SentenceSplitter(
    # chunk_size=512
)


'''
buffer_size:
  Default value is 1.
  This parameter controls how many sentences are grouped together when evaluating semantic similarity.
  When set to 1, each sentence is considered individually.
  When set to a value greater than 1, multiple sentences are grouped together for evaluation.
  For example, if set to 3, every 3 sentences are evaluated as a group for semantic similarity.

breakpoint_percentile_threshold:
  Default value is 95.
  This parameter controls when to create a split point between sentence groups.
  It represents the percentile threshold of cosine dissimilarity.
  When the dissimilarity between sentence groups exceeds this threshold, a new node is created.
  Lower values generate more nodes (because the split threshold is easier to reach).
  Higher values generate fewer nodes (because greater dissimilarity is required to split).

These two parameters jointly influence text splitting:
  buffer_size determines the granularity of semantic similarity evaluation.
  breakpoint_percentile_threshold determines the strictness of splitting.
Example:
  If buffer_size=2 and breakpoint_percentile_threshold=90:
    Every 2 sentences are grouped together.
    A split occurs when the dissimilarity between groups exceeds 90%.
    This produces relatively more nodes.
  If buffer_size=3 and breakpoint_percentile_threshold=98:
    Every 3 sentences are grouped together.
    Greater dissimilarity is required before splitting.
    This produces relatively fewer nodes.
'''


# Use the semantic chunker to chunk the documents
semantic_nodes = splitter.get_nodes_from_documents(documents)
print("\n=== Semantic Chunking Results ===")
print(f"Number of chunks from semantic chunker: {len(semantic_nodes)}")
for i, node in enumerate(semantic_nodes, 1):
    print(f"\n--- Semantic Chunk {i} ---")
    print(f"Content:\n{node.text}")
    print("-" * 50)

# Use the basic sentence chunker to chunk the documents
base_nodes = base_splitter.get_nodes_from_documents(documents)
print("\n=== Basic Sentence Chunking Results ===")
print(f"Number of chunks from basic sentence chunker: {len(base_nodes)}")
for i, node in enumerate(base_nodes, 1):
    print(f"\n--- Sentence Chunk {i} ---")
    print(f"Content:\n{node.text}")
    print("-" * 50)
