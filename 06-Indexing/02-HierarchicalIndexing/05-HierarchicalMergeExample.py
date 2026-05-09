from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Configure global settings
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
# Prepare game knowledge text
game_knowledge = """"Chronicles of the Divine: Wukong" features an exquisitely designed combat system. Players can freely switch between multiple combat forms during battle, each with its own unique advantages. In the Vajra form... the Demon Buddha form..."""
# Create Document objects
documents = [Document(text=game_knowledge)]
# Create hierarchical node parser and process documents
# Use HierarchicalNodeParser to create text hierarchy
# chunk_sizes represents the text chunk sizes for different levels
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[256, 128, 64]  # Chunk sizes from root nodes to leaf nodes
)
nodes = node_parser.get_nodes_from_documents(documents)
# Get leaf nodes (smallest granularity text chunks) and root nodes
leaf_nodes = get_leaf_nodes(nodes)
root_nodes = get_root_nodes(nodes)
# Build storage and index
# Create document store and add all nodes
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
# Create storage context
storage_context = StorageContext.from_defaults(docstore=docstore)
# Create vector index for leaf nodes
base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context
)
# Create base retriever and auto-merging retriever
base_retriever = base_index.as_retriever(similarity_top_k=6)
auto_merging_retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    verbose=True  # Show merging process
)
# Prepare test questions
test_questions = [
    # "What are the differences between the Vajra form and the Demon Buddha form in the game?",
    "What are the characteristics of the Ruyi Jingu Bang in different forms?",
    # "How is the difficulty design of the game?"
]
print("=== Results from auto-merging retriever ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    # Retrieve using auto-merging retriever
    merge_nodes = auto_merging_retriever.retrieve(question)
    print(f"Retrieved {len(merge_nodes)} merged nodes:")
    for node in merge_nodes:
        print(f"\nSimilarity: {node.score}")
        print(f"Content: {node.node.text}")
        print("-" * 50)
print("\n=== Results from base retriever (for comparison) ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    # Retrieve using base retriever
    base_nodes = base_retriever.retrieve(question)
    print(f"Retrieved {len(base_nodes)} base nodes:")
    for node in base_nodes:
        print(f"\nSimilarity: {node.score}")
        print(f"Content: {node.node.text}")
        print("-" * 50)
