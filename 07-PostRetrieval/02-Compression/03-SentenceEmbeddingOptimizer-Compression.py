from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
# Load documents
documents = SimpleDirectoryReader("data/shanxi-culture-tourism").load_data()
index = VectorStoreIndex.from_documents(documents)
# Query without optimization
print("Without optimization:")
query_engine = index.as_query_engine()
response = query_engine.query("What are the main tourist attractions in Shanxi Province?")
print(f"Answer: {response}")
# Query with optimization (percentile cutoff)
print("\nWith optimization (percentile_cutoff=0.5):")
query_engine = index.as_query_engine(node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)])
response = query_engine.query("What are the main tourist attractions in Shanxi Province?")
print(f"Answer: {response}")
# Query with optimization (threshold cutoff)
print("\nWith optimization (threshold_cutoff=0.7):")
query_engine = index.as_query_engine(node_postprocessors=[SentenceEmbeddingOptimizer(threshold_cutoff=0.7)])
response = query_engine.query("What are the main tourist attractions in Shanxi Province?")
print(f"Answer: {response}")
