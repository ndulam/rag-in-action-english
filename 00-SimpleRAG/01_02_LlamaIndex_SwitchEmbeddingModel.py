# Import the required libraries
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # requires: pip install llama-index-embeddings-huggingface

# Load the local embedding model
# import os
# os.environ['HF_ENDPOINT']= 'https://hf-mirror.com' # If the default endpoint is blocked, you can set a mirror
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh" # Model path and name (will be downloaded from HuggingFace on first run)
    )

# Load data
documents = SimpleDirectoryReader(input_files=["90-Data/black-myth-wukong/settings.txt"]).load_data()

# Build the index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# Create the Q&A engine
query_engine = index.as_query_engine()

# Start Q&A
print(query_engine.query("What combat weapons are available in Black Myth: Wukong?"))
