"""
Run a large language model locally using Ollama — no OpenAI API key required.

1. Install the Ollama Server:
   - Windows: Visit https://ollama.com/download to download the installer
   - Linux/Mac: Run curl -fsSL https://ollama.com/install.sh | sh

2. Download and run a model:
   - Open a terminal and run the following commands to download a model:
     ollama pull qwen:7b  # Download the Qwen 7B model
     # or
     ollama pull llama2:7b  # Download the Llama2 7B model
     # or
     ollama pull mistral:7b  # Download the Mistral 7B model

3. Set environment variables:
   - Add the following to your .env file:
     OLLAMA_MODEL=qwen:7b  # Or the name of another downloaded model
"""

# Line 1: Import the required libraries
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama # requires: pip install llama-index-llms-ollama
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load the local embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

# Create an Ollama LLM, default URL: http://localhost:11434
llm = Ollama(
    model=os.getenv("OLLAMA_MODEL"),
    request_timeout=300.0
)

# Line 2: Load data
documents = SimpleDirectoryReader(input_files=["90-Data/black-myth-wukong/settings.txt"]).load_data()

# Line 3: Build the index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# Line 4: Create the Q&A engine
query_engine = index.as_query_engine(
    llm=llm
)

# Line 5: Start Q&A
print(query_engine.query("What combat weapons are available in Black Myth: Wukong?"))
