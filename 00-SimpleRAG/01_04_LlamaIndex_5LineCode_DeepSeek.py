# Line 1: Import the required libraries
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load the local embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

# Create a DeepSeek LLM
llm = DeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY")
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
