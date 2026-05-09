# Import the required libraries
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # requires: pip install llama-index-embeddings-huggingface
from llama_index.llms.deepseek import DeepSeek  # requires: pip install llama-index-llms-deepseek

from llama_index.core import Settings # Check what Settings are available
# https://docs.llamaindex.ai/en/stable/examples/llm/deepseek/
# Settings.llm = DeepSeek(model="deepseek-chat")
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh")
# Settings.llm = OpenAI(model="gpt-3.5-turbo")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load environment variables
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Create a DeepSeek LLM (calls the latest DeepSeek model via API)
llm = DeepSeek(
    model="deepseek-reasoner", # Use the latest reasoning model R1
    api_key=os.getenv("DEEPSEEK_API_KEY")  # Load the API key from environment variables
)

# Load data
documents = SimpleDirectoryReader(input_files=["90-Data/black-myth-wukong/settings.txt"]).load_data()

# Build the index
index = VectorStoreIndex.from_documents(
    documents,
    # llm=llm  # Set the language model for index building (generally not needed)
)

# Create the Q&A engine
query_engine = index.as_query_engine(
    llm=llm  # Set the generation model
    )

# Start Q&A
print(query_engine.query("What combat weapons are available in Black Myth: Wukong?"))
