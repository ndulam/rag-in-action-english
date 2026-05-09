"""
Note: Before running this code, make sure you have set the OpenAI API key in your environment variables.
On Linux/Mac, you can set it with the following command:
export OPENAI_API_KEY='your-api-key'

On Windows, you can set it with the following command:
set OPENAI_API_KEY=your-api-key

If you cannot obtain an OpenAI API key, that's okay — we have alternative options. Please refer to other scripts.
"""

# Line 1: Import the required libraries
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# Line 2: Load data
documents = SimpleDirectoryReader(input_files=["90-Data/black-myth-wukong/settings.txt"]).load_data()
# Line 3: Build the index
index = VectorStoreIndex.from_documents(documents)
# Line 4: Create the Q&A engine
query_engine = index.as_query_engine()
# Line 5: Start Q&A
print(query_engine.query("What combat weapons are available in Black Myth: Wukong?"))
