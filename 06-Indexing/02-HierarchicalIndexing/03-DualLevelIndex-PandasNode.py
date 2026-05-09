# Dual-tier retrieval - Billionaires - requires pip install openpyxl
import os
from dotenv import load_dotenv
import pandas as pd
import logging
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Configure global settings
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 3. Load Excel file and prepare data
excel_file = "90-Data/complex-pdf/top-10-billionaires/WorldTop10Billionaires.xlsx"

# Initialize Node Parser
node_parser = SentenceSplitter(
    chunk_size=1024,  # Size of each chunk
    chunk_overlap=20,  # Overlap size between chunks
    include_metadata=True  # Include metadata
)

# Store all table DataFrames and query engines
table_dfs = []
df_query_engines = []
documents = []

# Read all sheets from the Excel file and insert data
with pd.ExcelFile(excel_file) as xls:
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            logging.info(f"Processing sheet: {sheet_name}")

            # Convert DataFrame to string
            table_content = df.to_string(index=False)

            # Create Document object
            doc = Document(
                text=table_content,
                metadata={"table_name": sheet_name}
            )
            documents.append(doc)

            # Store DataFrame and create query engine
            table_dfs.append(df)
            df_query_engine = PandasQueryEngine(df, llm=Settings.llm)
            df_query_engines.append(df_query_engine)

            logging.info(f"Successfully processed sheet: {sheet_name}")

        except Exception as e:
            logging.error(f"Error processing sheet {sheet_name}: {str(e)}")
            logging.error(f"Error details: {e.__class__.__name__}")
            continue

# Create IndexNode objects
summaries = [
    f"This node provides information about the world's richest billionaires in {sheet_name}"
    for sheet_name in xls.sheet_names
]

df_nodes = [
    IndexNode(text=summary, index_id=f"pandas{idx}") # Details for each table
    for idx, summary in enumerate(summaries)
]

# Create query engine mapping
df_id_query_engine_mapping = {
    f"pandas{idx}": df_query_engine
    for idx, df_query_engine in enumerate(df_query_engines)
}

# Create vector index
vector_index = VectorStoreIndex(documents + df_nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

# Create recursive retriever
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_id_query_engine_mapping,
    verbose=True,
)

# Create response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="compact")

# Create query engine
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever, response_synthesizer=response_synthesizer
)

def generate_answer(question):
    # Generate answer using query engine
    response = query_engine.query(question)
    return str(response)

# Test example
if __name__ == "__main__":
    test_question = "Who was the world's richest person in 2020? How much was their wealth?"
    answer = generate_answer(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
