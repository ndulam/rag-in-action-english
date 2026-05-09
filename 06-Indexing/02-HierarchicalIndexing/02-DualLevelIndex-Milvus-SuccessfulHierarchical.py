# Dual-tier retrieval - Billionaires - requires pip install openpyxl
import os
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize embedding model
embedding_function = SentenceTransformer(
    'BAAI/bge-m3',
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    trust_remote_code=True
)

# Connect to Milvus
client = MilvusClient("richman_bge_m3_v2.db")

# 1. Create summary vector database
summary_collection_name = "billionaires_summary"
summary_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100)
]

summary_schema = CollectionSchema(summary_fields, "Billionaires yearly summary")
if not client.has_collection(summary_collection_name):
    client.create_collection(
        collection_name=summary_collection_name,
        schema=summary_schema
    )

# 2. Create details vector database
details_collection_name = "billionaires_details"
details_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000)  # Store entire table content
]

details_schema = CollectionSchema(details_fields, "Billionaires detailed information")
if not client.has_collection(details_collection_name):
    client.create_collection(
        collection_name=details_collection_name,
        schema=details_schema
    )

# 3. Load Excel file and prepare data
excel_file = "90-Data/complex-pdf/top-10-billionaires/WorldTop10Billionaires.xlsx"

# Read all sheets from the Excel file and insert data
with pd.ExcelFile(excel_file) as xls:
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            logging.info(f"Processing sheet: {sheet_name}")

            # Insert summary data - only store table name
            summary_embedding = embedding_function.encode([sheet_name])[0]

            client.insert(
                collection_name=summary_collection_name,
                data=[{
                    "vector": summary_embedding.tolist(),
                    "table_name": sheet_name
                }]
            )

            # Insert details data - store entire table content
            # Convert entire DataFrame to string
            table_content = df.to_string(index=False)
            detail_embedding = embedding_function.encode([table_content])[0]

            client.insert(
                collection_name=details_collection_name,
                data=[{
                    "vector": detail_embedding.tolist(),
                    "table_name": sheet_name,
                    "content": table_content
                }]
            )

            logging.info(f"Successfully processed sheet: {sheet_name}")

        except Exception as e:
            logging.error(f"Error processing sheet {sheet_name}: {str(e)}")
            logging.error(f"Error details: {e.__class__.__name__}")
            continue

# 4. Create indexes
# Delete existing indexes (if any)
try:
    client.drop_index(collection_name=summary_collection_name, index_name="vector")
except Exception as e:
    logging.warning(f"Error deleting summary index: {str(e)}")

try:
    client.drop_index(collection_name=details_collection_name, index_name="vector")
except Exception as e:
    logging.warning(f"Error deleting details index: {str(e)}")

# Create new indexes
try:
    # Use prepare_index_params method to create index parameters
    summary_index_params = client.prepare_index_params()
    summary_index_params.add_index(
        field_name="vector",  # Specify which field to create an index for, here it's the vector field
        index_type="IVF_FLAT",  # Index type
        metric_type="COSINE",  # Use cosine similarity as vector similarity measure
        params={"nlist": 1024}  # Index parameters
    )

    client.create_index(
        collection_name=summary_collection_name,
        index_params=summary_index_params
    )
    logging.info("Successfully created summary index")
except Exception as e:
    logging.error(f"Error creating summary index: {str(e)}")

try:
    # Use prepare_index_params method to create index parameters
    details_index_params = client.prepare_index_params()
    details_index_params.add_index(
        field_name="vector",  # Specify which field to create an index for, here it's the vector field
        index_type="IVF_FLAT",  # Index type
        metric_type="COSINE",  # Use cosine similarity as vector similarity measure
        params={"nlist": 1024}  # Index parameters
    )

    client.create_index(
        collection_name=details_collection_name,
        index_params=details_index_params
    )
    logging.info("Successfully created details index")
except Exception as e:
    logging.error(f"Error creating details index: {str(e)}")

# Load collections to activate indexes
try:
    client.load_collection(summary_collection_name)
    client.load_collection(details_collection_name)
    logging.info("Successfully loaded collections")
except Exception as e:
    logging.error(f"Error loading collections: {str(e)}")

def search_relevant_table(question):
    # First-tier retrieval: search the summary collection for the most relevant sheet
    query_embedding = embedding_function.encode([question])[0]

    summary_results = client.search(
        collection_name=summary_collection_name,
        data=[query_embedding.tolist()],
        limit=1,
        output_fields=["table_name"],
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
    )

    if not summary_results or not summary_results[0]:
        return None, None

    matched_table = summary_results[0][0]['entity']['table_name']

    # Second-tier retrieval: search the details collection for specific information
    details_results = client.search(
        collection_name=details_collection_name,
        data=[query_embedding.tolist()],
        filter=f"table_name == '{matched_table}'",
        limit=1,
        output_fields=["content"],
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
    )

    if not details_results or not details_results[0]:
        return None, None

    return matched_table, details_results[0][0]['entity']['content']

def generate_answer(question):
    # Retrieve relevant information
    table_name, content = search_relevant_table(question)

    if not table_name or not content:
        return "Sorry, no relevant information was found."

    # Build prompt
    prompt = f"""Answer the question based on the following table information:

Table name: {table_name}

Table content:
{content}

Question: {question}

Please provide a detailed answer based on the above information:"""

    # Generate answer using DeepSeek
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=1024
    )

    return response.choices[0].message.content

# Test example
if __name__ == "__main__":
    test_question = "Who was the world's richest person in 2023? How much was their wealth?"
    answer = generate_answer(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
