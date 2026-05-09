# ingest_q2sql.py
import logging
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from pymilvus import model
import torch
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Initialize embedding function
embedding_function = model.dense.OpenAIEmbeddingFunction(model_name='text-embedding-3-large')

# 2. Load Q->SQL pairs (assumes q2sql_pairs.json is an array, each item: { "question": ..., "sql": ... })
with open("90-Data/sakila/q2sql_pairs.json", "r") as f:
    pairs = json.load(f)
    logging.info(f"[Q2SQL] Loaded {len(pairs)} question-answer pairs from JSON file")

# 3. Connect to Milvus
client = MilvusClient("text2sql_milvus_sakila.db")

# 4. Define Collection Schema
vector_dim = len(embedding_function(["dummy"])[0])
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="sql_text", dtype=DataType.VARCHAR, max_length=2000),
]
schema = CollectionSchema(fields, description="Q2SQL Knowledge Base", enable_dynamic_field=False)

# 5. Create Collection (if it does not exist)
collection_name = "q2sql_knowledge"
if not client.has_collection(collection_name):
    client.create_collection(collection_name=collection_name, schema=schema)
    logging.info(f"[Q2SQL] Created new collection {collection_name}")
else:
    logging.info(f"[Q2SQL] Collection {collection_name} already exists")

# 6. Add index to vector field
index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE", params={"nlist": 1024})
client.create_index(collection_name=collection_name, index_params=index_params)

# 7. Batch insert Q2SQL pairs
data = []
texts = []
for pair in pairs:
    texts.append(pair["question"])
    data.append({"question": pair["question"], "sql_text": pair["sql"]})

logging.info(f"[Q2SQL] Preparing to process {len(data)} question-answer pairs")

# Generate all embeddings
embeddings = embedding_function(texts)
logging.info(f"[Q2SQL] Successfully generated {len(embeddings)} vector embeddings")

# Organize into Milvus insert format
records = []
for emb, rec in zip(embeddings, data):
    rec["vector"] = emb
    records.append(rec)

res = client.insert(collection_name=collection_name, data=records)
logging.info(f"[Q2SQL] Successfully inserted {len(records)} records into Milvus")
logging.info(f"[Q2SQL] Insert result: {res}")

logging.info("[Q2SQL] Knowledge base construction complete")
