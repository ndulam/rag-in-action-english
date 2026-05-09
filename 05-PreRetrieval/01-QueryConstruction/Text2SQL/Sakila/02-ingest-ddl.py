# ingest_ddl.py
import logging
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from pymilvus import model
import torch
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Initialize embedding function
embedding_function = model.dense.OpenAIEmbeddingFunction(model_name='text-embedding-3-large')

# 2. Load DDL list (assumes ddl_statements.yaml contains {table_name: "CREATE TABLE ..."})
with open("90-Data/sakila/ddl_statements.yaml","r") as f:
    ddl_map = yaml.safe_load(f)
    logging.info(f"[DDL] Loaded {len(ddl_map)} table/view definitions from YAML file")

# 3. Connect to Milvus
client = MilvusClient("text2sql_milvus_sakila.db")

# 4. Define Collection Schema
#    Fields: id, vector, table_name, ddl_text
vector_dim = len(embedding_function(["dummy"])[0])
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="ddl_text", dtype=DataType.VARCHAR, max_length=2000),
]
schema = CollectionSchema(fields, description="DDL Knowledge Base", enable_dynamic_field=False)

# 5. Create Collection (if it does not exist)
collection_name = "ddl_knowledge"
if not client.has_collection(collection_name):
    client.create_collection(collection_name=collection_name, schema=schema)
    logging.info(f"[DDL] Created new collection {collection_name}")
else:
    logging.info(f"[DDL] Collection {collection_name} already exists")

# 6. Add index to vector field
index_params = client.prepare_index_params()
index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE", params={"nlist": 1024})
client.create_index(collection_name=collection_name, index_params=index_params)

# 7. Batch insert DDL
data = []
texts = []
for tbl, ddl in ddl_map.items():
    texts.append(ddl)
    data.append({"table_name": tbl, "ddl_text": ddl})

logging.info(f"[DDL] Preparing to process DDL statements for {len(data)} tables/views")

# Generate all embeddings
embeddings = embedding_function(texts)
logging.info(f"[DDL] Successfully generated {len(embeddings)} vector embeddings")

# Organize into Milvus insert format
records = []
for emb, rec in zip(embeddings, data):
    rec["vector"] = emb
    records.append(rec)

res = client.insert(collection_name=collection_name, data=records)
logging.info(f"[DDL] Successfully inserted {len(records)} records into Milvus")
logging.info(f"[DDL] Insert result: {res}")

logging.info("[DDL] Knowledge base construction complete")
