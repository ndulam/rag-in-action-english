# Prepare sample dataset
import pandas as pd
data_records = [
    {
        "monster_id": "BM001",
        "monster_name": "Tiger Vanguard",
        "location": "Bamboo Forest Pass",
        "difficulty": "High",
        "synonyms": "Fierce Tiger Demon, Tiger Demon",
        "description": "A fierce tiger-type monster appearing in the bamboo forest level, with immense strength."
    },
    {
        "monster_id": "BM002",
        "monster_name": "Fire Ape",
        "location": "Volcanic Cave",
        "difficulty": "Low",
        "synonyms": "Flame Ape, Blazing Ape",
        "description": "An ape-type monster living in the volcanic cave, just a minor comic-relief soldier."
    },]
df = pd.DataFrame(data_records)

# Establish/connect to Milvus
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from pymilvus import model
db_path = "./wukong.db"
client = MilvusClient(db_path)
collection_name = "Wukong_Monsters"

# Get the vector dimensions from the embedding model
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
embedding_function = SentenceTransformerEmbeddingFunction(model_name='BAAI/bge-large-zh')
sample_embedding = embedding_function(["sample text"])[0]
vector_dim = len(sample_embedding)

# Define collection schema and create collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
    FieldSchema(name="monster_id", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="monster_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="difficulty", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="synonyms", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=500),
]
schema = CollectionSchema(fields, description=" Wukong Monsters", enable_dynamic_field=True)
if not client.has_collection(collection_name):
    client.create_collection(collection_name=collection_name, schema=schema)

# Create index
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="L2",
    params={"nlist": 1024}
)
client.create_index(
    collection_name=collection_name,
    index_params=index_params
)

# Batch insert data
from tqdm import tqdm
for start_idx in tqdm(range(0, len(df)), desc="Inserting data"):
    row = df.iloc[start_idx]
    # Prepare vector text
    doc_parts = [str(row['monster_name'])]
    if row['synonyms']:
        doc_parts.append(f"(Aliases: {row['synonyms']})")
    if row['location']:
        doc_parts.append(f"Location: {row['location']}")
    if row['description']:
        doc_parts.append(f"Description: {row['description']}")
    doc_text = "; ".join(doc_parts)
    # Generate vector and insert data
    embedding = embedding_function([doc_text])[0]
    data_to_insert = [{
        "vector": embedding,
        "monster_id": str(row["monster_id"]),
        "monster_name": str(row["monster_name"]),
        "location": str(row["location"]),
        "difficulty": str(row["difficulty"]),
        "synonyms": str(row["synonyms"]),
        "description": str(row["description"])
    }]
    client.insert(collection_name=collection_name, data=data_to_insert)

# Test search
search_query = "high difficulty monster"
search_embedding = embedding_function([search_query])[0]
search_result = client.search(
    collection_name=collection_name,
    data=[search_embedding.tolist()],
    limit=3,
    output_fields=["monster_name", "location", "difficulty", "synonyms"]
)
print(f"Search results for '{search_query}':", search_result)

# Test conditional query
query_result = client.query(
    collection_name=collection_name,
    filter="difficulty == 'Low'",
    output_fields=["monster_name", "location", "difficulty", "synonyms"]
)
print(f"Monsters with Low difficulty:", query_result)
