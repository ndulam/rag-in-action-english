from pymilvus import MilvusClient, DataType
import random

# 1. Set up Milvus client
client = MilvusClient(uri="http://localhost:19530")
COLLECTION_NAME = "group_search_demo"

# Drop collection if it already exists
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

# 2. Create schema
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
schema.add_field(field_name="docId", datatype=DataType.INT64)
schema.add_field(field_name="chunk", datatype=DataType.VARCHAR, max_length=100)

# 3. Create collection
client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

# 4. Insert sample data
num_vectors = 1000
vectors = [[random.random() for _ in range(128)] for _ in range(num_vectors)]
ids = list(range(num_vectors))
doc_ids = [random.randint(1, 100) for _ in range(num_vectors)]  # Assume 100 documents
chunks = [f"chunk_{random.randint(1, 1000)}" for _ in range(num_vectors)]
entities = [{"id": ids[i], "vector": vectors[i], "docId": doc_ids[i], "chunk": chunks[i]} for i in range(num_vectors)]

client.insert(collection_name=COLLECTION_NAME, data=entities)

# 5. Create index
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vector",
    metric_type="L2",
    index_type="FLAT",
    index_name="vector_index",
    params={}
)
client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params,
    sync=True
)

# 6. Load collection
client.load_collection(collection_name=COLLECTION_NAME)

# 7. Basic group search example
print("\n=== Basic Group Search ===")
query_vector = [random.random() for _ in range(128)]
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    limit=5,  # Return 5 different document groups
    group_by_field="docId",  # Group by document ID
    output_fields=["docId", "chunk"]
)

print("Basic group search results:")
for hits in results:
    for hit in hits:
        print(f"Document ID: {hit['entity']['docId']}, Chunk: {hit['entity']['chunk']}, Distance: {hit['distance']}")

# 8. Group search with configured group size example
print("\n=== Group Search With Configured Group Size ===")
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    limit=3,  # Return 3 different document groups
    group_by_field="docId",
    group_size=2,  # Return 2 most similar results per group
    strict_group_size=True,  # Strictly ensure 2 results per group
    output_fields=["docId", "chunk"]
)

print("Group search results with configured group size:")
for hits in results:
    print(f"\nResults for document group {hits[0]['entity']['docId']}:")
    for hit in hits:
        print(f"Chunk: {hit['entity']['chunk']}, Distance: {hit['distance']}")

# 9. Cleanup
client.release_collection(collection_name=COLLECTION_NAME)
