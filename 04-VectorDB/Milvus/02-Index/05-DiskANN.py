from pymilvus import MilvusClient, DataType
import random

# 1. Set up Milvus client
client = MilvusClient(uri="http://localhost:19530")
COLLECTION_NAME = "index_demo"

# Drop collection if it already exists
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

# 2. Create schema
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)

# 3. Create collection
client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

# 4. Insert random vector data
num_vectors = 1000
vectors = [[random.random() for _ in range(128)] for _ in range(num_vectors)]
ids = list(range(num_vectors))
entities = [{"id": ids[i], "vector": vectors[i]} for i in range(num_vectors)]

client.insert(collection_name=COLLECTION_NAME, data=entities)
# flush ensures data is persisted to disk
# client.flush([COLLECTION_NAME])

# 5. Create DiskANN index
index_params = MilvusClient.prepare_index_params()
index_params.add_index(
    field_name="vector",
    metric_type="L2",  # Supports L2, IP, or COSINE
    index_type="DISKANN",  # Use DiskANN index
    index_name="vector_index"
)
client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params,
    sync=True
)

# Verify index
print("Index list:", client.list_indexes(collection_name=COLLECTION_NAME))
print("Index details:", client.describe_index(
    collection_name=COLLECTION_NAME,
    index_name="vector_index"
))

# 6. Load first, then search
client.load_collection(collection_name=COLLECTION_NAME)
search_vectors = [[random.random() for _ in range(128)]]
results = client.search(
    collection_name=COLLECTION_NAME,
    data=search_vectors,
    ann_field="vector",
    limit=5,
    output_fields=["id"],
    search_params={
        "params": {
            "search_list": 32  # Candidate list size during search
        }
    }
)

print("\nSearch results:")
for hits in results:
    for hit in hits:
        print(f"ID: {hit['id']}, Distance: {hit['distance']}")

# Cleanup
client.release_collection(collection_name=COLLECTION_NAME)
# client.disconnect()
