from pymilvus import MilvusClient, DataType
import random

# 1. Set up Milvus client
client = MilvusClient(uri="http://localhost:19530")
COLLECTION_NAME = "ann_search_demo"

# Drop collection if it already exists
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

# 2. Create schema
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
schema.add_field(field_name="color", datatype=DataType.VARCHAR, max_length=100)
schema.add_field(field_name="likes", datatype=DataType.INT64)

# 3. Create collection
client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

# 4. Insert random vector data
num_vectors = 1000
vectors = [[random.random() for _ in range(128)] for _ in range(num_vectors)]
ids = list(range(num_vectors))
colors = [f"color_{random.randint(1, 1000)}" for _ in range(num_vectors)]
likes = [random.randint(1, 1000) for _ in range(num_vectors)]
entities = [{"id": ids[i], "vector": vectors[i], "color": colors[i], "likes": likes[i]} for i in range(num_vectors)]

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

# 7. Standard filtered search example
print("\n=== Standard Filtered Search ===")
query_vector = [random.random() for _ in range(128)]
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    limit=3,
    search_params={"metric_type": "L2"},
    filter='color like "color_%" and likes > 500',  # Filter: color starts with "color_" and likes > 500
    output_fields=["color", "likes"]  # Specify output fields
)

print("Filtered search results:")
for hits in results:
    for hit in hits:
        print(f"ID: {hit['id']}, Distance: {hit['distance']}, Color: {hit['entity']['color']}, Likes: {hit['entity']['likes']}")

# 8. Iterative filtered search example
print("\n=== Iterative Filtered Search ===")
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    limit=3,
    search_params={
        "metric_type": "L2",
        "hints": "iterative_filter"  # Enable iterative filtering
    },
    filter='color like "color_%" and likes > 500',
    output_fields=["color", "likes"]
)

print("Iterative filtered search results:")
for hits in results:
    for hit in hits:
        print(f"ID: {hit['id']}, Distance: {hit['distance']}, Color: {hit['entity']['color']}, Likes: {hit['entity']['likes']}")

# 9. Cleanup
client.release_collection(collection_name=COLLECTION_NAME)
