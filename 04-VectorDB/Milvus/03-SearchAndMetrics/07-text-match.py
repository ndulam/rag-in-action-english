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
schema.add_field(
    field_name='text',
    datatype=DataType.VARCHAR,
    max_length=1000,
    enable_analyzer=True,  # Enable text analysis
    enable_match=True      # Enable text matching
)

# 3. Create collection
client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

# 4. Insert random vector data
num_vectors = 1000
vectors = [[random.random() for _ in range(128)] for _ in range(num_vectors)]
ids = list(range(num_vectors))
colors = [f"color_{random.randint(1, 1000)}" for _ in range(num_vectors)]
texts = [f"text_{random.randint(1, 1000)}" for _ in range(num_vectors)]  # Add random text
entities = [{"id": ids[i], "vector": vectors[i], "color": colors[i], "text": texts[i]} for i in range(num_vectors)]

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

# 7. Single-vector search example
print("\n=== Single-Vector Search ===")
query_vector = [random.random() for _ in range(128)]
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    limit=3,
    search_params={"metric_type": "L2"}
)

print("Search results:")
for hits in results:
    for hit in hits:
        print(f"ID: {hit['id']}, Distance: {hit['distance']}")

# 8. Batch vector search example
print("\n=== Batch Vector Search ===")
query_vectors = [[random.random() for _ in range(128)] for _ in range(2)]
results = client.search(
    collection_name=COLLECTION_NAME,
    data=query_vectors,
    anns_field="vector",
    limit=3,
    search_params={"metric_type": "L2"}
)

print("Batch search results:")
for i, hits in enumerate(results):
    print(f"\nResults for query vector {i+1}:")
    for hit in hits:
        print(f"ID: {hit['id']}, Distance: {hit['distance']}")

# 9. Search with output fields example
print("\n=== Search With Output Fields ===")
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    limit=3,
    search_params={"metric_type": "L2"},
    output_fields=["color", "text"]  # Add text field to output
)

print("Search results with output fields:")
for hits in results:
    for hit in hits:
        print(f"ID: {hit['id']}, Distance: {hit['distance']}, Color: {hit['entity']['color']}, Text: {hit['entity']['text']}")

# 10. Text match search example
print("\n=== Text Match Search ===")
filter = "TEXT_MATCH(text, 'text_1 text_2')"  # Search documents containing text_1 or text_2
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    filter=filter,
    limit=3,
    search_params={"metric_type": "L2"},
    output_fields=["text"]
)

print("Text match search results:")
for hits in results:
    for hit in hits:
        print(f"ID: {hit['id']}, Distance: {hit['distance']}, Text: {hit['entity']['text']}")

# 11. Cleanup
client.release_collection(collection_name=COLLECTION_NAME)
