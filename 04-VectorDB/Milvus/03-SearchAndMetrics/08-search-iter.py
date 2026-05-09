from pymilvus import MilvusClient, DataType
import random

# 1. Set up Milvus client
client = MilvusClient(uri="http://localhost:19530")
COLLECTION_NAME = "search_iterator_demo"

# Drop collection if it already exists
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)

# 2. Create schema
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
schema.add_field(field_name="color", datatype=DataType.VARCHAR, max_length=100)

# 3. Create collection
client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

# 4. Insert random vector data
num_vectors = 20000  # Insert more data to demonstrate SearchIterator
vectors = [[random.random() for _ in range(128)] for _ in range(num_vectors)]
ids = list(range(num_vectors))
colors = [f"color_{random.randint(1, 1000)}" for _ in range(num_vectors)]
entities = [{"id": ids[i], "vector": vectors[i], "color": colors[i]} for i in range(num_vectors)]

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

# 7. Search using SearchIterator
print("\n=== Search Using SearchIterator ===")
query_vector = [random.random() for _ in range(128)]

# Create SearchIterator
iterator = client.search_iterator(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field="vector",
    search_params={"metric_type": "L2"},
    batch_size=1000,  # Return 1000 results per batch
    limit=20000,      # Return 20000 results total
    output_fields=["color"]
)

# Use iterator to retrieve results
all_results = []
while True:
    result = iterator.next()
    if not result:
        iterator.close()
        break

    # Convert results to dict and add to results list
    for hit in result:
        all_results.append(hit.to_dict())

print(f"Total results retrieved: {len(all_results)}")
print("\nFirst 5 results:")
for result in all_results[:5]:
    print(f"ID: {result['id']}, Distance: {result['distance']}, Color: {result['entity']['color']}")

# 8. Cleanup
client.release_collection(collection_name=COLLECTION_NAME)
