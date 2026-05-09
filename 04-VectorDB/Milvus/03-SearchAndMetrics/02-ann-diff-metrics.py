from pymilvus import MilvusClient, DataType
import random
import numpy as np

'''
L2 is suitable for continuous data
IP is suitable for non-normalized data
COSINE is suitable for directional similarity comparison
'''

# 1. Set up Milvus client
client = MilvusClient(uri="http://localhost:19530")

# Define metric types and corresponding collection names
metric_types = ["L2", "IP", "COSINE"]
collections = {metric: f"ann_search_demo_{metric.lower()}" for metric in metric_types}

# 2. Create data
def create_data(num_vectors=1000, dim=128):
    vectors = [[random.random() for _ in range(dim)] for _ in range(num_vectors)]
    ids = list(range(num_vectors))
    colors = [f"color_{random.randint(1, 1000)}" for _ in range(num_vectors)]
    return vectors, ids, colors

vectors, ids, colors = create_data()

# 3. Create collections and indexes for each metric type
def create_collection_with_metric(collection_name, metric_type):
    # Drop collection if it already exists
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create schema
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=128)
    schema.add_field(field_name="color", datatype=DataType.VARCHAR, max_length=100)

    # Create collection
    client.create_collection(collection_name=collection_name, schema=schema)

    # Insert data
    entities = [{"id": ids[i], "vector": vectors[i], "color": colors[i]} for i in range(len(ids))]
    client.insert(collection_name=collection_name, data=entities)

    # Create index
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type=metric_type,
        index_type="FLAT",
        index_name="vector_index",
        params={}
    )
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
        sync=True
    )

    # Load collection
    client.load_collection(collection_name=collection_name)

# Create collections for each metric type
for metric_type, collection_name in collections.items():
    print(f"\nCreating collection for {metric_type} metric type...")
    create_collection_with_metric(collection_name, metric_type)

# 4. Generate query vectors
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

query_vector = [random.random() for _ in range(128)]
normalized_query_vector = normalize_vector(query_vector)

# 5. Search using different metric types
print("\n=== Search Comparison Across Different Metric Types ===")
for metric_type, collection_name in collections.items():
    print(f"\nSearching with {metric_type} metric type:")
    search_vector = normalized_query_vector if metric_type == "COSINE" else query_vector

    results = client.search(
        collection_name=collection_name,
        data=[search_vector],
        anns_field="vector",
        limit=3,
        search_params={"metric_type": metric_type},
        output_fields=["color"]
    )

    print(f"Search results (metric type: {metric_type}):")
    for hits in results:
        for hit in hits:
            print(f"ID: {hit['id']}, Distance: {hit['distance']}, Color: {hit['entity']['color']}")

# 6. Batch vector search example
print("\n=== Batch Vector Search (Different Metric Types) ===")
query_vectors = [[random.random() for _ in range(128)] for _ in range(2)]
normalized_query_vectors = [normalize_vector(v) for v in query_vectors]

for metric_type, collection_name in collections.items():
    print(f"\nBatch search using {metric_type} metric type:")
    search_vectors = normalized_query_vectors if metric_type == "COSINE" else query_vectors

    results = client.search(
        collection_name=collection_name,
        data=search_vectors,
        anns_field="vector",
        limit=3,
        search_params={"metric_type": metric_type}
    )

    print(f"Batch search results (metric type: {metric_type}):")
    for i, hits in enumerate(results):
        print(f"\nResults for query vector {i+1}:")
        for hit in hits:
            print(f"ID: {hit['id']}, Distance: {hit['distance']}")

# 7. Cleanup
for collection_name in collections.values():
    client.release_collection(collection_name=collection_name)
