from pymilvus import MilvusClient, DataType, Function, FunctionType
import json

# 1. Set up Milvus client
client = MilvusClient(uri="http://localhost:19530")
COLLECTION_NAME = "full_text_search_demo"

# Drop collection if it already exists
if client.has_collection(COLLECTION_NAME):
    print(f"Dropping existing collection: {COLLECTION_NAME}")
    client.drop_collection(COLLECTION_NAME)

# 2. Create schema
print("\nCreating schema...")
schema = client.create_schema()

# Add required fields
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000, enable_analyzer=True)
schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

# 3. Define BM25 function
print("Defining BM25 function...")
bm25_function = Function(
    name="text_bm25_emb",
    input_field_names=["text"],
    output_field_names=["sparse"],
    function_type=FunctionType.BM25,
)

# Add function to schema
schema.add_function(bm25_function)

# 4. Configure index parameters
print("Configuring index parameters...")
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="sparse",
    index_name="sparse_inverted_index",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
    params={
        "inverted_index_algo": "DAAT_MAXSCORE",
        "bm25_k1": 1.2,
        "bm25_b": 0.75
    },
)

# 5. Create collection
print(f"Creating collection: {COLLECTION_NAME}")
client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params
)

# 6. Insert text data
print("\nInserting text data...")
documents = [
    {'text': 'Information retrieval is a field of study.'},
    {'text': 'Information retrieval focuses on finding relevant information in large datasets.'},
    {'text': 'Data mining and information retrieval overlap in research.'},
    {'text': 'Search engines are a typical example of information retrieval systems.'},
    {'text': 'Natural language processing plays an important role in information retrieval.'},
]

insert_result = client.insert(COLLECTION_NAME, documents)
print(f"Insert result: {insert_result}")

# 7. Load collection
print("\nLoading collection...")
client.load_collection(collection_name=COLLECTION_NAME)

# 8. Execute full-text search
print("\n=== Full-Text Search Example ===")
search_params = {
    'params': {'drop_ratio_search': 0.2},
}

query_text = "information retrieval"
print(f"\nExecuting search, query text: {query_text}")
results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_text],
    anns_field='sparse',
    limit=3,
    search_params=search_params,
    output_fields=["text"]  # Add output field to show original text
)

print("\nSearch result structure:")
print(json.dumps(results, indent=2, ensure_ascii=False))

print("\nSearch results:")
if results and len(results) > 0:
    for hits in results:
        for hit in hits:
            # Print complete hit structure
            print("\nHit structure:")
            print(json.dumps(hit, indent=2, ensure_ascii=False))
            # Try different field access methods
            if 'entity' in hit:
                print(f"ID: {hit.get('id', 'N/A')}, Text: {hit['entity'].get('text', 'N/A')}")
            else:
                print("entity field not found")
else:
    print("No matching results found")

# 9. Cleanup
print("\nCleaning up resources...")
client.release_collection(collection_name=COLLECTION_NAME)
