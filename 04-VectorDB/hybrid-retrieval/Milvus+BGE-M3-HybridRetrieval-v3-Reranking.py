# 1. Load and preprocess dataset
import json
from typing import Optional, Dict
with open("90-Data/chronicles-of-gods/combat-scenarios.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)

docs = []
metadata = []

for item in dataset['data']:
    text_parts = [item['title'], item['description']]

    if 'combat_details' in item:
        text_parts.extend(item['combat_details'].get('combat_style', []))
        text_parts.extend(item['combat_details'].get('abilities_used', []))

    if 'scene_info' in item:
        text_parts.extend([
            item['scene_info'].get('location', ''),
            item['scene_info'].get('environment', ''),
            item['scene_info'].get('time_of_day', '')
        ])

    docs.append(' '.join(filter(None, text_parts)))
    metadata.append(item)

print(f"Loaded {len(docs)} records")

# 2. Import and use BGE-M3 to generate embedding vectors
from milvus_model.hybrid import BGEM3EmbeddingFunction

ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
print("Starting vector embedding generation...")
docs_embeddings = ef(docs)
print(f"Vector generation complete, dense vector dimensions: {ef.dim['dense']}")

# 3. Import Milvus and connect to service
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)

collection_name = "wukong_hybrid"
connections.connect(uri="./wukong.db")

# 4. Create Milvus collection and indexes
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])
]

schema = CollectionSchema(fields)

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema, consistency_level="Strong")

collection.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
collection.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})

collection.load()

# 5. Insert data into the collection
batch_size = 50
for i in range(0, len(docs), batch_size):
    end_idx = min(i + batch_size, len(docs))
    batch_data = []

    for j in range(i, end_idx):
        item = metadata[j]
        batch_data.append({
            "text": docs[j],
            "id": item["id"],
            "title": item["title"],
            "category": item["category"],
            "location": item.get("scene_info", {}).get("location", ""),
            "environment": item.get("scene_info", {}).get("environment", ""),
            "sparse_vector": docs_embeddings["sparse"]._getrow(j),
            "dense_vector": docs_embeddings["dense"][j]
        })

    collection.insert(batch_data)

print(f"Data insertion complete, total: {collection.num_entities}")

# 6. Define and execute hybrid search
from pymilvus import AnnSearchRequest, WeightedRanker, RRFRanker

query = "Combat scene in the snow"
category = "combat"
environment = "Snow"
limit = 5
search_type = "hybrid"
rerank_method = "rrf"  # Options: 'weighted' or 'rrf'
weights = {"sparse": 0.7, "dense": 1.0}
rrf_k = 60  # RRF parameter

query_embeddings = ef([query])

# Print detailed query vector information
print("\nQuery vector information:")
print(f"Dense vector dimensions: {len(query_embeddings['dense'][0])}")
sparse_row = query_embeddings['sparse']._getrow(0)
print(f"Number of non-zero elements in sparse vector: {sparse_row.nnz}")
print(f"Sample non-zero elements in sparse vector: {dict(zip(sparse_row.indices[:5], sparse_row.data[:5]))}")

# Build filter expression
expr = None
conditions = []
if category:
    conditions.append(f'category == "{category}"')
if environment:
    conditions.append(f'environment == "{environment}"')
if conditions:
    expr = " && ".join(conditions)

search_params = {
    "metric_type": "IP",
    "params": {}
}
if expr:
    search_params["expr"] = expr

if search_type == "hybrid":
    dense_req = AnnSearchRequest(
        data=[query_embeddings["dense"][0]],
        anns_field="dense_vector",
        param=search_params,
        limit=limit
    )
    sparse_req = AnnSearchRequest(
        data=[query_embeddings["sparse"]._getrow(0)],
        anns_field="sparse_vector",
        param=search_params,
        limit=limit
    )

    # Create different rerankers based on the selected reranking method
    if rerank_method == "weighted":
        rerank = WeightedRanker(weights["sparse"], weights["dense"])
        print(f"\nUsing weighted reranking, weights: sparse={weights['sparse']}, dense={weights['dense']}")
    else:  # rrf
        rerank = RRFRanker(rrf_k)
        print(f"\nUsing RRF reranking, k={rrf_k}")

    # Execute separate sparse and dense searches before the hybrid search
    print("\nExecuting individual searches:")
    dense_results = collection.search(
        data=[query_embeddings["dense"][0]],
        anns_field="dense_vector",
        param=search_params,
        limit=limit,
        output_fields=["text", "id", "title", "category", "location", "environment"]
    )[0]

    sparse_results = collection.search(
        data=[query_embeddings["sparse"]._getrow(0)],
        anns_field="sparse_vector",
        param=search_params,
        limit=limit,
        output_fields=["text", "id", "title", "category", "location", "environment"]
    )[0]

    print("\nDense vector search results:")
    for i, hit in enumerate(dense_results):
        print(f"{i+1}. {hit.entity.title} (score: {hit.distance:.4f})")

    print("\nSparse vector search results:")
    for i, hit in enumerate(sparse_results):
        print(f"{i+1}. {hit.entity.title} (score: {hit.distance:.4f})")

    # Execute hybrid search
    results = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=rerank,
        limit=limit,
        output_fields=["text", "id", "title", "category", "location", "environment"]
    )[0]
else:
    field = "dense_vector" if search_type == "dense" else "sparse_vector"
    vec = query_embeddings["dense"][0] if search_type == "dense" else query_embeddings["sparse"]._getrow(0)
    results = collection.search(
        data=[vec],
        anns_field=field,
        param=search_params,
        limit=limit,
        output_fields=["text", "id", "title", "category", "location", "environment"]
    )[0]

# 7. Display search results
print(f"\nQuery: {query}")
print("\nHybrid search results:")
for i, hit in enumerate(results):
    print(f"\n{i+1}. {hit.entity.title}")
    print(f"ID: {hit.entity.id}")
    print(f"Category: {hit.entity.category}")
    print(f"Location: {hit.entity.location}")
    print(f"Environment: {hit.entity.environment}")
    print(f"Final similarity score: {hit.distance:.4f}")
    print(f"Text: {hit.entity.text[:200]}...")
