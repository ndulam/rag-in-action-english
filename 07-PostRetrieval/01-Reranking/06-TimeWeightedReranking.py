from datetime import datetime, timedelta
import faiss
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

"""
Time-Weighted Reranking Algorithm Implementation

TimeWeightedVectorStoreRetriever is a retriever that considers temporal factors,
using the time recency of documents as an important ranking factor.

Core principles:
1. Time decay: a document's relevance score decays over time
2. Access update: updates the "last access time" of a document each time it is accessed
3. Comprehensive scoring: combine semantic similarity and time freshness for comprehensive ranking

Mathematical model:
- final_score = semantic_score * time_decay_factor
- time_decay_factor = exp(-decay_rate * time_since_last_access)

Technical characteristics:
- Time-aware: prioritizes recently accessed or created documents
- Dynamic update: the time weight of documents is dynamically adjusted based on access patterns
- Decay control: adjustable decay rate to adapt to different application needs

Applicable scenarios:
- News retrieval: the latest news is more important
- Knowledge base maintenance: recently updated documents are more reliable
- Trend analysis: focus on the latest data and information
- Real-time systems: applications that need to consider information timeliness

Parameter description:
- decay_rate: decay rate, controls the influence of time on relevance
- k: number of documents to return
- last_accessed_at: the last access time of the document
"""

print("Initializing time-weighted reranking system...")

# 1. Configure embedding model
print("Configuring OpenAI embedding model...")
embeddings_model = OpenAIEmbeddings()
print("  Model: OpenAI Embeddings")
print("  Dimensions: 1536-dimensional vectors")
print("  Note: requires OPENAI_API_KEY environment variable to be configured")

# 2. Initialize FAISS vector store
print(f"\nInitializing vector storage system...")
print("  Creating FAISS index (L2 distance)...")
index = faiss.IndexFlatL2(1536)  # OpenAI embedding dimensions are 1536
print(f"    Index type: IndexFlatL2")
print(f"    Vector dimensions: 1536")

print("  Configuring document store...")
vectorstore = FAISS(
    embeddings_model,           # Embedding model
    index,                      # FAISS index
    InMemoryDocstore({}),       # In-memory document store
    {}                          # Mapping from index to document ID
)
print("  Vector storage system initialization complete")

# 3. Create time-weighted retriever
print(f"\nCreating time-weighted retriever...")
print("  Retriever configuration parameters:")
decay_rate = 0.5
k_value = 1
print(f"    - decay_rate: {decay_rate} (decay rate, higher value means stronger time influence)")
print(f"    - k: {k_value} (number of documents to return)")
print("  Decay mechanism description:")
print("    - Decay formula: score = semantic_score * exp(-decay_rate * hours_passed)")
print("    - Decay rate 0.5 means the document weight decays by about 39% per hour")

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=decay_rate,      # Decay rate: controls the strength of time's influence on relevance
    k=k_value                   # Number of documents to return
)
print("  Time-weighted retriever created successfully")

# 4. Prepare test documents
print(f"\nPreparing test documents...")

# First document: set to accessed yesterday
yesterday = datetime.now() - timedelta(days=1)
print(f"  Setting document 1 access time to yesterday: {yesterday.strftime('%Y-%m-%d %H:%M:%S')}")

print("  Adding first document (accessed yesterday)...")
doc1 = Document(
    page_content="hello world",
    metadata={"last_accessed_at": yesterday, "doc_id": "doc_1", "topic": "greeting"}
)
retriever.add_documents([doc1])
print(f"    Content: {doc1.page_content}")
print(f"    Access time: {yesterday.strftime('%Y-%m-%d %H:%M:%S')}")

# Second document: current time (default)
current_time = datetime.now()
print(f"\n  Second document will use current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

print("  Adding second document (current time)...")
doc2 = Document(
    page_content="hello foo",
    metadata={"doc_id": "doc_2", "topic": "greeting"}
)
retriever.add_documents([doc2])
print(f"    Content: {doc2.page_content}")
print("    Access time: current time (default)")

# 5. Execute retrieval and time-sensitivity analysis
print(f"\nExecuting time-weighted retrieval...")
query = "hello world"
print(f"Query: {query}")

print(f"\n  Time-weighted retrieval process:")
print("    1. Calculate semantic similarity between query and each document")
print("    2. Calculate time decay factor based on document's last access time")
print("    3. Combine semantic score and time factor to get final score")
print("    4. Sort documents by final score in descending order")

print(f"\n  Executing retrieval...")
results = retriever.get_relevant_documents(query)

# 6. Analyze and display results
print(f"\nTime-weighted retrieval results analysis:")
print(f"{'='*60}")
print(f"Retrieval Results")
print(f"{'='*60}")
print(f"Query: {query}")
print(f"Documents returned: {len(results)}")

for i, doc in enumerate(results, 1):
    print(f"\nRank {i}:")
    print(f"   Document content: {doc.page_content}")
    print(f"   Document ID: {doc.metadata.get('doc_id', 'Unknown')}")
    print(f"   Topic: {doc.metadata.get('topic', 'Unknown')}")

    # Analyze time-sensitivity impact
    if 'last_accessed_at' in doc.metadata:
        access_time = doc.metadata['last_accessed_at']
        time_diff = datetime.now() - access_time
        hours_passed = time_diff.total_seconds() / 3600
        decay_factor = 1.0 / (1.0 + decay_rate * hours_passed)  # Simplified decay calculation

        print(f"   Last accessed: {access_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Time interval: {time_diff}")
        print(f"   Hours elapsed: {hours_passed:.2f}")
        print(f"   Time decay factor: {decay_factor:.4f}")
    else:
        print(f"   Last accessed: current time (just added)")
        print(f"   Time decay factor: 1.0000 (no decay)")

print(f"\nResult interpretation:")
if len(results) > 0:
    first_doc = results[0]
    if "foo" in first_doc.page_content:
        print("  'hello foo' ranked first")
        print("  Reason: although the semantic similarity to query 'hello world' may be lower,")
        print("       since it is the most recently added document (high time weight), its total score is higher")
    else:
        print("  'hello world' ranked first")
        print("  Reason: high semantic similarity, sufficient to overcome the impact of time decay")

# 7. Time simulation experiment
print(f"\nTime simulation experiment...")
print("  Simulating retrieval results a few hours later...")

# Use mock to simulate future time
from langchain_core.utils import mock_now
import datetime as dt

# Simulate retrieval 8 hours later
future_time = dt.datetime(2028, 8, 8, 12, 0)  # Simulate future time
print(f"  Simulated time: {future_time.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"  Executing retrieval at simulated time...")
with mock_now(future_time):
    future_results = retriever.get_relevant_documents(query)

print(f"\nSimulated time retrieval results:")
print(f"Query: {query}")
print(f"Simulated time: {future_time.strftime('%Y-%m-%d %H:%M:%S')}")

for i, doc in enumerate(future_results, 1):
    print(f"\nRank {i} (simulated):")
    print(f"   Document content: {doc.page_content}")
    print(f"   Document ID: {doc.metadata.get('doc_id', 'Unknown')}")

    # Calculate decay at simulated time
    if 'last_accessed_at' in doc.metadata:
        access_time = doc.metadata['last_accessed_at']
        time_diff = future_time - access_time
        hours_passed = time_diff.total_seconds() / 3600
        print(f"   Simulated time interval: {time_diff}")
        print(f"   Simulated hours elapsed: {hours_passed:.2f}")
    else:
        # For documents added at current time, calculate interval from addition to simulated time
        time_diff = future_time - current_time
        hours_passed = time_diff.total_seconds() / 3600
        print(f"   Simulated time interval: {time_diff}")
        print(f"   Simulated hours elapsed: {hours_passed:.2f}")

print(f"\nTime-Weighted Reranking Summary:")
print("- Time awareness: prioritizes recently accessed or created documents")
print("- Dynamic weight: document importance dynamically adjusted over time")
print("- Adjustable control: control time influence strength through decay_rate parameter")
print("- Comprehensive scoring: balance semantic relevance and time freshness")
print("- Applicable scenarios: news retrieval, knowledge base maintenance, real-time systems")
print("- Parameter tuning: adjust decay rate and return count based on application scenario")
print("- Notes: need to properly set time metadata")
print("- Best practice: combine with other retrieval methods to form multi-stage retrieval pipeline")
