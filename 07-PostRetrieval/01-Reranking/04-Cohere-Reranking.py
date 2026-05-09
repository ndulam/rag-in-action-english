# Import required libraries
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
load_dotenv()

"""
Cohere Reranking Algorithm Implementation

Cohere Rerank is a commercial-grade reranking API service provided by Cohere, based on advanced language model technology.

Core characteristics:
1. Enterprise-grade performance: based on large-scale pre-trained models with powerful semantic understanding capabilities
2. Multi-language support: supports reranking tasks in multiple languages including Chinese
3. Ready to use: no local model deployment required, accessible via API calls
4. Continuous optimization: model continuously updated with improving performance

Technical advantages:
- High precision: based on advanced Transformer architecture and large-scale training data
- Low latency: optimized inference engine, supporting real-time reranking needs
- Easy integration: standard REST API interface, easy to integrate into existing systems
- Scalable: supports parallel reranking of large document batches

Applicable scenarios:
- Commercial-grade search systems
- Applications with high precision requirements
- Rapid prototype development and testing
- Multi-language retrieval systems

Cost considerations:
- Billed by number of API calls
- Suitable for small to medium-scale applications or cost-insensitive scenarios
- Recommend cost assessment during development phase
"""

print("Initializing Cohere reranking service...")

# 1. API key configuration
print("Configuring Cohere API key...")
print("API key URL: https://dashboard.cohere.com/api-keys")

# Get Cohere API key - two configuration methods
import os

# Method 1: Read from environment variable (recommended)
api_key_from_env = os.getenv('CO_API_KEY')

# Method 2: Set directly (for testing only, use environment variables in production)
api_key = 'XXXX'  # Please replace with your actual API key
os.environ['CO_API_KEY'] = api_key

if api_key_from_env:
    print("Successfully read API key from environment variable")
else:
    print("Warning: Using hardcoded API key (please use environment variables in production)")

print("Security reminder: please ensure the security of your API key, do not commit it to code repositories")

# 2. Prepare example documents
print("\nPreparing test documents...")
documents = [
    Document(
        page_content="Mount Wutai is one of the four major Buddhist mountains in China, famous as the sacred site of Manjushri Bodhisattva.",
        metadata={"source": "Shanxi Tourism Guide", "category": "Buddhist culture", "location": "Xinzhou"}
    ),
    Document(
        page_content="Yungang Grottoes is one of the three major stone grottoes in China, renowned for its exquisite Buddhist sculptures.",
        metadata={"source": "Shanxi Tourism Guide", "category": "Grotto art", "location": "Datong"}
    ),
    Document(
        page_content="Pingyao Ancient City is one of the most intact ancient county cities in China, listed as a World Cultural Heritage site.",
        metadata={"source": "Shanxi Tourism Guide", "category": "Ancient architecture", "location": "Jinzhong"}
    )
]

print(f"Number of documents: {len(documents)}")
for i, doc in enumerate(documents, 1):
    print(f"  Document {i}:")
    print(f"    Content: {doc.page_content}")
    print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"    Category: {doc.metadata.get('category', 'Unknown')}")
    print(f"    Location: {doc.metadata.get('location', 'Unknown')}")

# 3. Create BM25 retriever (as initial retrieval)
print(f"\nCreating BM25 initial retriever...")
print("  BM25 is used for the first stage retrieval to provide candidate document sets")
retriever = BM25Retriever.from_documents(documents)
retriever.k = 3  # Set to return top 3 results
print(f"BM25 retriever configured, returning Top-{retriever.k} results")

# 4. Set up Cohere reranker
print(f"\nConfiguring Cohere reranker...")
reranker = CohereRerank(
    model="rerank-multilingual-v3.0"  # Multi-language reranking model, supports Chinese
)
print(f"Using model: rerank-multilingual-v3.0")
print("  Model characteristics:")
print("  - Supports multiple languages (including Chinese)")
print("  - Based on advanced Transformer architecture")
print("  - Specifically optimized for reranking tasks")
print("  - Continuously updated and improved")

# 5. Execute query and reranking
print(f"\nStarting query and reranking execution...")
query = "What are the famous tourist attractions in Shanxi?"
print(f"Query: {query}")

print(f"\nPhase 1 - BM25 initial retrieval:")
print("  Using BM25 algorithm for initial retrieval...")
initial_docs = retriever.invoke(query)
print(f"  BM25 retrieved {len(initial_docs)} candidate documents")

for i, doc in enumerate(initial_docs, 1):
    print(f"    {i}. {doc.page_content}")

print(f"\nPhase 2 - Cohere reranking:")
print("  Calling Cohere API for semantic reranking...")
print("  Processing (may take a few seconds)...")

try:
    # Use Cohere reranker to rerank BM25 results
    reranked_docs = reranker.compress_documents(
        documents=initial_docs,
        query=query
    )
    print("  Cohere reranking complete")

    # 6. Output reranking results
    print(f"\n{'='*60}")
    print(f"Cohere Reranking Final Results")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"\nReranked results (in descending order of relevance):")

    for i, doc in enumerate(reranked_docs, 1):
        print(f"\nRank {i}:")
        print(f"   Document content: {doc.page_content}")

        # Show document metadata
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"   Document source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Attraction category: {doc.metadata.get('category', 'Unknown')}")
            print(f"   Location: {doc.metadata.get('location', 'Unknown')}")

        # Show reranking score if available
        if hasattr(doc, 'score'):
            print(f"   Reranking score: {doc.score:.4f}")

except Exception as e:
    print(f"  Cohere API call failed: {str(e)}")
    print("  Possible causes:")
    print("    - API key invalid or expired")
    print("    - Network connection issue")
    print("    - API quota exhausted")
    print("    - Please check API key configuration and network status")

print(f"\nCohere Reranking Summary:")
print("- Enterprise-grade performance: based on large-scale pre-trained models")
print("- Multi-language support: native support for Chinese and other languages")
print("- Ready to use: no local deployment required, API call ready")
print("- Continuous optimization: model regularly updated with improving performance")
print("- Cost consideration: billed by number of API calls")
print("- Security reminder: protect your API key")
print("- Applicable scenarios: commercial search, rapid prototyping, multi-language applications")
