# Import required libraries
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
load_dotenv()

# Get Cohere API key
# URL: https://dashboard.cohere.com/api-keys
# If CO_API_KEY is not set in the env file, you can also use the following method:
# import os
# api_key = 'XXXX'
# os.environ['CO_API_KEY'] = api_key

documents = [
    Document(
        page_content="Mount Wutai is one of the four major Buddhist mountains in China, famous as the sacred site of Manjushri Bodhisattva.",
        metadata={"source": "Shanxi Tourism Guide"}
    ),
    Document(
        page_content="Yungang Grottoes is one of the three major stone grottoes in China, renowned for its exquisite Buddhist sculptures.",
        metadata={"source": "Shanxi Tourism Guide"}
    ),
    Document(
        page_content="Pingyao Ancient City is one of the most intact ancient county cities in China, listed as a World Cultural Heritage site.",
        metadata={"source": "Shanxi Tourism Guide"}
    )
]
# Create BM25 retriever
retriever = BM25Retriever.from_documents(documents)
retriever.k = 3  # Set to return top 3 results
# Set up Cohere reranker
# Model URL: https://huggingface.co/Cohere/rerank-multilingual-v3.0
compressor = CohereRerank(model="rerank-multilingual-v3.0")
# Create ContextualCompressionRetriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
# Execute query, reranking, and compression
query = "What are the famous tourist attractions in Shanxi?"
compressed_docs = compression_retriever.invoke(query)
# Output compression results
print(f"Query: {query}\n")
print("Results after reranking and compression:")
for i, doc in enumerate(compressed_docs, 1):
    print(f"{i}. {doc.page_content}")
