import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
from pymilvus import MilvusClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize embedding model
embedding_function = SentenceTransformer(
    'BAAI/bge-m3',
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    trust_remote_code=True
)

# Connect to Milvus
client = MilvusClient("richman_bge_m3_v2.db")

def search_relevant_table(question):
    # First-tier retrieval: search the summary collection for the most relevant sheet
    query_embedding = embedding_function.encode([question])[0]

    summary_results = client.search(
        collection_name="billionaires_summary",
        data=[query_embedding.tolist()],
        limit=1,
        output_fields=["table_name"],
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
    )

    if not summary_results or not summary_results[0]:
        return None, None

    matched_table = summary_results[0][0]['entity']['table_name']

    # Second-tier retrieval: search the details collection for specific information
    details_results = client.search(
        collection_name="billionaires_details",
        data=[query_embedding.tolist()],
        filter=f"table_name == '{matched_table}'",
        limit=1,
        output_fields=["content"],
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
    )

    if not details_results or not details_results[0]:
        return None, None

    return matched_table, details_results[0][0]['entity']['content']

def generate_answer(question):
    # Retrieve relevant information
    table_name, content = search_relevant_table(question)

    if not table_name or not content:
        return "Sorry, no relevant information was found."

    # Build prompt
    prompt = f"""Answer the question based on the following table information:

Table name: {table_name}

Table content:
{content}

Question: {question}

Please provide a detailed answer based on the above information:"""

    # Generate answer using DeepSeek
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=1024
    )

    return response.choices[0].message.content

# Test query list
test_queries = [
    # Basic queries
    "Who was the world's richest person in 2023? How much was their wealth?",
    "Who was the world's richest person in 2020? How much was their wealth?",
    "How many of the top 10 richest people in the world in 2022 were from the United States?",
    "How many of the top 10 richest people in the world in 2021 were from China?",

    # Comparison queries
    "What was the wealth gap between the world's richest and second richest person in 2020?",
    "In 2019, how did the number of billionaires from the tech industry compare to those from the luxury goods industry in the top 10?",

    # Trend queries
    "What was the proportion of wealth held by tech industry billionaires among the top 10 in the world in 2019?",
    "Who was the oldest billionaire in the top 10 richest people in the world in 2022?",

    # Complex queries
    "What industries did the European billionaires in the top 10 richest people in the world in 2022 mainly work in?",
    "What was the average age of tech industry billionaires in the top 10 richest people in the world in 2021?"
]

# Run tests
if __name__ == "__main__":
    print("Starting dual-tier RAG system test...\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}/{len(test_queries)}")
        print(f"Question: {query}")
        print("-" * 50)

        answer = generate_answer(query)
        print(f"Answer: {answer}")
        print("-" * 50)
