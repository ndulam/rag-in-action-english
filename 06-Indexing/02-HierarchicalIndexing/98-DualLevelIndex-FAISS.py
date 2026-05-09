import os
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# Load environment variables
load_dotenv()

# 1. Prepare table description data
table_descriptions = [
    "The 2023 world top 10 billionaires list, showing the ten wealthiest people globally that year and their wealth status.",
    "The 2022 world top 10 billionaires list, recording the ten wealthiest people globally that year and their wealth status.",
    "The 2021 world top 10 billionaires list, showing the ten wealthiest people globally that year and their wealth status.",
    "The 2020 world top 10 billionaires list, recording the ten wealthiest people globally that year and their wealth status.",
    "The 2019 world top 10 billionaires list, showing the ten wealthiest people globally that year and their wealth status."
]

# 2. Set up the first-tier embedding model (for matching years)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
desc_embeddings = model.encode(table_descriptions)

# 3. Create first-tier vector store
dimension = desc_embeddings.shape[1]
desc_index = faiss.IndexFlatL2(dimension)
desc_index.add(desc_embeddings.astype('float32'))

# 4. Load Excel file and prepare second-tier data
excel_file = "90-Data/complex-pdf/top-10-billionaires/WorldTop10Billionaires.xlsx"
all_tables_data = {}

# Read all sheets from the Excel file
with pd.ExcelFile(excel_file) as xls:
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Convert DataFrame to text format
        table_text = df.to_string(index=False)
        all_tables_data[sheet_name] = table_text

# 5. Create second-tier vector store
table_embeddings = model.encode(list(all_tables_data.values()))
table_index = faiss.IndexFlatL2(dimension)
table_index.add(table_embeddings.astype('float32'))

def search_relevant_table(question):
    # First-tier retrieval: match year
    query_embedding = model.encode([question])[0]
    distances, indices = desc_index.search(
        np.array([query_embedding]).astype('float32'),
        k=1
    )
    matched_year = indices[0][0]

    # Second-tier retrieval: search for specific information in the matched year's table
    table_embedding = model.encode([all_tables_data[f"billionaires_table_{matched_year+2}"]])[0]
    distances, indices = table_index.search(
        np.array([table_embedding]).astype('float32'),
        k=1
    )

    return table_descriptions[matched_year], all_tables_data[f"billionaires_table_{matched_year+2}"]

def generate_answer(question):
    # Retrieve relevant information
    year_context, table_context = search_relevant_table(question)

    # Build prompt
    prompt = f"""Answer the question based on the following reference information:

Year information: {year_context}

Related data:
{table_context}

Question: {question}

Please provide a detailed answer based on the above information:"""

    # Generate answer using DeepSeek
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

# Test example
if __name__ == "__main__":
    test_question = "Who was the world's richest person in 2023? How much was their wealth?"
    answer = generate_answer(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
