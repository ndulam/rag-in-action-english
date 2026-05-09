import pdfplumber
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from typing import List

pdf_path = "90-Data/complex-pdf/billionaires_page-1-5.pdf"

# Open the PDF and parse tables
with pdfplumber.open(pdf_path) as pdf:
    tables = []
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            tables.append(table)

# Convert all tables to DataFrames and build documents
documents: List[Document] = []
if tables:
    # Iterate over all tables
    for i, table in enumerate(tables, 1):
        # Convert the table to a DataFrame
        df = pd.DataFrame(table)

        # Save to CSV file
        # csv_filename = f"billionaires_table_{i}.csv"
        # df.to_csv(csv_filename, index=False)
        # print(f"\nTable {i} data saved to {csv_filename}")

        # Convert DataFrame to text
        text = df.to_string()

        # Create a Document object
        doc = Document(text=text, metadata={"source": f"Table {i}"})
        documents.append(doc)

# Build the index
index = VectorStoreIndex.from_documents(documents)

# Create the query engine
query_engine = index.as_query_engine()

# Example Q&A
questions = [
    "Who is the richest person in 2023?",
    "Who is the youngest billionaire?"
]

print("\n===== Q&A Demo =====")
for question in questions:
    response = query_engine.query(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {response}")
