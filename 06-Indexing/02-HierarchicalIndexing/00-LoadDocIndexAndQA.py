# Import LlamaIndex related modules
from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Global settings
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ---------------------------
# 1. Parse PDF structure to extract text and tables
# ---------------------------
file_path = "90-Data/complex-pdf/billionaires_page-1-5.pdf"  # Modify to your file path

# Use PyMuPDFReader to load PDF
reader = PyMuPDFReader()
documents = reader.load(file_path)

# ---------------------------
# 2. Create vector index
# ---------------------------
index = VectorStoreIndex.from_documents(documents)

# ---------------------------
# 3. Create query engine
# ---------------------------
query_engine = index.as_query_engine(
    similarity_top_k=3,
    verbose=True
)

# ---------------------------
# 4. Test queries
# ---------------------------
query = "Who was the second richest billionaire in 2023?"
response = query_engine.query(query)
print("Query:", query)
print("Response:", response)

# Show retrieved text chunks
print("\nRetrieved Text Chunks:")
for i, source_node in enumerate(response.source_nodes):
    print(f"\nChunk {i+1}:")
    print("Text content:")
    print(source_node.text)
    print("-" * 50)

# Generate answer
response = query_engine.query(query)
print("Query:", query)
print("Response:", response)




