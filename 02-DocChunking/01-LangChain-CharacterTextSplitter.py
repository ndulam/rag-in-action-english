from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
loader = TextLoader("90-Data/shanxi-tourism/YunGangGrottoes.txt")
documents = loader.load()
# Set up the chunker with a chunk size of 100 characters and 10-character overlap
text_splitter = CharacterTextSplitter(
    chunk_size=100,  # Each text chunk is 100 characters
    chunk_overlap=10,  # 10-character overlap between chunks
)
chunks = text_splitter.split_documents(documents)
print("\n=== Document Chunking Results ===")
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- Chunk {i} ---")
    print(f"Content: {chunk.page_content}")
    print(f"Metadata: {chunk.metadata}")
    print("-" * 50)
