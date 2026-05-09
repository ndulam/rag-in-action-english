file_path = ("90-Data/shanxi-tourism/YunGangGrottoes-en.pdf")
from langchain_unstructured import UnstructuredLoader
loader = UnstructuredLoader(
    file_path=file_path,  # PDF file path
    strategy="hi_res",    # Use high-resolution strategy for document processing
    # partition_via_api=True,  # Partition the document via API
    # coordinates=True,     # Extract text coordinate information
)
docs = []

# lazy_load() is a lazy loading method
# It does not load all documents into memory at once; instead it loads them one by one as needed.
# This saves memory when processing large PDF files.
for doc in loader.lazy_load():
    docs.append(doc)

print(docs)
