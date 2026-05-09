from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import CSVLoader

loader = DirectoryLoader(
    path="data/black-myth-wukong",  # Specify the directory containing your CSV files
    glob="**/*.csv",                # Use a glob pattern to match CSV files
    loader_cls=CSVLoader            # Specify CSVLoader as the loader class
)

docs = loader.load()
print(f"Number of documents: {len(docs)}")  # Print total number of documents
print(docs[0])
