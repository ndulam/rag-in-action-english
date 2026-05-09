from llama_index.core import SimpleDirectoryReader
# Load files from a directory using SimpleDirectoryReader
dir_reader = SimpleDirectoryReader("90-Data/black-myth-wukong")
documents = dir_reader.load_data()
# Check the number of loaded documents and their content
print(f"Number of documents: {len(documents)}")
print(documents[0].text[:100])  # Print the first 100 characters of the first document

# Load only a specific file
dir_reader = SimpleDirectoryReader(input_files=["90-Data/black-myth-wukong/settings.txt"])
documents = dir_reader.load_data()
print(f"Number of documents: {len(documents)}")
print(documents[0].text[:100])  # Print the first 100 characters of the first document


