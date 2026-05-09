from langchain_community.document_loaders import DirectoryLoader, TextLoader
# Load all files in the directory, skipping files that cause errors (e.g., image files that TextLoader cannot load)
import os
# Get the directory of the current script file
script_dir = os.path.dirname(__file__)
print(f"Current script directory: {script_dir}")
# Build the full path using the relative path
data_dir = os.path.join(script_dir, '../../90-Data/black-myth-wukong')

# Load all Markdown files in the directory
loader = DirectoryLoader(data_dir,
                          silent_errors=True,
                         loader_cls=TextLoader
                         )

docs = loader.load()
print(docs[0].page_content[:100])  # Print the first 100 characters of the first document
