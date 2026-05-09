from langchain_community.document_loaders import DirectoryLoader

import os
# Get the directory of the current script file
script_dir = os.path.dirname(__file__)
print(f"Current script directory: {script_dir}")
# Build the full path using the relative path
data_dir = os.path.join(script_dir, '../../90-Data/black-myth-wukong')

loader = DirectoryLoader(data_dir,
                         glob="**/*.md",
                         use_multithreading=True,
                         show_progress=True,
                         )
docs = loader.load()
print(f"Number of documents: {len(docs)}")  # Print total number of documents
print(docs[0])  # Print the first document
