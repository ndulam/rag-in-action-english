# Load a single txt file
import os
from langchain_community.document_loaders import TextLoader
# Get the directory of the current script file
script_dir = os.path.dirname(__file__)
print(f"Current script directory: {script_dir}")
# Build the full path using the relative path
file_dir = os.path.join(script_dir, '../../90-Data/black-myth-wukong/settings.txt')

loader = TextLoader(file_dir)
documents = loader.load()
print(documents)
