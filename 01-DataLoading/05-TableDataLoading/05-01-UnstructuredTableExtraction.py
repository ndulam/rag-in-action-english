"""
PDF table extraction using the unstructured library

[System dependency installation]
Before running this script, install the following system dependencies:

1. Install poppler-utils (for PDF processing):
   sudo apt update
   sudo apt install -y poppler-utils

2. Install Tesseract OCR (for text recognition):
   sudo apt install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng

[Common error solutions]
- Error: "PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH?"
  Solution: Install poppler-utils

- Error: "TesseractNotFoundError: tesseract is not installed or it's not in your PATH"
  Solution: Install tesseract-ocr

- Error: "No such file or directory" or path display issues
  Solution: Make sure to run the script from the project root directory, or use absolute paths

[Verify installation]
Use the following commands to verify the installation:
- pdfinfo -v
- tesseract --version

[VSCode usage notes]
- Make sure the VSCode terminal working directory is at the project root
- If you encounter path encoding issues, it is recommended to run the script in the VSCode terminal
- It is recommended to set the Python interpreter to the correct virtual environment in VSCode

[Python dependencies]
91-Environment/requirements_llamaindex_Ubuntu-with-CPU.txt

"""

import os
import sys
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

# Ensure the working directory is correct
# Get the parent directory of the script (the project root directory)
script_dir = Path(__file__).parent.parent.parent
if script_dir.exists():
    os.chdir(script_dir)
    print(f"Working directory set to: {os.getcwd()}")

# Import LlamaIndex-related modules
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Global settings
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Parse the PDF structure and extract text and tables
# Use a relative path starting from the project root directory
file_path = "90-Data/complex-pdf/billionaires_page-1-5.pdf"

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found - {file_path}")
    print(f"Current working directory: {os.getcwd()}")
    print("Please ensure:")
    print("1. Run the script from the project root directory")
    print("2. The PDF file path is correct")
    sys.exit(1)

print(f"Processing file: {file_path}")

elements = partition_pdf(
    file_path,
    strategy="hi_res",  # Use high-accuracy strategy
)  # Parse the PDF document

# Create a mapping from element ID to element
element_map = {element.id: element for element in elements if hasattr(element, 'id')}

for element in elements:
    if element.category == "Table": # Only print table data
        print("\nTable data:")
        print("Table metadata:", vars(element.metadata))  # Use vars() to display all metadata attributes
        print("Table content:")
        print(element.text)  # Print table text content

        # Get and print parent node information
        parent_id = getattr(element.metadata, 'parent_id', None)
        if parent_id and parent_id in element_map:
            parent_element = element_map[parent_id]
            print("\nParent node information:")
            print(f"Type: {parent_element.category}")
            print(f"Content: {parent_element.text}")
            if hasattr(parent_element, 'metadata'):
                print(f"Parent node metadata: {vars(parent_element.metadata)}")  # Also use vars() to display all metadata
        else:
            print(f"Parent node not found (ID: {parent_id})")
        print("-" * 50)

text_elements = [el for el in elements if el.category == "Text"]
table_elements = [el for el in elements if el.category == "Table"]


