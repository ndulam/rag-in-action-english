# Load all documents in a directory using LangChain

"""
# Requires Tesseract OCR to be installed
### On Ubuntu, run the following commands:
sudo apt update
sudo apt install tesseract-ocr -y

### Notes

DirectoryLoader tries to find a suitable loader for each file type it encounters.
For complex formats like .pptx (PowerPoint), .pdf, and .jpg, DirectoryLoader typically
relies on the unstructured library for processing.
The unstructured library depends on the nltk (Natural Language Toolkit) library when
handling certain file types (especially those requiring text extraction and processing).
The nltk library needs to download additional data packages (such as tokenizers, POS taggers, etc.)
to work properly.

If you encounter the error zipfile.BadZipFile: File is not a zip file
This occurs inside nltk.data.py, which strongly suggests nltk is having trouble loading its data packages.
It tries to open a file as a zip archive to find data, but the file is not a valid zip file.


Hello!

Based on the error information and file list you provided, the issue arises when
langchain_community.document_loaders.DirectoryLoader tries to load .pptx files in the directory.

Root cause analysis:

DirectoryLoader tries to find a suitable loader for each file type it encounters.
For complex formats like .pptx (PowerPoint), .pdf, and .jpg, DirectoryLoader typically
relies on the unstructured library for processing.
The unstructured library depends on the nltk (Natural Language Toolkit) library when
handling certain file types (especially those requiring text extraction and processing).
The error zipfile.BadZipFile: File is not a zip file occurs inside nltk.data.py,
which strongly suggests nltk is having trouble loading its data packages. It tries to
open a file as a zip archive to find data, but the file is not a valid zip file.
Conclusion: The error is not because your .pptx file is corrupted, but because when
unstructured calls nltk, nltk cannot find or correctly load the required data packages,
causing the BadZipFile error. When only .csv files are present, DirectoryLoader may use
a simpler loader that does not depend on unstructured or nltk, hence no error.

# Solution:
# The most direct solution is to
# download the required nltk data packages. Run the following code once in your Python environment:
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
"""
import os
from langchain_community.document_loaders import DirectoryLoader

# Get the directory of the current script file
script_dir = os.path.dirname(__file__)
print(f"Current script directory: {script_dir}")
# Build the full path using the relative path
data_dir = os.path.join(script_dir, '../../90-Data/black-myth-wukong')

loader = DirectoryLoader(data_dir)
docs = loader.load()
print(f"Number of documents: {len(docs)}")  # Print total number of documents
print(docs[0])  # Print the first document
