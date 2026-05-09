"""
Q: When using unstructured to parse PPT or other Office files, what should I do if I get a FileNotFoundError: soffice command was not found error?
A: This is because LibreOffice is missing from the system. unstructured needs to call LibreOffice's soffice command to process Office documents.
The solution is to install LibreOffice on your system.
On Debian/Ubuntu systems, you can install it using the following command:
sudo apt-get update && sudo apt-get install -y libreoffice

- Install instructions: https://www.libreoffice.org/get-help/install-howto/
- Mac: https://formulae.brew.sh/cask/libreoffice
- Debian: https://wiki.debian.org/LibreOffice
"""
from unstructured.partition.ppt import partition_ppt
# Parse the PPT file
ppt_elements = partition_ppt(filename="90-Data/black-myth-wukong/black-myth-wukong.pptx")
print("PPT content:")
# for element in ppt_elements:
#     print(element.text)

from langchain_core.documents import Document
# Convert to Document data structure
documents = [
Document(page_content=element.text,
         metadata={"source": "data/black-myth-wukong-PPT.pptx"})
    for element in ppt_elements
]

# Output the converted Documents
print(documents[0:3])


