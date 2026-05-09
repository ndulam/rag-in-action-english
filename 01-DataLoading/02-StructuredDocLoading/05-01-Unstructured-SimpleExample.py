# Load a webpage using UnstructuredLoader
from langchain_unstructured import UnstructuredLoader
page_url = "https://en.wikipedia.org/wiki/Black_Myth:_Wukong"
loader = UnstructuredLoader(web_url=page_url)
docs = loader.load()
for doc in docs[:5]:
    print(f'{doc.metadata["category"]}: {doc.page_content}')
