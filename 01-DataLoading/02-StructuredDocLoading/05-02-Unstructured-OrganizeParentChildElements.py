from langchain_unstructured import UnstructuredLoader
from typing import List
from langchain_core.documents import Document
page_url = "https://en.wikipedia.org/wiki/Black_Myth:_Wukong"
def _get_setup_docs_from_url(url: str) -> List[Document]:
    loader = UnstructuredLoader(web_url=url)
    setup_docs = []
    # parent_id = None  # Initialize parent_id
    # current_parent = None  # Used to store the current parent element
    for doc in loader.load():
        # Check whether it is a Title or Table
        if doc.metadata["category"] == "Title" or doc.metadata["category"] == "Table":
            parent_id = doc.metadata["element_id"]
            current_parent = doc  # Update the current parent element
            setup_docs.append(doc)
        elif doc.metadata.get("parent_id") == parent_id:
            setup_docs.append((current_parent, doc))  # Store parent and child elements together
    return setup_docs
docs = _get_setup_docs_from_url(page_url)
for item in docs:
    if isinstance(item, tuple):
        parent, child = item
        print(f'Parent - {parent.metadata["category"]}: {parent.page_content}')
        print(f'Child - {child.metadata["category"]}: {child.page_content}')
    else:
        print(f'{item.metadata["category"]}: {item.page_content}')
    print("-" * 80)
