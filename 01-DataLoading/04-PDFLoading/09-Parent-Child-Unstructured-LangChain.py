file_path = '90-Data/shanxi-tourism/YunGangGrottoes-en.pdf'
from langchain_unstructured import UnstructuredLoader
loader = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res",
    # partition_via_api=True, # Call Unstructured via API
    # coordinates=True, # Return element position coordinates
)
docs = []
for doc in loader.lazy_load():
    docs.append(doc)


# Filter Docs from the first page only
page_number = 1
page_docs = [doc for doc in docs if doc.metadata.get("page_number") == page_number]

# Iterate and print detailed information for each Doc
for i, doc in enumerate(page_docs, 1):
    print(f"Doc {i}:")
    print(f"  Content: {doc.page_content}")
    print(f"  Category: {doc.metadata.get('category')}")
    print(f"  ID: {doc.metadata.get('element_id')}")
    print(f"  Parent ID: {doc.metadata.get('parent_id')}")
    # print(f"  Position: {doc.metadata.get('position')}")
    # print(f"  Coordinates: {doc.metadata.get('coordinates')}")
    print("="*50)

# Filter Titles from the first page only
page_number = 1
title_dict = {}

# Collect Titles and build a mapping from parent_id -> Title
for doc in docs:
    if doc.metadata.get("category") == "Title" and doc.metadata.get("page_number") == page_number:
        title_id = doc.metadata.get("element_id")  # Unique ID of the Title
        title_text = doc.page_content.strip()  # Strip whitespace
        # Avoid adding duplicate titles
        if title_text not in [data["title"] for data in title_dict.values()]:
            title_dict[title_id] = {"title": title_text, "content": []}

# Associate Titles with their corresponding Text
for doc in docs:
    if doc.metadata.get("category") in ["NarrativeText", "Text"] and doc.metadata.get("page_number") == page_number:
        parent_id = doc.metadata.get("parent_id")
        if parent_id in title_dict:
            content = doc.page_content.strip()  # Strip whitespace
            if content:  # Only add non-empty content
                title_dict[parent_id]["content"].append(content)

# Command-line output
for title_data in title_dict.values():
    if title_data["content"]:  # Only output titles that have content
        print("\n=== " + title_data["title"] + " ===")
        for content in title_data["content"]:
            print(content)
        print()
