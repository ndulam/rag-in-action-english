from unstructured.documents.elements import Title, NarrativeText, Text
from unstructured.partition.pdf import partition_pdf

file_path = '90-Data/shanxi-tourism/YunGangGrottoes-en.pdf'

# Read the PDF directly using unstructured
elements = partition_pdf(
    filename=file_path,
    strategy="hi_res",
    # include_metadata=True,  # If position information is needed
)

print(elements[0].to_dict())

# Add debug information to view the complete details of the first element
if elements:
    first_elem = elements[0]
    print("=== Detailed information of the first element ===")
    print(f"Type: {type(first_elem)}")
    print(f"Text: {first_elem.text}")
    print("Metadata attributes:")
    print(vars(first_elem.metadata))  # Print all metadata attributes
    print("All element attributes:")
    print(vars(first_elem))  # Print all element attributes
    print("="*50)

# Filter elements from the first page only
page_number = 1
page_elements = [elem for elem in elements if getattr(elem.metadata, "page_number", None) == page_number]

# Iterate and print detailed information for each element
for i, elem in enumerate(page_elements, 1):
    print(f"\nElement {i}:")
    print(f"  Content: {elem.text}")
    print(f"  Category: {type(elem).__name__}")
    print(f"  ID: {getattr(elem, '_element_id', None)}")
    print("="*50)

# Filter Titles from the first page only
title_dict = {}

# Collect Titles and build a mapping from parent_id -> Title
for elem in elements:
    if (isinstance(elem, Title) and
        getattr(elem.metadata, "page_number", None) == page_number):
        title_id = getattr(elem, '_element_id', None) # Get the element's ID
        title_text = elem.text.strip()
        if title_text not in [data["title"] for data in title_dict.values()]:
            title_dict[title_id] = {"title": title_text, "content": []}

# Associate Titles with their corresponding Text
for elem in elements:
    if (isinstance(elem, (NarrativeText, Text)) and
        getattr(elem.metadata, "page_number", None) == page_number):
        parent_id = getattr(elem.metadata, "parent_id", None)
        if parent_id in title_dict:
            content = elem.text.strip()
            if content:  # Only add non-empty content
                title_dict[parent_id]["content"].append(content)

# Command-line output
for title_data in title_dict.values():
    if title_data["content"]:  # Only output titles that have content
        print("\n=== " + title_data["title"] + " ===")
        for content in title_data["content"]:
            print(content)
        print()
