file_path = ("90-Data/shanxi-tourism/YunGangGrottoes-en.pdf")
from langchain_unstructured import UnstructuredLoader
loader = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res",
    # partition_via_api=True,
    # coordinates=True,
)
docs = []
for doc in loader.lazy_load():
    docs.append(doc)

def extract_basic_structure(docs):
    """Basic structure extraction: organize content by document type"""
    # Define category mapping
    category_map = {
        'Title': 'title',
        'NarrativeText': 'text',
        'Image': 'image',
        'Table': 'table',
        'Footer': 'footer',
        'Header': 'header'
    }

    # Initialize the structure dictionary
    structure = {cat: [] for cat in category_map.values()}
    structure['metadata'] = [] # Extra metadata category

    # Iterate over documents and categorize
    for doc in docs:
        category = doc.metadata.get('category', 'Unknown')
        content = {
            'content': doc.page_content,
            'page': doc.metadata.get('page_number'),
            'coordinates': doc.metadata.get('coordinates')
        }

        target_category = category_map.get(category)
        if target_category:
            structure[target_category].append(content)

    return structure

# Example usage
structure = extract_basic_structure(docs)

# Output the titles on page 2
print("Page 2 titles:")
for title in [t for t in structure['title'] if t['page'] == 2]:
    print(f"- {title['content']}")


def analyze_layout(docs):
    """Analyze the page layout structure of the document"""
    layout_analysis = {}

    for doc in docs:
        page = doc.metadata.get('page_number')
        coords = doc.metadata.get('coordinates', {})

        # Initialize page information
        if page not in layout_analysis:
            layout_analysis[page] = {
                'elements': [],
                'dimensions': {
                    'width': coords.get('layout_width', 0),
                    'height': coords.get('layout_height', 0)
                }
            }

        # Get element position information
        points = coords.get('points', [])
        if points:
            # Only need top-left and bottom-right coordinate points
            (x1, y1), (_, _), (x2, y2), _ = points

            # Build element information
            element = {
                'type': doc.metadata.get('category'),
                'content': (doc.page_content[:50] + '...') if len(doc.page_content) > 50 else doc.page_content,
                'position': {
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
            }
            layout_analysis[page]['elements'].append(element)

    return layout_analysis

# Example usage
layout = analyze_layout(docs)

# Analyze the layout of the first page
print("Page 1 layout analysis:")
if 1 in layout:
    page = layout[1]
    print(f"Page dimensions: {page['dimensions']['width']} x {page['dimensions']['height']}")
    print("\nElement distribution:")

    # Sort and display elements by vertical position
    for elem in sorted(page['elements'], key=lambda x: x['position']['y1']):
        print(f"\nType: {elem['type']}")
        print(f"Position: ({elem['position']['x1']:.0f}, {elem['position']['y1']:.0f})")
        print(f"Dimensions: {elem['position']['width']:.0f} x {elem['position']['height']:.0f}")
        print(f"Content: {elem['content']}")

# Find parent-child relationships
cave6_docs = []
parent_id = -1
for doc in docs:
    if doc.metadata["category"] == "Title" and "Cave 6" in doc.page_content:
        parent_id = doc.metadata["element_id"]
    if doc.metadata.get("parent_id") == parent_id:
        cave6_docs.append(doc)

for doc in cave6_docs:
    print(doc.page_content)

external_docs = [] # Create a list to store child documents with external links
parent_id = -1 # Initialize parent ID to -1
for doc in docs:
    # Check if the document is a Title type and its content contains "External links"
    if doc.metadata["category"] == "Title" and "External links" in doc.page_content:
        parent_id = doc.metadata["element_id"]
        external_docs.append(doc)
    # Check if the document's parent_id matches the title ID we found
    if doc.metadata.get("parent_id") == parent_id:
        external_docs.append(doc) # Add all child documents belonging to this title to the result list
for doc in external_docs:
    print(doc.page_content)


# def find_tables_and_titles(docs):
#   results = []
#   for doc in docs:
#     # Check if the document is a Table type
#     if doc.metadata.get("category") == "Table":
#       table = doc
#       parent_id = doc.metadata.get("parent_id")
#       # Find the title document corresponding to the table (parent_id matches element_id)
#       title = next((doc for doc in docs if doc.metadata.get("element_id") == parent_id), None)
#       if title:
#         results.append({"table": table.page_content, "title": title.page_content})
#   return results

# results = find_tables_and_titles(cave6_docs)
# if results:
#   for result in results:
#     print("Found table and title:")
#     print(f"Title: {result['title']}")
#     print(f"Table: {result['table']}")
# else:
#   print("No tables and titles found")
