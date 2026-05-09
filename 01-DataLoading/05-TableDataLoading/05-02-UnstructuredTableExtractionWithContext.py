from unstructured.partition.pdf import partition_pdf
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Global settings
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Parse the PDF structure and extract text and tables
file_path = "90-Data/complex-pdf/billionaires_page-1-5.pdf"  # Modify to your file path

elements = partition_pdf(
    file_path,
    # strategy="hi_res",  # Use high-accuracy strategy
)  # Parse the PDF document

# Create a mapping from element ID to element
element_map = {element.id: element for element in elements if hasattr(element, 'id')}

# Create a mapping from element index to element
element_index_map = {i: element for i, element in enumerate(elements)}

for i, element in enumerate(elements):
    if element.category == "Table":
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
                print(f"Parent node metadata: {vars(parent_element.metadata)}")
        else:
            print(f"Parent node not found (ID: {parent_id})")

        # Print the content of the 3 nodes preceding the table
        print("\nContent of the 3 nodes preceding the table:")
        for j in range(max(0, i-3), i):
            prev_element = element_index_map.get(j)
            if prev_element:
                print(f"Node {j} ({prev_element.category}):")
                print(prev_element.text)

        print("-" * 50)

text_elements = [el for el in elements if el.category == "Text"]
table_elements = [el for el in elements if el.category == "Table"]


