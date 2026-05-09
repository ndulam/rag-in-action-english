# Import the partition function from unstructured for PDF parsing
from unstructured.partition.auto import partition

# Set the PDF file path
filename = "90-Data/black-myth-wukong/black-myth-wukong.pdf"

# Use the partition function to parse the PDF file
# content_type specifies the file type as PDF
elements = partition(filename=filename,
                   content_type="application/pdf"
                  )

# Display the types and content of the parsed elements
print("Element types after PDF parsing:")
for i, element in enumerate(elements[:5]):
    print(f"\nElement {i+1}:")
    print(f"Type: {type(element).__name__}")
    print(f"Content: {str(element)}")
    print("-" * 50)

# Count the number of each element type
element_types = {}
for element in elements:
    element_type = type(element).__name__
    element_types[element_type] = element_types.get(element_type, 0) + 1

print("\nElement type statistics:")
for element_type, count in element_types.items():
    print(f"{element_type}: {count} elements")
