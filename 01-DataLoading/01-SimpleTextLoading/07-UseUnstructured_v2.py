from unstructured.partition.text import partition_text
text = "90-Data/black-myth-wukong/settings.txt"
elements = partition_text(text)
for element in elements:
    print(element)

# Use __dict__ to view all available metadata
for i, element in enumerate(elements):
    print(f"\n--- Element {i+1} ---")
    print(f"Type: {type(element)}")
    print(f"Element class: {element.__class__.__name__}")
    print(f"Text content: {element.text}")

    if hasattr(element, 'metadata'):
        print("Metadata:")
        metadata_dict = element.metadata.__dict__
        for key, value in metadata_dict.items():
            if not key.startswith('_') and value is not None:
                print(f"  {key}: {value}")
