from pdf2image import convert_from_path
import base64
import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()
output_dir = "temp_images"

# 1. Convert PDF to images
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = convert_from_path("90-Data/black-myth-wukong/black-myth-wukong.pdf")
image_paths = []
for i, image in enumerate(images):
    image_path = os.path.join(output_dir, f'page_{i+1}.jpg')
    image.save(image_path, 'JPEG')
    image_paths.append(image_path)
print(f"Successfully converted {len(image_paths)} pages")


# 2. Analyze images with GPT-4o
print("\nStarting image analysis...")
results = []
for image_path in image_paths:
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please describe in detail the content of this PPT slide, including the title, body text, and image content."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
results.append(response.choices[0].message.content)


# 3. Convert to LangChain Document data structure
from langchain_core.documents import Document

documents = [
    Document(
        page_content=result,
        metadata={"source": "data/black-myth-wukong/black-myth-wukong.pdf", "page_number": i + 1}
    )
    for i, result in enumerate(results)
]

# Output all generated Document objects
print("\nAnalysis results:")
for doc in documents:
    print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}\n")
    print("-" * 80)

# Clean up temporary files
for image_path in image_paths:
    os.remove(image_path)
os.rmdir(output_dir)
