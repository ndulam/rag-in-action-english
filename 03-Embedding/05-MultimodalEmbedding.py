"""
Simple multimodal embedding example: Encoding images using the Visualized-BGE model.
To install visual_bge, refer to:
https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge#readme

# If visual_bge is installed but still cannot be used, try switching to a physical environment;
# virtual environments may not support it.
"""

import torch
from visual_bge.modeling import Visualized_BGE
from PIL import Image
import numpy as np

# Initialize the encoder
model_name = "BAAI/bge-base-en-v1.5"
# Define the model path (use an absolute path if there are issues)
# Download the model weight file in advance:
# wget https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_base_en_v1.5.pth
model_path = "/root/AI-BOX/code/rag/rag-in-action/03-Embedding/Visualized_base_en_v1.5.pth"
model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
model.eval()

# Define the image path (use an absolute path if there are issues)
image_path = "/root/AI-BOX/code/rag/rag-in-action/90-Data/multimodal/query_image.jpg"

# Encode the image
with torch.no_grad():
    # Encode using image only
    image_embedding = model.encode(image=image_path)

    # Encode using both image and text
    text = "This is a sample battle image of Wukong"
    multimodal_embedding = model.encode(image=image_path, text=text)

# Move tensors to CPU and convert to numpy arrays
image_embedding_np = image_embedding.cpu().numpy()
multimodal_embedding_np = multimodal_embedding.cpu().numpy()

# Print embedding information
print("=== Image Embedding Information ===")
print(f"Embedding dimensions: {image_embedding_np.shape[1]}")
print(f"Embedding sample (first 10 elements): {image_embedding_np[0][:10]}")
print(f"Embedding norm: {np.linalg.norm(image_embedding_np[0])}")

print("\n=== Multimodal Embedding Information ===")
print(f"Embedding dimensions: {multimodal_embedding_np.shape[1]}")
print(f"Embedding sample (first 10 elements): {multimodal_embedding_np[0][:10]}")
print(f"Embedding norm: {np.linalg.norm(multimodal_embedding_np[0])}")
