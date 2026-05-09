"""
Multimodal image retrieval system: Implemented with Visualized-BGE and Milvus.
Features: Multimodal encoding of images and text, retrieval of similar content from an image database.
"""

# ==================== 1. Initialize Encoder ====================
import torch
from visual_bge.modeling import Visualized_BGE
from dataclasses import dataclass
from typing import List, Optional
import json
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from pymilvus import MilvusClient

class WukongEncoder:
    """Multimodal encoder: encodes images and text into vectors"""
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        """Encode a combined image and text query"""
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        """Encode image only"""
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]

# Initialize encoder
model_name = "BAAI/bge-m3"
model_path = "./Visualized_m3.pth"
encoder = WukongEncoder(model_name, model_path)

# ==================== 2. Dataset Management ====================
@dataclass
class WukongImage:
    """Image metadata structure"""
    image_id: str
    file_path: str
    title: str
    category: str
    description: str
    tags: List[str]
    game_chapter: str
    location: str
    characters: List[str]
    abilities_shown: List[str]
    environment: str
    time_of_day: str

class WukongDataset:
    """Image dataset management class"""
    def __init__(self, data_dir: str, metadata_path: str):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.images: List[WukongImage] = []
        self._load_metadata()

    def _load_metadata(self):
        """Load image metadata"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for img_data in data['images']:
                # Ensure image path is relative to data_dir
                img_data['file_path'] = f"{self.data_dir}/{img_data['file_path'].split('/')[-1]}"
                self.images.append(WukongImage(**img_data))

# Initialize dataset
dataset = WukongDataset("90-Data/multimodal", "90-Data/multimodal/metadata.json")

# ==================== 3. Generate Image Embeddings ====================
# Generate embedding vectors for all images
image_dict = {}
for image in tqdm(dataset.images, desc="Generating image embeddings"):
    try:
        image_dict[image.file_path] = encoder.encode_image(image.file_path)
    except Exception as e:
        print(f"Failed to process image {image.file_path}: {str(e)}")
        continue

print(f"Successfully encoded {len(image_dict)} images")

# ==================== 4. Milvus Vector Store Setup ====================
# Connect to / create Milvus database
collection_name = "wukong_scenes"
milvus_client = MilvusClient(uri="./wukong_images.db")

# Create vector collection
dim = len(list(image_dict.values())[0])
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=dim,
    auto_id=True,
    enable_dynamic_field=True
)

# Insert data into Milvus
insert_data = []
for image in dataset.images:
    if image.file_path in image_dict:
        insert_data.append({
            "image_path": image.file_path,
            "vector": image_dict[image.file_path],
            "title": image.title,
            "category": image.category,
            "description": image.description,
            "tags": ",".join(image.tags),
            "game_chapter": image.game_chapter,
            "location": image.location,
            "characters": ",".join(image.characters),
            "abilities": ",".join(image.abilities_shown),
            "environment": image.environment,
            "time_of_day": image.time_of_day
        })

result = milvus_client.insert(
    collection_name=collection_name,
    data=insert_data
)
print(f"Index built, inserted {result['insert_count']} records")

# ==================== 5. Search Function Implementation ====================
def search_similar_images(
    query_image: str,
    query_text: str,
    limit: int = 9
) -> List[dict]:
    """
    Search for similar images.
    Parameters:
        query_image: Query image path
        query_text: Query text
        limit: Number of results to return
    Returns:
        List of retrieval results
    """
    # Generate query vector
    query_vec = encoder.encode_query(query_image, query_text)

    # Build search parameters
    search_params = {
        "metric_type": "COSINE",
        "params": {
            "nprobe": 10,
            "radius": 0.1,
            "range_filter": 0.8
        }
    }

    # Execute search
    results = milvus_client.search(
        collection_name=collection_name,
        data=[query_vec],
        output_fields=[
            "image_path", "title", "category", "description",
            "tags", "game_chapter", "location", "characters",
            "abilities", "environment", "time_of_day"
        ],
        limit=limit,
        search_params=search_params
    )[0]

    return results

# ==================== 6. Visualization Function ====================
def visualize_results(query_image: str, results: List[dict], output_path: str):
    """
    Visualize search results.
    Parameters:
        query_image: Query image path
        results: List of search results
        output_path: Output image path
    """
    # Set image size and grid parameters
    img_size = (300, 300)
    grid_size = (3, 3)

    # Create canvas
    canvas_height = img_size[0] * (grid_size[0] + 1)
    canvas_width = img_size[1] * (grid_size[1] + 1)
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    # Add query image
    query_img = Image.open(query_image).convert("RGB")
    query_array = np.array(query_img)
    query_resized = cv2.resize(query_array, (img_size[0] - 20, img_size[1] - 20))
    bordered_query = cv2.copyMakeBorder(
        query_resized, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT,
        value=(255, 0, 0)
    )
    canvas[:img_size[0], :img_size[1]] = bordered_query

    # Add result images
    for idx, result in enumerate(results[:grid_size[0] * grid_size[1]]):
        row = (idx // grid_size[1]) + 1
        col = idx % grid_size[1]

        img = Image.open(result["entity"]["image_path"]).convert("RGB")
        img_array = np.array(img)
        resized = cv2.resize(img_array, (img_size[0], img_size[1]))

        y_start = row * img_size[0]
        x_start = col * img_size[1]

        canvas[y_start:y_start + img_size[0], x_start:x_start + img_size[1]] = resized

        # Add similarity score
        score_text = f"Score: {result['distance']:.2f}"
        cv2.putText(
            canvas,
            score_text,
            (x_start + 10, y_start + img_size[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

    cv2.imwrite(output_path, canvas)

# ==================== 7. Execute Query Example ====================
# Execute query
query_image = "90-Data/multimodal/query_image.jpg"
query_text = "Find Wukong facing a building in a combat scene"

results = search_similar_images(
    query_image=query_image,
    query_text=query_text,
    limit=9
)

# Output detailed information
print("\nSearch results:")
for idx, result in enumerate(results):
    print(f"\nResult {idx}:")
    print(f"Image: {result['entity']['image_path']}")
    print(f"Title: {result['entity']['title']}")
    print(f"Description: {result['entity']['description']}")
    print(f"Similarity score: {result['distance']:.4f}")

# Visualize results
visualize_results(query_image, results, "search_results.jpg")
