"""
Retrieval-only program: Retrieval based on an already-built Milvus vector store.
"""

import torch
from pymilvus import MilvusClient
from PIL import Image
import cv2
import numpy as np
from typing import List, Optional, Dict
from visual_bge.modeling import Visualized_BGE

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

class MilvusSearcher:
    """Milvus retriever"""
    def __init__(self, db_path: str, collection_name: str):
        self.client = MilvusClient(uri=db_path)
        self.collection_name = collection_name

    def search(
        self,
        query_vector: List[float],
        limit: int = 9,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Execute vector retrieval.
        Parameters:
            query_vector: Query vector
            limit: Number of results to return
            filters: Filter conditions
        Returns:
            List of retrieval results
        """
        # Build search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        # Build filter expression
        filter_expr = None
        if filters:
            conds = []
            for k, v in filters.items():
                if isinstance(v, list):
                    vs = ", ".join(f"'{x}'" for x in v)
                    conds.append(f"{k} in [{vs}]")
                else:
                    conds.append(f"{k} == '{v}'")
            filter_expr = " and ".join(conds)

        # Execute search
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            filter=filter_expr,
            limit=limit,
            output_fields=[
                "image_path", "title", "category", "description",
                "tags", "game_chapter", "location", "characters",
                "abilities", "environment", "time_of_day"
            ],
            search_params=search_params
        )[0]

        return results

def visualize_results(query_image: str, results: List[dict], output_path: str):
    """
    Visualize search results.
    Parameters:
        query_image: Query image path
        results: List of search results
        output_path: Output image path
    """
    img_size = (300, 300)
    grid_size = (3, 3)

    canvas_h = img_size[1] * (grid_size[0] + 1)
    canvas_w = img_size[0] * (grid_size[1] + 1)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # Place query image
    qimg = Image.open(query_image).convert("RGB")
    qarr = np.array(qimg)
    qrs = cv2.resize(qarr, (img_size[0]-20, img_size[1]-20))
    bq = cv2.copyMakeBorder(qrs, 10,10,10,10, cv2.BORDER_CONSTANT, value=(255,0,0))
    canvas[0:img_size[1], 0:img_size[0]] = bq

    # Place result images
    for i, r in enumerate(results[:grid_size[0]*grid_size[1]]):
        row = (i // grid_size[1]) + 1
        col = i % grid_size[1]
        img = Image.open(r["entity"]["image_path"]).convert("RGB")
        arr = np.array(img)
        rs = cv2.resize(arr, img_size)
        y0 = row * img_size[1]
        x0 = col * img_size[0]
        canvas[y0:y0+img_size[1], x0:x0+img_size[0]] = rs
        text = f"Score:{r['distance']:.2f}"
        cv2.putText(canvas, text, (x0+10, y0+img_size[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    cv2.imwrite(output_path, canvas)

def print_results(results: List[dict]):
    """Print search results"""
    print("\nSearch results:")
    for idx, r in enumerate(results):
        print(f"\nResult {idx+1}:")
        print(f"  Image: {r['entity']['image_path']}")
        print(f"  Title: {r['entity']['title']}")
        print(f"  Description: {r['entity']['description']}")
        print(f"  Environment: {r['entity']['environment']}")
        print(f"  Category: {r['entity']['category']}")
        print(f"  Similarity: {r['distance']:.4f}")

if __name__ == "__main__":
    # Initialize encoder (switch to Chinese model if needed)
    model_name = "BAAI/bge-base-en-v1.5"
    model_path = "./Visualized_base_en_v1.5.pth"
    encoder = WukongEncoder(model_name, model_path)

    # Initialize retriever
    searcher = MilvusSearcher("./wukong_images.db", "wukong_scenes")

    # Generate query vector
    query_image = "90-Data/multimodal/query_image.jpg"
    query_text = "Find similar snow combat scenes"
    qvec = encoder.encode_query(query_image, query_text)

    # Retrieval with filter conditions
    filters = {"environment": "Snow", "category": "combat"}
    res_f = searcher.search(qvec, limit=9, filters=filters)
    print("\nFiltered results:")
    print_results(res_f)
    visualize_results(query_image, res_f, "search_with_filter.jpg")

    # Retrieval without filter conditions
    res_nf = searcher.search(qvec, limit=9)
    print("\nUnfiltered results:")
    print_results(res_nf)
    visualize_results(query_image, res_nf, "search_without_filter.jpg")
