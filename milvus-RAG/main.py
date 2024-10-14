import os
from glob import glob

import torch
from FlagEmbedding.visual.modeling import Visualized_BGE
from pymilvus import MilvusClient
from tqdm import tqdm


class Encoder:
    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


# Change to your own value if using a different model path
model_name = "BAAI/bge-base-en-v1.5"
model_path = "./Visualized_base_en_v1.5.pth"
encoder = Encoder(model_name, model_path)


# load data
# Change to your own value if using a different data directory
data_dir = (
    "/home/stardust/Downloads/test_images/"
)

image_list = glob(
    os.path.join(data_dir, "images", "*.png")
)  # We will only use images ending with ".jpg"
image_dict = {}
for image_path in tqdm(image_list, desc="Generating image embeddings: "):
    try:
        image_dict[image_path] = encoder.encode_image(image_path)
    except Exception:
        print(f"Failed to generate embedding for {image_path}. Skipped.")
        continue
print("Number of encoded images:", len(image_dict))



# Insert into Milvus

dim = len(list(image_dict.values())[0])
collection_name = "multimodal_rag_demo"

milvus_client = MilvusClient(uri="./milvus_demo.db")

milvus_client.create_collection(
    collection_name=collection_name,
    auto_id=True,
    dimension=dim,
    enable_dynamic_field=True,
)

milvus_client.insert(
    collection_name=collection_name,
    data=[{"image_path": k, "vector": v} for k, v in image_dict.items()],
)


# Multimodal Search with Generative Reranker

query_image = os.path.join(
    data_dir, "image.png"
)  # Change to your own query image path
query_text = "find the waiting-benches of this style"

query_vec = encoder.encode_query(image_path=query_image, text=query_text)

search_results = milvus_client.search(
    collection_name=collection_name,
    data=[query_vec],
    output_fields=["image_path"],
    limit=9,  # Max number of search results to return
    search_params={"metric_type": "COSINE", "params": {}},  # Search parameters
)[0]

retrieved_images = [hit.get("entity").get("image_path") for hit in search_results]
print(retrieved_images)
