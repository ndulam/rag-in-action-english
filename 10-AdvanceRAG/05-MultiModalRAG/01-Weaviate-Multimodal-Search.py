import weaviate
import os
import base64
from weaviate.classes.config import Configure
from weaviate.classes.query import NearMediaType
# from IPython.display import Image, Audio, Video

# Connect to local Weaviate instance
client = weaviate.connect_to_local()

# Check and create the "Monkey" collection
if client.collections.exists("Monkey"):
    client.collections.delete("Monkey")

client.collections.create(
    name="Monkey",
    vectorizer_config=Configure.Vectorizer.multi2vec_bind(
        image_fields=["image"],
        audio_fields=["audio"],
        video_fields=["video"]
    )
)

# Convert file to base64 encoding
def to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Insert image data
image_dir = "90-Data/multimodal/Weaviate"
image_files = os.listdir(image_dir)
monkey = client.collections.get("Monkey")
for name in image_files:
    path = os.path.join(image_dir, name)
    monkey.data.insert({
        "name": name,
        "path": path,
        "image": to_base64(path),
        "mediaType": "image"
    })

# # Insert audio data
# audio_dir = "./data/audio/"
# audio_files = os.listdir(audio_dir)
# for name in audio_files:
#     path = os.path.join(audio_dir, name)
#     animals.data.insert({
#         "name": name,
#         "path": path,
#         "audio": to_base64(path),
#         "mediaType": "audio"
#     })

# # Insert video data
# video_dir = "./data/video/"
# video_files = os.listdir(video_dir)
# for name in video_files:
#     path = os.path.join(video_dir, name)
#     animals.data.insert({
#         "name": name,
#         "path": path,
#         "video": to_base64(path),
#         "mediaType": "video"
#     })

# Text search example
query = "Monkey with fire"
response = monkey.query.near_text(
    query=query,
    return_properties=["name", "path", "mediaType"],
    limit=3
)
print(f"Objects similar to query '{query}':")
for obj in response.objects:
    print(obj.properties)

query = "Monsters"
response = monkey.query.near_text(
    query=query,
    return_properties=["name", "path", "mediaType"],
    limit=3
)
print(f"Objects similar to query '{query}':")
for obj in response.objects:
    print(obj.properties)

# Image search example
test_image_path = "90-Data/multimodal/query_image.jpg"
response = monkey.query.near_image(
    near_image=to_base64(test_image_path),
    return_properties=["name", "path", "mediaType"],
    limit=3
)
print("Objects similar to the current image:")
for obj in response.objects:
    print(obj.properties)

# # Audio search example
# test_audio_path = "./test/test-audio.wav"
# response = animals.query.near_media(
#     media=to_base64(test_audio_path),
#     media_type=NearMediaType.AUDIO,
#     return_properties=["name", "path", "mediaType"],
#     limit=3
# )
# for obj in response.objects:
#     print(obj.properties)

# # Video search example
# test_video_path = "./test/test-video.mp4"
# response = animals.query.near_media(
#     media=to_base64(test_video_path),
#     media_type=NearMediaType.VIDEO,
#     return_properties=["name", "path", "mediaType"],
#     limit=3
# )
# for obj in response.objects:
#     print(obj.properties)

# img_path = "90-Data/multimodal/Weaviate/02.jpg"
# with open(img_path, "rb") as f:
#     image_base64 = base64.b64encode(f.read()).decode("utf-8")

# animals.data.insert({
#     "name": "02.jpg",
#     "image": image_base64,
#     "mediaType": "image"
# })


client.close()