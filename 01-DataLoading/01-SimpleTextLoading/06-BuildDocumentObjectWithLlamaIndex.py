from llama_index.core import Document

# Create multiple document objects and add metadata to them
documents = [
    Document(
        text="A subterranean cavern filled with blazing flames and the scent of sulfur, where fire spews ceaselessly from the ground, illuminating the entire abyss. The scene features rivers of lava flowing through it, with burning volcanic rocks floating in the air. Sun Wukong must use his jumping ability and the staff to navigate between the lava flows while fighting off fire demons from the depths of hell.",
        metadata={
            "filename": "fire-abyss-scene.md",
            "category": "game scene",
            "file_path": "/data/black-myth-wukong/fire-abyss-scene.md",
            "author": "GameScience",
            "creation_date": "2024-11-20",
            "last_modified_date": "2024-11-21",
            "file_type": "markdown",
            "word_count": 28,
        },
    ),
    Document(
        text="A towering mountain range that rises into the clouds, shrouded in mist with powerful winds. Sun Wukong must leap across cliffs, fly on the golden cloud, and maintain balance in the strong winds to traverse the scene. Enemies are mainly bird demons lurking in the clouds and mechanical rock beasts.",
        metadata={
            "filename": "wind-soaring-sky-scene.md",
            "category": "game scene",
            "file_path": "/data/black-myth-wukong/wind-soaring-sky-scene.md",
            "author": "GameScience",
            "creation_date": "2024-11-20",
            "last_modified_date": "2024-11-21",
            "file_type": "markdown",
            "word_count": 28,
        },
    )]

# Print the metadata of each document
for doc in documents:
    print(f"Metadata for {doc.metadata['filename']}:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")
print("-" * 40)
