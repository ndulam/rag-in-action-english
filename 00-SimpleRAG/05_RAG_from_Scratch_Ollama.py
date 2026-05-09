# 1. Prepare document data
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

docs = [
    "The combat in Black Myth: Wukong feels like a martial arts novel come to life — when the staff clashes with demons, sparks fly and every move flows seamlessly. Sun Wukong can freely switch between a ferocious or agile fighting style, sweeping thousands aside with a single strike, or darting about like a butterfly among flowers.",
    "The 72 Transformations are not merely about changing form — they are the key to unlocking new worlds. Transforming into a flying squirrel lets you sneak into demon lairs to gather intelligence; becoming a goldfish lets you explore the secrets of underwater ruins. Each transformation is a unique adventure.",
    "Every BOSS battle is a breathtaking showdown. You might clash with the massive nine-headed python at the top of a waterfall, or trade spells with the Thunder God and Lightning Mother in a cloud-sea electrified by lightning — every move fraught with danger.",
    "Soaring through this mythical world on the golden cloud, the breathtaking scenery leaves you speechless. Immortal mountains drift in and out of mist, ancient demon lairs hide thousand-year-old treasures, and the bell of an ancient temple echoes through moonlit valleys.",
    "This is not the Journey to the West you remember. As Sun Wukong sets out to uncover the mystery of his origins, he will encounter all manner of gods and demons — some familiar faces, like the equally rebellious Nezha; others formidable foes, like Erlang Shen with his Three-Pointed Double-Edged Spear.",
    "As the Great Sage Equal to Heaven, Sun Wukong's powers go beyond the staff. His Fiery Eyes and Golden Pupils can see through a demon's true form, and a single somersault covers 108,000 li. These abilities can also be further enhanced by collecting materials like meteoric iron and enlightenment stones.",
    "Every corner of the world hides a story. You might find the ruins of an ancient great power in a cave, discover a fallen celestial soldier's treasure vault in a heavenly palace, or stumble upon a fox spirit selling ginseng fruit in a mortal market.",
    "The story takes place in the primeval world before the Tang Dynasty, when the Heavenly Court had not yet unified the Three Realms and demon kings ruled their own territories. It was a turbulent era of battles between gods and demons — and the starting point of Sun Wukong's quest for the truth.",
    "The game's music is like an epic spanning a thousand years. Ancient qin and orchestral strings weave the intensity of battle, while flutes and wooden fish instruments compose a zen-like serenity. When Sun Wukong enters a key scene, the ancient-style score makes you feel transported back to the age of myth."
    ]

# 2. Set up the embedding model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
doc_embeddings = model.encode(docs)
print(f"Document vector dimensions: {doc_embeddings.shape}")

# 3. Create the vector store
import faiss # pip install faiss-cpu
import numpy as np
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.astype('float32'))
print(f"Number of documents in the vector database: {index.ntotal}")

# 4. Perform similarity search
question = "What are the characteristics of the combat system in Black Myth: Wukong?"
query_embedding = model.encode([question])[0]
distances, indices = index.search(
    np.array([query_embedding]).astype('float32'),
    k=3
)
context = [docs[idx] for idx in indices[0]]
print("\nRetrieved relevant documents:")
for i, doc in enumerate(context, 1):
    print(f"[{i}] {doc}")

# 5. Build the prompt
prompt = f"""Answer the question based on the following reference information and provide the source numbers.
If you cannot find the answer in the reference information, please say you are unable to answer.
Reference information:
{chr(10).join(f"[{i+1}] {doc}" for i, doc in enumerate(context))}
Question: {question}
Answer:"""

# 6. Generate the answer using Ollama
from ollama import chat

response = chat(
    model=os.getenv("OLLAMA_MODEL"),
    messages=[{
        "role": "user",
        "content": prompt
    }],
)
print(f"\nGenerated answer: {response.message.content}")
