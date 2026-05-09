from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.postprocessor import PrevNextNodePostprocessor, AutoPrevNextNodePostprocessor
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Configure global settings
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
Settings.node_parser = SentenceSplitter()
# Prepare game story text
game_story = """When Wukong first awakened, he found himself trapped in an ancient cave. With hazy memories, he could only remember that he was Sun Wukong, the Great Sage Equal to Heaven, but could not recall why he was there. Inside the cave was a broken mirror; through it he saw himself covered in wounds, and his once-mighty Ruyi Jingu Bang was reduced to a broken stump. After leaving the cave, Wukong encountered a mysterious old monk. The old monk told him that this place was the "Phantom Realm," a special space between reality and illusion. Five hundred years ago, Heaven suffered an unprecedented catastrophe; the gods fell and the celestial realm collapsed. Wukong, who was causing havoc in Heaven at the time, was also swept into it, losing most of his magical power and memories, and was sealed in this world. The old monk suggested that Wukong go looking for memory fragments scattered throughout the Phantom Realm. The first destination was Wangchuan Temple in the east, which housed a Mirror of Memory that might help him recover some memories. However, Wangchuan Temple had been occupied by a group of evil demons, and Wukong needed to defeat them first. At Wangchuan Temple, Wukong saw partial scenes of the Heaven catastrophe through the Mirror of Memory. It turned out that a mysterious ancient force was manipulating things from behind, using the power of "the wishes of all beings" to distort the rules of heaven and earth. Although Wukong was powerful at the time, he was unable to stop the disaster. After obtaining these memories, the old monk told Wukong that the next destination should be the Mountain of Burning Fire in the west. There, a transformed demon clan held more truth. But the Mountain of Burning Fire was surrounded by raging flames year-round, and ordinary beings could not approach it. Wukong first needed to find the legendary Samadhi Fire Armor to enter safely. While searching for the Samadhi Fire Armor, Wukong met an old friend, the Demon King. The Demon King told him that after Heaven's collapse, the order of the six realms fell into chaos, and various forces rose one after another. Some used the banner of rebuilding Heaven, while others wanted to establish a completely new order. A greater catastrophe was brewing. After obtaining the Samadhi Fire Armor, Wukong successfully infiltrated the Mountain of Burning Fire. In the showdown with the demon clan leader, he finally recalled more truth. It turned out that the goal of the ancient force was not simple destruction, but to reshape the rules of the entire world. They believed that the existing order had fundamental flaws that caused all beings to suffer. Returning to the old monk's side, Wukong expressed his desire to unite various forces to fight the force behind the scenes. But the old monk told him that things might not be as simple as they appeared on the surface. Whether the world order should be reshaped was a question without a standard answer. He suggested that Wukong continue to seek more truth before making a decision. Wukong decided to set off for the Star-Sinking Sea in the south. Legend had it that there was an ancient library there, containing many books about the origin of the world. However, before he could depart, the Phantom Realm suddenly shook violently, as if some great change was about to happen..."""
# Create Document objects
documents = [Document(text=game_story)]
# Build document storage and index, using node_parser from Settings to parse documents
nodes = Settings.node_parser.get_nodes_from_documents(documents)
# Create document store and add nodes
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
# Create storage context
storage_context = StorageContext.from_defaults(docstore=docstore)
# Build vector index
index = VectorStoreIndex(nodes, storage_context=storage_context)
# Create different query engines
# Base query engine
base_engine = index.as_query_engine(
    similarity_top_k=1,
    response_mode="tree_summarize"
)
# Query engine with fixed forward/backward context
prev_next_engine = index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[
        PrevNextNodePostprocessor(docstore=docstore, num_nodes=2)
    ],
    response_mode="tree_summarize"
)
# Query engine with automatic forward/backward context
auto_engine = index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[
        AutoPrevNextNodePostprocessor(
            docstore=docstore,
            num_nodes=3,
            verbose=True
        )
    ],
    response_mode="tree_summarize"
)
# Test different types of questions with different query engines
test_questions = [
    "What happened after Wukong obtained memories from Wangchuan Temple?",  # Should look in subsequent text
    "How did Wukong get to the Mountain of Burning Fire?",  # Should look in previous text
    "Why did Wukong wake up in the cave?",  # Should look in previous text
]
print("=== Results from base query engine ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    response = base_engine.query(question)
    print(f"Answer: {response}\n")
    print("-" * 50)
print("\n=== Results from fixed forward/backward context query engine ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    response = prev_next_engine.query(question)
    print(f"Answer: {response}\n")
    print("-" * 50)
print("\n=== Results from automatic forward/backward context query engine ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    response = auto_engine.query(question)
    print(f"Answer: {response}\n")
print("-" * 50)
