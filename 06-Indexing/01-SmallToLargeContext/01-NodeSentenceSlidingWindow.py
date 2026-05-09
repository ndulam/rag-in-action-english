from llama_index.core import  VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import  SentenceWindowNodeParser, SentenceSplitter
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import MetadataReplacementPostProcessor # Metadata replacement post-processor
# Configure global settings
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
Settings.text_splitter = SentenceSplitter(separator="\n",  chunk_size=50, chunk_overlap=0)
# Prepare knowledge text and create Document objects
game_knowledge = """
"Chronicles of the Divine: Wukong" is an action role-playing game. The game is set in an imaginary mythological world.
Players will take on the role of Sun Wukong, the Great Sage Equal to Heaven, embarking on adventures in a world full of Eastern mythological elements.
The game's combat system is highly distinctive, featuring a unique "Transformation System". Wukong can switch between different forms during combat.
Each form has its own unique combat style and skill combinations. The Vajra form focuses on power strikes, delivering overwhelming destructive force.
The Demon Buddha form focuses on magical attacks, capable of unleashing powerful magical damage.
The game world is filled with iconic mythological characters; besides the protagonist Sun Wukong, there are also gods and demons from Buddhist, Taoist, and other sects.
These characters may be either allies of Wukong or powerful opponents that need to be defeated.
The equipment system includes a rich variety of weapon choices; besides the famous Ruyi Jingu Bang, Wukong can also use various divine artifacts and treasures.
Different weapons have their own special effects, and players need to choose flexibly based on the combat scenario.
The game's visuals are highly characteristic of Eastern aesthetics, with scenes blending ink painting styles that perfectly present mountains, buildings, and other elements.
Combat effects incorporate both traditional Chinese cultural elements and the visual impact of modern games.
In terms of difficulty design, boss battles are highly challenging, requiring players to precisely master combat rhythm and skill usage.
At the same time, the game also provides multiple difficulty options to accommodate players of different skill levels."""
# Create Document objects
documents = [Document(text=game_knowledge)]
# Create sentence parser with context window (retain n sentences on each side of the target sentence as context)
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)
# Process documents with window parser
nodes = node_parser.get_nodes_from_documents(documents)
# Process documents with base parser (for comparison)
base_nodes = Settings.text_splitter.get_nodes_from_documents(documents)
# Build two indexes for comparison
sentence_index = VectorStoreIndex(nodes)
base_index = VectorStoreIndex(base_nodes)
# Create query engine with context window
window_query_engine = sentence_index.as_query_engine(
    similarity_top_k=2,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ]
)
# Create base query engine
base_query_engine = base_index.as_query_engine(
    similarity_top_k=6
)
# Test Q&A
test_questions = [
    "What transformation forms does Wukong have in the game?",
    # "What is the visual style of the game?",
    # "How is the difficulty design of the game?"
]
print("=== Retrieval results using window parser ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    window_response = window_query_engine.query(question)
    print(f"Answer: {window_response}")

    # Show retrieved original sentences and window content
    print("\nRetrieval details:")
    for node in window_response.source_nodes:
        print(f"Original sentence: {node.node.metadata['original_text']}")
        print(f"Context window: {node.node.metadata['window']}")
        print("---")
print("\n=== Retrieval results using base parser (for comparison) ===")
for question in test_questions:
    print(f"\nQuestion: {question}")
    base_response = base_query_engine.query(question)
print(f"Answer: {base_response}")
