from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import IndexNode, Document
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from typing import List
# Configure global settings
Settings.llm = DeepSeek(model="deepseek-chat", temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")
# Create game scene descriptions (main documents)
scene_descriptions = [
    """
    Flower Fruit Mountain: This is the birthplace of Sun Wukong, the Great Sage Equal to Heaven. The mountain is perpetually shrouded in celestial mist, with waterfalls plunging from a thousand meters high,
    forming the "Heavenly River Waterfall". The mountain grows various immortal herbs and spiritual medicines, and also houses many animals that have cultivated into spirits.
    """,
    """
    Water Curtain Cave: Located at the peak of Flower Fruit Mountain, there is a naturally formed water curtain at the cave entrance, serving as both a natural barrier and a sacred cultivation site.
    """,
    """
    East Sea Dragon Palace: A magnificent palace at the bottom of the East Sea, decorated with coral and night-luminous pearls. This is the place where Sun Wukong obtained the Ruding Shenzhen (the pillar that stabilizes the sea).
    """
]
# Convert scene descriptions to Document objects
documents = [Document(text=desc) for desc in scene_descriptions]
# Use node parser to convert documents to nodes
doc_nodes = Settings.node_parser.get_nodes_from_documents(documents)
# Create IndexNodes representing hierarchical relationships
# Create scene detailed information (simulating documents with details)
scene_details = [
    """
    Flower Fruit Mountain - Detailed Settings
    1. Geographic location: Within Aolai Kingdom of the Eastern Superior Continent (Dongsheng Shenzhou)
    2. Natural environment: Ever-blooming exotic flowers and herbs, clear springs and waterfalls, dense ancient forests
    3. Special areas: Immortal Fruit Garden, growing various spiritual fruits; Training Ground, a flat open cultivation area; Rest Area, resting place for the monkey tribe
    """,
    """
    Water Curtain Cave - Detailed Settings
    1. Architectural structure: Exterior, giant natural rock cave; Entrance, 30-zhang high water curtain waterfall; Interior, complex cave system
    2. Functional areas: Cultivation Hall, equipped with various cultivation implements; Treasure Room, storing various magical artifacts and elixirs with powerful protective formations; Council Hall, able to accommodate hundreds of the monkey tribe for discussing important matters.
    """,
    """
    East Sea Dragon Palace - Detailed Settings
    1. Architectural features: Materials, coral, pearls, night-luminous pearls; Scale, spanning dozens of li; Style, underwater palace complex.
    2. Important locations: Dragon King's Treasury, storing countless treasures such as night-luminous pearls and divine weapons like the Ruding Shenzhen; Armory, various water-element magical weapons and divine soldiers; Main Hall, the formal hall for receiving guests and hosting underwater council meetings.
    """
]
# Create IndexNodes for each detailed information and create corresponding query engines
index_nodes = []
index_id_query_engine_mapping = {}
for idx, detail_text in enumerate(scene_details):
    # Create IndexNode, process text before putting into f-string
    index_id = f"detail{idx}"
    first_line = detail_text.split('\n')[1].strip()
    index_node = IndexNode(text=f"This node contains {first_line}", index_id=index_id)
    index_nodes.append(index_node)
    # Create corresponding TextNode, build separate index and query engine
    detail_node = Document(text=detail_text)
    detail_index = VectorStoreIndex.from_documents([detail_node])
    detail_query_engine = detail_index.as_query_engine()
    # Add query engine to mapping
    index_id_query_engine_mapping[index_id] = detail_query_engine
    # Output current mapping status
    print(f"\nCurrent index ID: {index_id}")
    print(f"Index node text: {index_node.text}")
    print(f"Corresponding scene detail length: {len(detail_text)} characters")
    print(f"Query engine type: {type(detail_query_engine).__name__}")
print("-" * 30)

# Merge document nodes and index nodes
all_nodes = doc_nodes + index_nodes
# Build main vector index
vector_index = VectorStoreIndex(all_nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=2)
# Create RecursiveRetriever object
recursive_retriever = RecursiveRetriever(
    "vector",  # Root retriever ID
    retriever_dict={"vector": vector_retriever},  # Retriever mapping
    query_engine_dict=index_id_query_engine_mapping,  # Query engine mapping
    verbose=True,  # Enable verbose output
)
# Create RetrieverQueryEngine with response mode "compact"
response_synthesizer = get_response_synthesizer(response_mode="compact")
# Create RetrieverQueryEngine, then pass in RecursiveRetriever and response synthesizer
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer,
)
# Define query function
def query_scene(question: str):
    print(f"Question: {question}\n")
    response = query_engine.query(question)
    print(f"Answer: {str(response)}\n")
    print("-" * 50)
# Example queries
if __name__ == "__main__":
    questions = [
        "What special places are in Flower Fruit Mountain?",
        "Describe the internal structure of Water Curtain Cave in detail.",
        "What treasures are stored in the East Sea Dragon Palace?",
    ]

    for q in questions:
        query_scene(q)
