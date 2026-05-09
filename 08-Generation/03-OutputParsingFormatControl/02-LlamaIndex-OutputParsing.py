from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

# Define the game information structure
class GameInfo(BaseModel):
    title: str = Field(description="Game title")
    developer: str = Field(description="Developer")
    release_date: str = Field(description="Release date")
    platforms: List[str] = Field(description="Supported platforms")
    main_features: List[str] = Field(description="Main features")
    story_summary: str = Field(description="Story summary")
    reception: str = Field(description="Market reception")

# Load data
documents = SimpleDirectoryReader(input_files=["90-Data/black-myth-wukong/black-myth-wukong-wiki.txt"], encoding="utf-8").load_data()
index = VectorStoreIndex.from_documents(documents)

# 1. Basic parsing mode - using COMPACT mode
print("=== Basic Parsing Mode ===")
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    verbose=True    # Show detailed information
)
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
response = query_engine.query("Please summarize the main content of the game Black Myth: Wukong")
print(response)

# 2. Structured parsing mode - using REFINE mode
print("\n=== Structured Parsing Mode ===")
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.REFINE,
    output_cls=GameInfo,  # Specify output class
    verbose=True
)
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
response = query_engine.query("Please extract key information about Black Myth: Wukong")
# Safely handle the response
if hasattr(response, 'response'):
    print(response.response)
else:
    print(response)

# 3. Table format parsing - using TREE_SUMMARIZE mode
print("\n=== Game Features Table Parsing ===")
table_prompt = PromptTemplate(
    template="Please present the following game features in a table format:\n{query_str}\nFormat requirements:\n| Category | Content |\n|------|------|\n"
)
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    summary_template=table_prompt,
    verbose=True
)
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
response = query_engine.query("Please summarize the main features of Black Myth: Wukong in table format")
print(response)

# 4. Bullet point parsing mode - using COMPACT_ACCUMULATE mode
print("\n=== Game Highlights Bullet Point Parsing ===")
bullet_prompt = PromptTemplate(
    template="Please present the following game highlights in bullet point format:\n{query_str}\nFormat requirements:\n1. \n2. \n3. "
)
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT_ACCUMULATE,
    text_qa_template=bullet_prompt,
    verbose=True,
    use_async=True  # Enable asynchronous processing
)
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
response = query_engine.query("Please summarize the highlights of Black Myth: Wukong in bullet point format")
print(response)

# 5. Story timeline parsing - using SIMPLE_SUMMARIZE mode
print("\n=== Game Story Timeline Parsing ===")
story_prompt = PromptTemplate(
    template="Please present the following game story in timeline format:\n{query_str}\nFormat requirements:\n- Timestamp: Event\n"
)
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.SIMPLE_SUMMARIZE,
    text_qa_template=story_prompt,
    verbose=True
)
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
response = query_engine.query("Please summarize the story development of Black Myth: Wukong in timeline format")
print(response)
