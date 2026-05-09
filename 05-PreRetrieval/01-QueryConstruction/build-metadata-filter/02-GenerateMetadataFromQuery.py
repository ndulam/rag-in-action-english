# Import required libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
# Define video metadata model
class VideoMetadata(BaseModel):
    """Video metadata model defining the video attributes to be extracted"""
    source: str = Field(description="Video ID")
    title: str = Field(description="Video title")
    description: str = Field(description="Video description")
    view_count: int = Field(description="View count")
    publish_date: str = Field(description="Publish date")
    length: int = Field(description="Video length (seconds)")
    author: str = Field(description="Author")
# Load video data
video_urls = [
    "https://www.youtube.com/watch?v=zDvnAY0zH7U",  # Shanxi Foguang Temple
    "https://www.youtube.com/watch?v=iAinNeOp6Hk",  # China's largest mansion
    "https://www.youtube.com/watch?v=gCVy6NQtk2U",  # Song Dynasty underground palace
]
# Load video metadata
videos = []
for url in video_urls:
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        docs = loader.load()
        doc = docs[0]
        videos.append(doc)
        print(f"Loaded: {doc.metadata['title']}")
    except Exception as e:
        print(f"Failed to load {url}: {str(e)}")
# Create vector store
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Chroma.from_documents(videos, embed_model)
# Configure metadata fields for the retriever
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="Video title (string)",
        type="string",
    ),
    AttributeInfo(
        name="author",
        description="Video author (string)",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="Video view count (integer)",
        type="integer",
    ),
    AttributeInfo(
        name="publish_date",
        description="Video publish date, a string in YYYY-MM-DD format",
        type="string",
    ),
    AttributeInfo(
        name="length",
        description="Video length, an integer in seconds",
        type="integer"
    ),
]
# Create SelfQueryRetriever
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)  # Deterministic output
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Video metadata including title, author, view count, publish date, and other information",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True
)
# Execute example queries
queries = [
    "Find videos with more than 100,000 views",
    "Show the most recently published video"
]
# Execute queries and output results
for query in queries:
    print(f"\nQuery: {query}")
    try:
        results = retriever.invoke(query)
        if not results:
            print("No matching videos found")
            continue
        for doc in results:
            print(f"Title: {doc.metadata['title']}")
            print(f"View count: {doc.metadata['view_count']}")
            print(f"Publish date: {doc.metadata['publish_date']}")
    except Exception as e:
        print(f"Query error: {str(e)}")
        continue
