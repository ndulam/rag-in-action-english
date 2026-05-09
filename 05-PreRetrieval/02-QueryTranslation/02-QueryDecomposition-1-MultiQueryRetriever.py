import logging
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever # Multi-perspective query retriever
# Configure logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
# Load game-related documents and build vector database
loader = TextLoader("90-Data/black-myth-wukong/settings.txt", encoding='utf-8')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Chroma.from_documents(documents=splits, embedding= embed_model)
# Use MultiQueryRetriever to generate multi-perspective queries
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
query = "Um, I just started playing this game and it's very hard. What is the difficulty level of this game, how many chapters are there, I can't get through the Potaraka chapter. What skills should I learn first? Beginner looking for guidance!"
# Call the retriever to perform query decomposition
docs = retriever_from_llm.invoke(query)
print(docs)
