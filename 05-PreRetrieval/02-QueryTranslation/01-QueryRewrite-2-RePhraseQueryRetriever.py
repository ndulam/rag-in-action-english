import logging
from langchain.retrievers import RePhraseQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Configure logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)
# Load game document data
loader = TextLoader("90-Data/black-myth-wukong/settings.txt", encoding='utf-8')
data = loader.load()
# Text chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
# Create vector store
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = Chroma.from_documents(documents=all_splits, embedding= embed_model)
# Set up RePhraseQueryRetriever
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
retriever_from_llm = RePhraseQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm # Use DeepSeek model as the rewriter
)
# Sample input: game-related query
query = "Um, I just started playing this game and it feels really hard. In the Potaraka chapter, I just can't get through. What skills should I learn first? Beginner looking for guidance!"
# Call RePhraseQueryRetriever for query rewriting
docs = retriever_from_llm.invoke(query)
print(docs)
