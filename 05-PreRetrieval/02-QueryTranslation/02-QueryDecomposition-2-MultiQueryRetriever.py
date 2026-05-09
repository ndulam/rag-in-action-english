import logging
from typing import List
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import BaseOutputParser
from langchain.prompts import PromptTemplate
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
# Custom output parser
class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Filter empty lines
output_parser = LineListOutputParser()
# Custom query prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an experienced game customer service representative. Please rewrite the user's query from 5 different perspectives to help players get more detailed game guidance.
                Please ensure each query focuses on a different aspect, such as skill selection, combat strategy, equipment pairing, etc.
                User's original question: {question}
                Please provide 5 different queries, one per line.""",
)
# Set up the large model processing pipeline
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
llm_chain = QUERY_PROMPT | llm | output_parser
# Use MultiQueryRetriever with a custom prompt template
retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(),
    llm_chain=llm_chain,
    parser_key="lines"
)
# Perform multi-perspective query
query = "Um, I just started playing this game and it's very hard. What is the difficulty level of this game, how many chapters are there, I can't get through the Potaraka chapter. What skills should I learn first? Beginner looking for guidance!"
# Call the retriever to perform query decomposition
docs = retriever.invoke(query)
print(docs)
