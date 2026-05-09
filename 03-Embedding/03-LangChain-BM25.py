import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma # pip install chromadb
from langchain_community.retrievers import BM25Retriever # pip install rank_bm25

battle_logs = [
    "The Monkey King wears chain mail armor.",
    "The Monkey King encountered a monster in Wuhui Valley; the monster attacked, and the Monkey King blocked with the Bronze Cloud Staff.",
    "The Monkey King unleashed the Flame Fist to repel the monster, then activated the Iron Body to withstand the divine weapon attack.",
    "The monster used Ice Arrows to attack the Monkey King but was repelled and defeated by the Flame Fist.",
    "The Monkey King summoned the Flame Fist and Devastating Roar to defeat the monster, then collected the monster's essence."
]
request = "What equipment and moves does the Monkey King have?"

bm25_retriever = BM25Retriever.from_texts(battle_logs)
bm25_response = bm25_retriever.invoke(request)
print(f"BM25 retrieval results:\n{bm25_response}")

docs = [Document(page_content=log) for log in battle_logs]
load_dotenv()
chroma_vs = Chroma.from_documents(
    docs,
    embedding=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("O3_API_KEY"),
        base_url=os.getenv("O3_BASE_URL")
        )
    )
chroma_retriever = chroma_vs.as_retriever()
chroma_response = chroma_retriever.invoke(request)
print(f"Chroma retrieval results:\n{chroma_response}")

# hybrid_response = list({doc.page_content for doc in bm25_response}) # missing chain mail
# hybrid_response = list({doc.page_content for doc in chroma_response}) # missing Bronze Cloud Staff
hybrid_response = list({doc.page_content for doc in bm25_response + chroma_response})
print(f"Hybrid retrieval results:\n{hybrid_response}")
prompt = ChatPromptTemplate.from_template("""
                Based on the following context, answer the question. If the context does not contain relevant information,
                please say "I cannot find relevant information from the provided context."
                Context: {context}
                Question: {question}
                Answer:"""
                                          )
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("O3_API_KEY"),
    base_url=os.getenv("O3_BASE_URL"))
doc_content = "\n\n".join(hybrid_response)
answer = llm.invoke(prompt.format(question=request, context=doc_content))
print(f"LLM answer:\n{answer.content}")
