# 1. Load documents
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Black_Myth:_Wukong",)
)
docs = loader.load()

# 2. Split documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. Set up the embedding model
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. Create the vector store
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 5. Define the RAG prompt
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

# 6. Define the application state
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 7. Define the retrieval step
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# 8. Define the generation step
def generate(state: State):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# 9. Build and compile the application
from langgraph.graph import START, StateGraph # pip install langgraph
graph = (
    StateGraph(State)
    .add_sequence([retrieve, generate])
    .add_edge(START, "retrieve")
    .compile()
)

# 10. Run the query
question = "What game scenes are available in Black Myth: Wukong?"
response = graph.invoke({"question": question})
print(f"\nQuestion: {question}")
print(f"Answer: {response['answer']}")
