from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
# Initialize language model and vector embedding model
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
# Prepare game knowledge text and create Document objects.
from langchain.schema import Document
game_knowledge = """
"Chronicles of the Divine: Wukong" is an action role-playing game. The game is set in an imaginary mythological world. Players will take on the role of Sun Wukong, the Great Sage Equal to Heaven, embarking on adventures in a world full of Eastern mythological elements. The game's combat system is highly distinctive, featuring a unique "Transformation System". Wukong can switch between different forms during combat. Each form has its own unique combat style and skill combinations. The Vajra form focuses on power strikes, delivering overwhelming destructive force. The Demon Buddha form focuses on magical attacks, capable of unleashing powerful magical damage. The game world is filled with iconic mythological characters; besides the protagonist Sun Wukong, there are also gods and demons from Buddhist, Taoist, and other sects. These characters may be either allies of Wukong or powerful opponents that need to be defeated. The equipment system includes a rich variety of weapon choices; besides the famous Ruyi Jingu Bang, Wukong can also use various divine artifacts and treasures. Different weapons have their own special effects, and players need to choose flexibly based on the combat scenario. The game's visuals are highly characteristic of Eastern aesthetics, with scenes blending ink painting styles that perfectly present mountains, buildings, and other elements. Combat effects incorporate both traditional Chinese cultural elements and the visual impact of modern games. In terms of difficulty design, boss battles are highly challenging, requiring players to precisely master combat rhythm and skill usage. At the same time, the game also provides multiple difficulty options to accommodate players of different skill levels.

"""
# Create Document objects
documents = [Document(page_content=game_knowledge)]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Parent chunk splitter (larger chunks)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
)
# Child chunk splitter (smaller chunks)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
)
# Create parent-child chunks
parent_docs = parent_splitter.split_documents(documents)
child_docs = child_splitter.split_documents(documents)
# Create storage and retriever, establish two-tier storage system
from langchain.retrievers import ParentDocumentRetriever # Parent document retriever
from langchain.storage import InMemoryStore # In-memory store
from langchain_community.vectorstores import Chroma # Vector store
vectorstore = Chroma(
    collection_name="game_knowledge",
    embedding_function=embed_model
)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore, # Vector store
    docstore=store, # Document store
    child_splitter=child_splitter, # Child chunk splitter
    parent_splitter=parent_splitter, # Parent chunk splitter
)
# Add chunks
retriever.add_documents(documents)
# Custom prompt template
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
prompt_template = """Answer the question based on the following context information. If you cannot find an answer, please say "I cannot find relevant information".
Context:
{context}
Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
# Create Q&A chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # Q&A chain type
    retriever=retriever,# Retriever
    return_source_documents=True, # Whether to return source documents
    chain_type_kwargs={"prompt": PROMPT}
)
# Test system through actual Q&A
test_questions = [
    "What transformation forms does Wukong have in the game?",
    "What is the visual style of the game?",
]
for question in test_questions:
    print(f"\nQuestion: {question}")
    result = qa_chain({"query": question})
    print(f"\nAnswer: {result['result']}")
    print("\nSource documents used:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\nRelated document {i}:")
        print(f"Length: {len(doc.page_content)} characters")
        print(f"Content snippet: {doc.page_content[:150]}...")
        print("---")
