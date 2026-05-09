from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain.chains import RetrievalQA
# System documents: focus on specific game mechanics and systems
system_docs = [
    '"Chronicles of the Divine: Wukong" uses a unique transformation system as the core combat mechanism',
    "In Vajra form, heavy weapons can be used, increasing attack power and defense",
    "The Demon Buddha form focuses on magical attacks and can release powerful magical damage",
    "Different forms can be switched freely during combat to create combos",
    "The game has three difficulty levels: Normal, Hard, and Shura"
]
# Lore documents: focus on story and background settings
lore_docs = [
    "The game is set in an imaginary mythological world, incorporating Eastern mythological elements",
    "Sun Wukong in the game was sealed for 500 years before awakening again",
    "The world contains gods and demons from multiple factions including Buddhism and Taoism",
    "The player-controlled Sun Wukong needs to seek the truth among various factions",
    "The game scenes include ink-painting-style mountains and buildings"
]
# Create two different retrievers: BM25 + vector retriever
from langchain_community.retrievers import BM25Retriever # BM25 retriever
from langchain_community.vectorstores import FAISS # Vector database, not a retriever at this point
from langchain.retrievers import EnsembleRetriever # Ensemble/hybrid retriever
# Create BM25 retriever
bm25_retriever = BM25Retriever.from_texts(
    system_docs + lore_docs,
    metadatas=[{"source": "system" if i < len(system_docs) else "lore"}
               for i in range(len(system_docs) + len(lore_docs))]
)
bm25_retriever.k = 2
# Create vector retriever
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
vectorstore = FAISS.from_texts(
    system_docs + lore_docs,
    embed_model,
    metadatas=[{"source": "system" if i < len(system_docs) else "lore"}
               for i in range(len(system_docs) + len(lore_docs))]
)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Create vector retriever based on vector database
# Create ensemble/hybrid retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], # Hybrid retriever containing two retrievers
    weights=[0.5, 0.5] # Weights to balance the contributions of the two retrievers -> weighted reranker
)
# Create Q&A chain using hybrid retriever and Q&A chain using single retriever (for comparison)
llm = ChatDeepSeek(model="deepseek-chat")
# Create hybrid retrieval Q&A chain -> somewhat like an ensemble learning approach
ensemble_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=ensemble_retriever,
    return_source_documents=True
)
# Create standalone vector retrieval Q&A chain (for comparison)
vector_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=faiss_retriever,
    return_source_documents=True
)
# Test different types of queries
test_queries = [
    "What is the transformation system like in the game?",  # System mechanics query
    "What is the world background of the game?",            # Background settings query
    "What combat forms does Wukong have?"                   # Mixed query
]
for query in test_queries:
    print(f"\nQuery: {query}")
    print("\n1. Hybrid retrieval results:")
    ensemble_docs = ensemble_retriever.invoke(query)
    print("Retrieved documents:")
    for i, doc in enumerate(ensemble_docs, 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")
    print("\n2. Vector retrieval results (for comparison):")
    vector_docs = faiss_retriever.invoke(query)
    print("Retrieved documents:")
    for i, doc in enumerate(vector_docs, 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")
# Test Q&A effectiveness
print("\n=== Q&A Effectiveness Test ===")
test_questions = [
    "What are the characteristics of the Vajra form?",
    "How are the factions distributed in the game?",
]
for question in test_questions:
    print(f"\nQuestion: {question}")
    print("\n1. Answer using hybrid retrieval:")
    ensemble_result = ensemble_qa.invoke({"query": question})
    print(f"Answer: {ensemble_result['result']}")
    print("\nSource documents used:")
    for i, doc in enumerate(ensemble_result['source_documents'], 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")
    print("\n2. Answer using pure vector retrieval (for comparison):")
    vector_result = vector_qa.invoke({"query": question})
    print(f"Answer: {vector_result['result']}")
    print("\nSource documents used:")
    for i, doc in enumerate(vector_result['source_documents'], 1):
        print(f"{i}. [{doc.metadata['source']}] {doc.page_content}")
