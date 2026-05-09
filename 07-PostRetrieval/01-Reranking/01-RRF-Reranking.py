# Import required libraries
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain.load import dumps, loads

"""
RRF (Reciprocal Rank Fusion) Reranking Algorithm Implementation

RRF is a simple yet effective algorithm for fusing results from multiple retrievals. It improves
retrieval accuracy and coverage by merging the rankings from multiple retrieval queries.

Core concept:
1. For a given user question, generate multiple queries from different perspectives
2. Perform retrieval for each query separately
3. Use the RRF algorithm to merge multiple retrieval result lists into a unified ranked list
4. RRF assigns scores to each document: score = 1/(rank + k), where rank is the document's ranking in a result list

Advantages:
- Improved retrieval coverage: multiple queries can retrieve relevant documents from different angles
- Reduced bias from single queries: merging multiple queries reduces the limitations of any single query
- Simple and efficient: low algorithm complexity, easy to implement and understand
"""

# Document directory configuration
doc_dir = "90-Data/shanxi-tourism"

def load_documents(directory):
    """
    Document loading function

    Function: Read all documents in the specified directory (supports PDF, TXT formats)

    Parameters:
        directory (str): Path to the directory containing documents

    Returns:
        list: List of loaded documents, each containing content and metadata

    Notes:
        - Iterates through all files in the directory
        - Selects the appropriate loader based on file extension
        - Supports PDF and TXT format files
        - Skips unsupported file formats
    """
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        if filename.endswith(".pdf"):
            # Use PyPDFLoader to load PDF files
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            # Use TextLoader to load TXT files
            loader = TextLoader(filepath)
        else:
            continue  # Skip unsupported file types

        # Load document and add to list
        documents.extend(loader.load())
    return documents

# Step 1: Load documents
print("Loading documents...")
docs = load_documents(doc_dir)
print(f"Successfully loaded {len(docs)} documents")

# Step 2: Text chunking (splitting)
print("\nPerforming text chunking...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # Maximum number of characters per text chunk
    chunk_overlap=50     # Number of overlapping characters between adjacent text chunks to ensure context continuity
)
splits = text_splitter.split_documents(docs)
print(f"Documents split into {len(splits)} text chunks")

# Step 3: Create vector index
print("\nCreating vector index...")
# Use HuggingFace's lightweight embedding model
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Use Chroma vector database to store document vectors
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
# Create retriever
retriever = vectorstore.as_retriever()
print("Vector index creation complete")

def reciprocal_rank_fusion(results: list[list], k=60):
    """
    RRF (Reciprocal Rank Fusion) Algorithm Implementation

    Function: Merge multiple retrieval result lists into a unified ranked list

    Parameters:
        results (list[list]): Multiple retrieval result lists, each list contains documents sorted by relevance
        k (int): Adjustment parameter for RRF algorithm, default value 60 (empirical value)

    Returns:
        list: List of (document, score) tuples merged by fusion, sorted by score in descending order

    Algorithm principle:
        1. For each document in each retrieval result list
        2. Calculate the document's RRF score: score = 1 / (rank + k)
        3. If the same document appears in multiple lists, accumulate its scores
        4. Sort all documents by their final scores

    Advantages:
        - Smaller rank (higher ranking) gives higher score
        - k parameter prevents zero denominator and adjusts the gap between different ranks
        - Documents appearing multiple times receive higher cumulative scores
    """
    print(f"RRF algorithm processing {len(results)} retrieval result lists...")

    fused_scores = {}  # Store the cumulative score for each document

    # Iterate through each retrieval result list
    for list_idx, docs in enumerate(results):
        print(f"  Processing result list {list_idx + 1}, containing {len(docs)} documents")

        # Iterate through each document in the list
        for rank, doc in enumerate(docs):
            # Serialize document to string as unique identifier
            doc_str = dumps(doc)

            # If this document appears for the first time, initialize its score
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0

            # Calculate RRF score and accumulate
            rrf_score = 1 / (rank + k)
            fused_scores[doc_str] += rrf_score

            # Debug info: show the rank and score of the document in the current list
            if rank < 3:  # Only show detailed info for the first 3 documents
                print(f"    Document {rank+1}: RRF score = 1/({rank}+{k}) = {rrf_score:.4f}")

    # Sort by score in descending order, return list of (document, score) tuples
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    print(f"RRF fusion complete, total {len(reranked_results)} unique documents")
    return reranked_results

# Step 4: Multi-query generation
print("\nConfiguring multi-query generator...")
template = """You are an assistant that helps users generate multiple search queries.

Please generate 4 different search queries from different angles based on the following question. These queries should:
1. Understand the original question from different perspectives
2. Use different keywords and expressions
3. Cover different aspects of the question

Original question: {question}

Please generate 4 related search queries:"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)
llm = ChatDeepSeek(model="deepseek-chat")

# Create query generation chain
generate_queries = (
    prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))  # Split generated queries by line
)
print("Multi-query generator configuration complete")

# Step 5: Test examples
print("\nStarting RRF reranking test...")
questions = [
    "What are the famous tourist attractions in Shanxi?",
    "What is the historical background of Yungang Grottoes?",
    "What is the cultural and religious significance of Mount Wutai?"
]

# Perform RRF retrieval and reranking for each question
for idx, question in enumerate(questions, 1):
    print(f"\n{'='*50}")
    print(f"Question {idx}: {question}")
    print('='*50)

    # Step 1: Generate multiple queries
    print("\n1. Generating multiple related queries...")
    queries = generate_queries.invoke({"question": question})
    # Filter empty queries
    queries = [q.strip() for q in queries if q.strip()]
    print(f"Generated {len(queries)} queries:")
    for i, query in enumerate(queries, 1):
        print(f"  Query {i}: {query}")

    # Step 2: Retrieve for each query
    print(f"\n2. Performing vector retrieval for each query...")
    all_results = []
    for i, query in enumerate(queries, 1):
        print(f"  Retrieving query {i}: {query}")
        docs = retriever.invoke(query)
        all_results.append(docs)
        print(f"    Retrieved {len(docs)} related documents")

    # Step 3: Merge results using RRF algorithm
    print(f"\n3. Merging retrieval results using RRF algorithm...")
    reranked_docs = reciprocal_rank_fusion(all_results)

    # Step 4: Show final results
    print(f"\n4. Final RRF reranked results (showing top 3):")
    print(f"Total of {len(reranked_docs)} unique documents merged")

    for i, (doc, score) in enumerate(reranked_docs[:3], 1):
        print(f"\nRank {i} (RRF score: {score:.4f}):")
        # Truncate to first 200 characters to avoid overly long output
        content_preview = doc.page_content[:200].replace('\n', ' ').strip()
        print(f"   Content preview: {content_preview}...")

        # Show document source information (if available)
        if hasattr(doc, 'metadata') and doc.metadata:
            source = doc.metadata.get('source', 'Unknown source')
            print(f"   Source: {source}")

print(f"\nRRF reranking test complete!")
print("\nRRF Algorithm Summary:")
print("- Multi-angle query generation: understand user questions from different perspectives")
print("- Multi-retrieval result fusion: integrate the advantages of multiple retrieval results")
print("- Ranking optimization: reorder documents using the RRF algorithm")
print("- Improved recall rate: reduce omissions from single queries")
print("- Enhanced relevance: documents appearing multiple times receive higher weight")
