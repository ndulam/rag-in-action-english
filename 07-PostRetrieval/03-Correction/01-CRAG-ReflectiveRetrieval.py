"""
CRAG (Corrective Retrieval-Augmented Generation) Reflective Retrieval System

CRAG is an improved RAG method that enhances retrieval quality through the following steps:
1. Retrieve: retrieve relevant documents from the vector database
2. Grade: evaluate the relevance of retrieved documents
3. Decide: based on grading results, decide whether to generate an answer directly or rewrite the query
4. Correct: if documents are irrelevant, rewrite the query and perform a web search
5. Generate: generate the final answer based on filtered relevant documents

This approach automatically detects and corrects inaccurate retrieval results, improving the reliability of RAG systems.
"""

# ================================
# Part 1: Data Preparation and Vector Database Construction
# ================================

#1 Create an index for 3 blog articles
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables (including OpenAI API key, etc.)
load_dotenv()

# Define the blog article URLs to index
# These are technical blogs about AI agents, prompt engineering, and adversarial attacks
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",        # AI agents related
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",  # Prompt engineering
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",      # LLM adversarial attacks
]

# Load the content of each URL using WebBaseLoader
docs = [WebBaseLoader(url).load() for url in urls]
# Flatten the nested list into a single document list
docs_list = [item for sublist in docs for item in sublist]

# Create a text splitter using tiktoken encoder to accurately count tokens
# chunk_size=250: each document chunk has at most 250 tokens
# chunk_overlap=0: no overlap between chunks, avoiding duplicate information
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
# Split documents into small chunks for retrieval and processing
doc_splits = text_splitter.split_documents(docs_list)

# Create vector database
# Use Chroma as the vector store, with OpenAI's embedding model for vectorization
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",  # Collection name
    embedding=OpenAIEmbeddings(),  # Use OpenAI's text-embedding-ada-002 model
)
# Convert the vector store to a retriever for subsequent similarity searches
retriever = vectorstore.as_retriever()

# ================================
# Part 2: Retrieval Grader - Core Component of CRAG
# ================================

#2 Retrieval grader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define the data model for grading results
# Use Pydantic to ensure output format consistency and type safety
class GradeDocuments(BaseModel):
    """Binary grading of the relevance of retrieved documents.

    This class defines the output format for document relevance grading,
    ensuring the model returns only a clear 'yes' or 'no' judgment.
    """

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Create language model with structured output
# temperature=0.5: moderate randomness, balancing consistency and creativity
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
# Restrict model output to GradeDocuments format
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Build the grading prompt template
# The system prompt defines the grader's role and grading criteria
system = """You are a grader assessing the relevance of a retrieved document to a user question. \n
    If the document contains keywords or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Create the retrieval grading chain: prompt template + structured language model
retrieval_grader = grade_prompt | structured_llm_grader

# Test the grader
question = "agent memory"  # Test question: about agent memory
docs = retriever.get_relevant_documents(question)  # Retrieve relevant documents
doc_txt = docs[1].page_content  # Get the content of the second document
# Print the grading result to verify the grader is working properly
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

# ================================
# Part 3: RAG Generation Chain
# ================================

#3 Set up the generation model
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Get a pre-built RAG prompt template from LangChain Hub
# This template is specifically designed for answering questions based on context
prompt = hub.pull("rlm/rag-prompt")

# Create the language model for generating answers
# temperature=0: ensures consistency and reproducibility of output
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Document formatting function
def format_docs(docs):
    """Format a list of documents into a single string.

    Args:
        docs: List of document objects

    Returns:
        str: Document content joined by double newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG generation chain: prompt template + language model + string parser
rag_chain = prompt | llm | StrOutputParser()

# Test the generation chain
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

# ================================
# Part 4: Query Rewriter
# ================================

#4 Set up the question rewriter
# Create the language model for query rewriting
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

# System prompt for query rewriting
# The goal is to rewrite vague or inaccurate queries into a form better suited for searching
system = """You are a question re-writer that converts an input question to a better version optimized for web search. \n
     Look at the input and try to reason about the underlying semantic intent / meaning."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

# Create the query rewriting chain
question_rewriter = re_write_prompt | llm | StrOutputParser()
# Test the query rewriting functionality
question_rewriter.invoke({"question": question})

# ================================
# Part 5: Web Search Tool
# ================================

#5 Set up web search tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Create the web search tool
# k=3: return at most 3 search results
web_search_tool = TavilySearchResults(k=3)

# ================================
# Part 6: Graph State Definition
# ================================

#6 Set up imports required for CRAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough

#7 Define graph state
from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of the CRAG workflow graph.

    This state is passed throughout the CRAG process, containing all key information during processing.

    Attributes:
        question: The user's original question or rewritten question
        generation: The final answer generated by the language model
        web_search: Flag indicating whether a web search is needed ("Yes"/"No")
        documents: List of retrieved documents (original retrieval results or web search results)
    """

    question: str        # Current question being processed
    generation: str      # Generated answer
    web_search: str      # Flag indicating whether a web search is needed
    documents: List[str] # List of documents

# ================================
# Part 7: CRAG Workflow Node Functions
# ================================

from langchain.schema import Document

def retrieve(state):
    """
    Retrieve node: retrieve relevant documents from the vector database.

    This is the first step in the CRAG process, retrieving potentially relevant documents based on the user's question.

    Args:
        state (dict): Current graph state, must contain the 'question' key

    Returns:
        state (dict): Updated state with the 'documents' key added
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Use the vector retriever to get relevant documents
    # Returns the most semantically similar document chunks
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate node: generate an answer based on retrieved documents.

    This is the final step in the CRAG process, using filtered relevant documents to generate the final answer.

    Args:
        state (dict): Current graph state, contains question and documents

    Returns:
        state (dict): Updated state with the generation key added
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # Use the RAG chain to generate an answer
    # Use documents as context, combined with the question to generate a response
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Document grading node: evaluate the relevance of retrieved documents.

    This is the core innovation of CRAG, using an LLM to evaluate whether each retrieved document is truly relevant.
    Only relevant documents are kept; if no relevant documents exist, web search is flagged.

    Args:
        state (dict): Current graph state

    Returns:
        state (dict): Updates documents to filtered relevant documents, sets the web_search flag
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Initialize filtering results
    filtered_docs = []           # Store relevant documents
    web_search = "No"            # Default: no web search needed
    has_relevant_docs = False    # Flag indicating whether relevant documents were found

    # Grade each retrieved document for relevance
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score

        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)    # Keep relevant document
            has_relevant_docs = True   # Mark that a relevant document was found
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue  # Skip irrelevant documents

    # Key CRAG logic: only perform web search if no relevant documents are found
    # This avoids unnecessary web searches and improves efficiency
    if not has_relevant_docs:
        web_search = "Yes"

    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query(state):
    """
    Query transformation node: rewrite the query to improve search quality.

    When all retrieved documents are irrelevant, rewrite the original query to get better search results.

    Args:
        state (dict): Current graph state

    Returns:
        state (dict): Updates the question key with the rewritten question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Use the query rewriter to generate an improved question
    # The rewritten question is usually more specific and better suited for searching
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state):
    """
    Web search node: obtain external information as a supplement.

    When the local document repository cannot provide relevant information, use web search to obtain additional information.

    Args:
        state (dict): Contains the current state
            - question: the question (possibly rewritten)
            - documents: list of documents

    Returns:
        state (dict): Appends web search results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Use the Tavily search tool to perform a web search
    search_results = web_search_tool.invoke(question)

    # Format search results as document objects
    # This keeps the format consistent with locally retrieved documents
    search_results_str = "\n".join([str(result) for result in search_results])
    web_results = Document(page_content=search_results_str)
    documents.append(web_results)

    return {"documents": documents, "question": question}

# ================================
# Part 8: Conditional Edge Logic
# ================================

### Edge handler functions

def decide_to_generate(state):
    """
    Decision node: determine the next action.

    This is the key decision point in the CRAG workflow:
    - If relevant documents exist: generate answer directly
    - If no relevant documents exist: transform query and perform web search

    Args:
        state (dict): Current graph state

    Returns:
        str: Name of the next node to execute
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]  # Get the flag indicating whether web search is needed
    state["documents"]

    if web_search == "Yes":
        # All local documents were graded as irrelevant
        # Need to rewrite the query and perform a web search to get better information
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # Relevant documents found, can generate answer directly
        print("---DECISION: GENERATE---")
        return "generate"

# ================================
# Part 9: Build and Compile the CRAG Workflow Graph
# ================================

#8 Compile the graph
from langgraph.graph import END, StateGraph, START

# Create the state graph workflow
workflow = StateGraph(GraphState)

# Add all nodes to the workflow graph
workflow.add_node("retrieve", retrieve)              # Retrieve node
workflow.add_node("grade_documents", grade_documents) # Document grading node
workflow.add_node("generate", generate)              # Answer generation node
workflow.add_node("transform_query", transform_query) # Query transformation node
workflow.add_node("web_search_node", web_search)     # Web search node

# Build the edge connections in the workflow graph
# Define the execution order and conditional jumps between nodes

# 1. From start node to retrieve node
workflow.add_edge(START, "retrieve")

# 2. From retrieve to document grading
workflow.add_edge("retrieve", "grade_documents")

# 3. From document grading to conditional branch
# Select the next node based on the return value of decide_to_generate
workflow.add_conditional_edges(
    "grade_documents",           # Source node
    decide_to_generate,          # Decision function
    {
        "transform_query": "transform_query",  # If web search is needed
        "generate": "generate",                # If relevant documents exist
    },
)

# 4. After query transformation, perform web search
workflow.add_edge("transform_query", "web_search_node")

# 5. After web search, generate answer
workflow.add_edge("web_search_node", "generate")

# 6. After generating answer, end
workflow.add_edge("generate", END)

# Compile the workflow graph into an executable application
app = workflow.compile()

# ================================
# Part 10: Run the CRAG System
# ================================

#9 Use the graph to answer a question

from pprint import pprint

# Prepare the input question
# First question: about types of agent memory (English)
inputs = {"question": "What are the types of agent memory?"}

# Second question example (commented out)
# inputs = {"question": "Why is Shanxi Province rich in tourism resources?"}

print("=== CRAG Workflow Execution Process ===")

# Stream the CRAG workflow execution
# The stream method lets us observe each node's execution process
for output in app.stream(inputs):
    for key, value in output.items():
        # Print the name of the currently executing node
        pprint(f"Node '{key}':")

        # Optional: print the full state information for each node
        # Useful for debugging and understanding the workflow
        # pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

print("=== Final Generation Result ===")
# Print the final generated answer
pprint(value["generation"])

"""
CRAG Workflow Summary:

1. Retrieve: retrieve candidate documents from the vector database
2. Grade (grade_documents): use LLM to evaluate document relevance
3. Decide (decide_to_generate): choose a path based on grading results
4a. Direct path: if relevant documents exist → generate answer
4b. Corrective path: if no relevant documents → transform query → web search → generate answer

This design ensures the system can automatically detect and correct retrieval errors,
significantly improving the accuracy and reliability of the RAG system.
"""
