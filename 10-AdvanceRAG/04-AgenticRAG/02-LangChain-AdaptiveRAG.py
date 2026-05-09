import os
import getpass
from typing import Literal, List
from pprint import pprint

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START

# ----- 0. Install dependencies -----
# pip install -U langchain_community tiktoken langchain-openai langchain-cohere \
#         langchainhub chromadb langchain langgraph tavily-python

# ----- 1. Set API Keys -----
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("COHERE_API_KEY")
_set_env("TAVILY_API_KEY")

# ----- 2. Build vector index -----
# 2.1 Embedding model
embd = OpenAIEmbeddings()
# 2.2 Document source URLs
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    # "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
# 2.3 Load and split documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [d for sub in docs for d in sub]
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
doc_splits = splitter.split_documents(docs_list)
# 2.4 Add to Chroma vector store
vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=embd)
retriever = vectorstore.as_retriever()

# ----- 3. Query routing model -----
class RouteQuery(BaseModel):
    """Route the question to vector store or web search."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ..., description="Select 'vectorstore' or 'web_search' based on the question."
    )

llm_router = ChatOpenAI(model="gpt-4o", temperature=0)
structured_router = llm_router.with_structured_output(RouteQuery)
route_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query routing expert. The vector store contains documents related to agents, prompt engineering, and adversarial attacks."),
    ("human", "{question}")
])
question_router = route_prompt | structured_router

# ----- 4. LLM grader: document relevance -----
class GradeDocuments(BaseModel):
    """Relevance score for retrieved documents: 'yes' or 'no'."""
    binary_score: str = Field(description="Relevant: 'yes' or 'no'.")

llm_doc_grader = ChatOpenAI(model="gpt-4o", temperature=0)
structured_doc_grader = llm_doc_grader.with_structured_output(GradeDocuments)
doc_grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance grader. A document is relevant if it contains keywords or semantics related to the question. Return 'yes' or 'no'."),
    ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}")
])
retrieval_grader = doc_grade_prompt | structured_doc_grader

# ----- 5. LLM grader: hallucination detection -----
class GradeHallucinations(BaseModel):
    """Detect whether the answer is grounded in facts: 'yes' or 'no'."""
    binary_score: str = Field(description="Grounded in facts: 'yes' or 'no'.")

llm_hallu_grader = ChatOpenAI(model="gpt-4o", temperature=0)
structured_hallu = llm_hallu_grader.with_structured_output(GradeHallucinations)
hallu_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a hallucination detector. Determine whether the answer is grounded in the provided facts. Return 'yes' or 'no'."),
    ("human", "Set of facts:\n\n{documents}\n\nLLM generation: {generation}")
])
hallucination_grader = hallu_prompt | structured_hallu

# ----- 6. LLM grader: answer completeness -----
class GradeAnswer(BaseModel):
    """Determine whether the answer addresses the question: 'yes' or 'no'."""
    binary_score: str = Field(description="Answer is complete: 'yes' or 'no'.")

llm_ans_grader = ChatOpenAI(model="gpt-4o", temperature=0)
structured_ans = llm_ans_grader.with_structured_output(GradeAnswer)
ans_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an answer quality grader. Determine whether the answer resolves the question. Return 'yes' or 'no'."),
    ("human", "User question:\n\n{question}\n\nLLM generation: {generation}")
])
answer_grader = ans_prompt | structured_ans

# ----- 7. Question rewriter -----
from langchain.schema import Document

question_rewriter_llm = ChatOpenAI(model="gpt-4o", temperature=0)
re_write_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the input question into a version better suited for vector retrieval, preserving semantic intent."),
    ("human", "Here is the initial question:\n\n{question}\nFormulate an improved question.")
])
from langchain_core.output_parsers import StrOutputParser
question_rewriter = re_write_prompt | question_rewriter_llm | StrOutputParser()

# ----- 8. Web search tool -----
web_search_tool = TavilySearchResults(k=3)

# ----- 9. Build graph state and workflow nodes -----
class GraphState(BaseModel):
    question: str
    generation: str = ""
    documents: List[Document] = []

    class Config:
        arbitrary_types_allowed = True

# Retrieval node
def retrieve_node(state: GraphState):
    question = state.question
    docs = retriever.get_relevant_documents(question)
    return {"documents": docs, "question": question}

# Document grading and filtering
def grade_docs_node(state: GraphState):
    question = state.question
    docs = state.documents
    filtered = []
    for d in docs:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            filtered.append(d)
    return {"documents": filtered, "question": question}

# Query rewrite node
def transform_query_node(state: GraphState):
    better_q = question_rewriter.invoke({"question": state.question})
    return {"documents": state.documents, "question": better_q}

# Web search node
def web_search_node(state: GraphState):
    results = web_search_tool.invoke({"query": state.question})
    content = "\n".join([r["content"] for r in results])
    return {"documents": [Document(page_content=content)], "question": state.question}

# Answer generation node
prompt = hub.pull("rlm/rag-prompt")
rag_chain = prompt | ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) | StrOutputParser()

def generate_node(state: GraphState):
    docs = state.documents
    if not docs:
        gen = "No relevant documents retrieved; unable to generate an answer."
    else:
        gen = rag_chain.invoke({"context": docs, "question": state.question})
    return {"generation": gen, "documents": docs, "question": state.question}

# Hallucination and answer evaluation node
def grade_generation_node(state: GraphState):
    h_score = hallucination_grader.invoke({"documents": state.documents, "generation": state.generation})
    if h_score.binary_score != "yes":
        return {"decision": "retry"}
    a_score = answer_grader.invoke({"question": state.question, "generation": state.generation})
    return {"decision": "end" if a_score.binary_score == "yes" else "rewrite"}

# ----- 10. Define and compile workflow -----
wf = StateGraph(GraphState)
wf.add_node("retrieve", retrieve_node)
wf.add_node("grade_documents", grade_docs_node)
wf.add_node("transform_query", transform_query_node)
wf.add_node("web_search", web_search_node)
wf.add_node("generate", generate_node)

# Routing conditions
wf.add_conditional_edges(
    START,
    lambda s: question_router.invoke({"question": s.question}).datasource,
    {"vectorstore": "retrieve", "web_search": "web_search"}
)
# Filter or rewrite after retrieval
wf.add_edge("retrieve", "grade_documents")
wf.add_conditional_edges(
    "grade_documents",
    lambda s: "transform_query" if not s.documents else "generate",
    {"transform_query": "transform_query", "generate": "generate"}
)
# Return to retrieval after rewrite
wf.add_edge("transform_query", "retrieve")
# Web search goes directly to generation
wf.add_edge("web_search", "generate")
# Evaluate after generation
wf.add_conditional_edges(
    "generate",
    lambda s: grade_generation_node(s)["decision"],
    {"retry": "generate", "rewrite": "transform_query", "end": END}
)

app = wf.compile()
try:
    from langchain_core.runnables.graph import MermaidDrawMethod
    png_data = app.get_graph(xray=True).draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER
    )
    with open("10-AdvanceRAG/04-AgenticRAG/AdaptiveRAG-Graph.png", "wb") as f:
        f.write(png_data)
    print("Saved as: AdaptiveRAG-Graph.png")
except Exception as e:
    print(f"Error saving graph image: {e}")


# ----- 11. Run example -----
for q in [
    # "Who is the president of the United States?",
    "What types of memory do agents have?"
]:
    print("\n" + "="*50)
    print(f"Question: {q}")
    print("="*50)

    result = app.invoke({"question": q})
    print("\n[Generated Answer]")
    print("-"*30)
    print(result["generation"])
    print("-"*30)

    print("\n" + "="*50 + "\n")
