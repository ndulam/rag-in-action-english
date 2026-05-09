import os
import getpass
from typing import Annotated, Sequence, TypedDict, List, Literal
from pprint import pprint

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

# ----- 1. Set API Keys -----
def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}: ")
_set_env("OPENAI_API_KEY")

# ----- 2. Build retriever and create retrieval tool -----
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    # "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
# Load documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
# Split into chunks
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=50)
doc_splits = splitter.split_documents(docs_list)
# Vector store
vectorstore = Chroma.from_documents(documents=doc_splits,
                                   collection_name="rag-chroma",
                                   embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
# Create retrieval tool
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information from Lilian Weng's blog about agents, prompt engineering, and adversarial attacks."
)
tools = [retriever_tool]

# ----- 3. Define Agent State -----
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    retrieval_done: bool
    graded: bool
    grade_result: str

# ----- 4. Define Node Functions -----
# Document relevance grading node
def grade_documents(state: AgentState) -> AgentState:
    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score: 'yes' or 'no'.")
    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    grader = PromptTemplate(
        template="""
You are a relevance grader.
Retrieved document:\n{context}\nUser question: {question}\nReturn 'yes' if relevant, otherwise 'no'.""",
        input_variables=["context","question"]
    ) | model.with_structured_output(Grade)
    msgs = state['messages']
    question = msgs[0].content
    docs = msgs[-1].content
    result = grader.invoke({"question": question, "context": docs})
    # Save grading result
    return {
        "messages": msgs,
        "retrieval_done": True,
        "graded": True,
        "grade_result": "generate" if result.binary_score == "yes" else "rewrite"
    }

# Conditional routing: evaluate document relevance
def route_after_grading(state: AgentState) -> str:
    if state.get("grade_result") == "generate":
        return "generate"
    else:
        return "rewrite"

# Custom retrieval node, merges historical messages
def retrieve(state: AgentState) -> AgentState:
    msgs = state['messages']
    tool = tools[0]  # Only one retrieval tool here
    question = msgs[0].content
    docs = tool.invoke(question)
    retrieval_msg = HumanMessage(content=docs)
    return {
        "messages": msgs + [retrieval_msg],
        "retrieval_done": True,
        "graded": False,
        "grade_result": ""
    }

# Agent decision node
def agent(state: AgentState) -> AgentState:
    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    model = model.bind_tools(tools)
    msgs = state.get('messages') or []
    if not msgs:
        raise ValueError("Message list is empty when invoking agent node; cannot generate a reply. Check upstream node output.")

    # Add system message to guide retrieval tool usage
    system_msg = HumanMessage(content="Please use the retrieval tool to answer the question.")
    response = model.invoke([system_msg] + msgs)
    return {
        "messages": msgs + [response],
        "retrieval_done": False,
        "graded": False,
        "grade_result": ""
    }

# Conditional routing: decide whether to use tools or end
def should_use_tools(state: AgentState) -> str:
    msgs = state['messages']
    last_msg = msgs[-1]
    # Check if message contains a tool call or requires retrieval
    if (hasattr(last_msg, "tool_calls") and last_msg.tool_calls) or \
       (isinstance(last_msg.content, str) and "retrieve" in last_msg.content.lower()):
        return "retrieve"
    return "end"

# Query rewrite node
def rewrite(state: AgentState) -> AgentState:
    msgs = state['messages']
    question = msgs[0].content
    prompt = HumanMessage(content=f"Rewrite the following question to better retrieve documents:\n{question}\n")
    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    resp = model.invoke([prompt])
    return {
        "messages": [resp],  # Reset messages, keeping only the new question
        "retrieval_done": False,
        "graded": False,
        "grade_result": ""
    }

# Answer generation node
def generate(state: AgentState) -> AgentState:
    msgs = state['messages']
    question = msgs[0].content
    docs = msgs[-1].content
    rag_prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": docs, "question": question})
    return {
        "messages": msgs + [HumanMessage(content=answer)],
        "retrieval_done": True,
        "graded": True,
        "grade_result": "generate"
    }

# ----- 5. Build and compile graph workflow -----
wf = StateGraph(AgentState)
wf.add_node("agent", agent)
wf.add_node("retrieve", retrieve)
wf.add_node("grade_documents", grade_documents)
wf.add_node("rewrite", rewrite)
wf.add_node("generate", generate)
# Connect nodes
wf.add_edge(START, "agent")
wf.add_conditional_edges("agent", should_use_tools, {"retrieve": "retrieve", "end": END})
wf.add_edge("retrieve", "grade_documents")
wf.add_conditional_edges("grade_documents", route_after_grading, {"generate": "generate", "rewrite": "rewrite"})
wf.add_edge("generate", END)
wf.add_edge("rewrite", "agent")

app = wf.compile()
# try:
#     from langchain_core.runnables.graph import MermaidDrawMethod
#     png_data = app.get_graph(xray=True).draw_mermaid_png(
#         draw_method=MermaidDrawMethod.PYPPETEER
#     )
#     with open("10-AdvanceRAG/04-AgenticRAG/AgenticRAG-Graph.png", "wb") as f:
#         f.write(png_data)
#     print("Saved as: AgenticRAG-Graph.png")
# except Exception as e:
#     print(f"Error saving graph image: {e}")

# ----- 6. Run example -----
# Initialize input message using HumanMessage type
from langchain_core.messages import HumanMessage
inputs = {
    "messages": [
        HumanMessage(content="What types of memory do agents have?")
    ],
    "retrieval_done": False,
    "graded": False,
    "grade_result": ""
}

# Run and print output from each node
final_output = None
for output in app.stream(inputs):
    print("\n=== Node Output ===")
    for node_name, state in output.items():
        print(f"\nNode name: {node_name}")
        if state and "messages" in state:
            print("Latest message:", state["messages"][-1].content)
        print(f"Retrieval status: {state.get('retrieval_done')}")
        print(f"Grading status: {state.get('graded')}")
        print(f"Grade result: {state.get('grade_result')}")
    print("===============")
    final_output = output

# Print final answer
if final_output:
    # Check all possible final nodes
    final_state = None
    for node in ["generate", "agent"]:
        if node in final_output:
            final_state = final_output[node]
            break

    if final_state and "messages" in final_state:
        print("\n=== Final Answer ===")
        print(final_state["messages"][-1].content)
        print("===============")
    else:
        print("\n=== Error: No messages found in state ===")
else:
    print("\n=== Error: Failed to obtain final output ===")
