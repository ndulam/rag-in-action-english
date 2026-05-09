from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_deepseek import ChatDeepSeek

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source"""
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given the user question, select the most appropriate data source to answer it",
    )

def create_router():
    """Create and return the routing model"""
    # LLM with function calling
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)

    # Prompt template
    system = """You are an expert at routing user questions to the appropriate data source.
Based on the programming language the question pertains to, route it to the relevant data source."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

    # Define the router
    return prompt | structured_llm

def route_question(question: str) -> str:
    """Route the user question to the appropriate data source"""
    router = create_router()
    result = router.invoke({"question": question})
    return result.datasource

# Usage example
if __name__ == "__main__":
    # Test question
    test_question = "What is the difference between lists and tuples in Python?"
    result = route_question(test_question)
    print(f"Question: {test_question}")
    print(f"Routing result: {result}")
