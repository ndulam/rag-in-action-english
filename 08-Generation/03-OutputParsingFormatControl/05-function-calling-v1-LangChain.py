from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

# Define tool schema
class get_weather(BaseModel):
    """Get weather information"""
    location: str = Field(..., description="City name")
    temperature: float = Field(..., description="Temperature")

# Initialize LLM
llm = ChatDeepSeek(model="deepseek-chat")

# Bind tools
llm_with_tools = llm.bind_tools([get_weather])

# Send request
response = llm_with_tools.invoke("Please tell me the weather in Shanghai")

# Parse output
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool name: {tool_call['name']}")
        print(f"Arguments: {tool_call['args']}")
else:
    print("No tool calls made")
