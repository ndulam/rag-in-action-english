from openai import OpenAI
from os import getenv
# Initialize OpenAI client, pointing to DeepSeek URL
client = OpenAI(
    base_url="https://api.deepseek.com",
    api_key=getenv("DEEPSEEK_API_KEY")
)
def rewrite_query(question: str) -> str:
    """Rewrite the query using a large language model"""
    prompt = """As a game customer service representative, you need to help users rewrite their questions.

Rules:
1. Remove irrelevant information (e.g., personal situation, casual chat)
2. Use precise game terminology
3. Preserve the core intent of the question
4. Convert vague questions into specific queries
Original question: {question}
Please provide only the rewritten query (without any prefix or explanation)."""
    # Use DeepSeek model to rewrite the query
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt.format(question=question)}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()
# Start testing
query = "Um, I just started playing this game and it feels really hard. In the Potaraka chapter, I just can't get through. What skills should I learn first? Beginner looking for guidance!"
print(f"\nOriginal query: {query}")
print(f"Rewritten query: {rewrite_query(query)}")
