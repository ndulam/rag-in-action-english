from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

def get_code_snippet() -> str:
    """
    Retrieve the code snippet to be analyzed.
    Returns:
        str: A string containing the code snippet
    """
    return """
            def handle_request(request):
                # Check if token is present in request headers
                if 'token' not in request.headers:
                    return {'status': 401, 'message': 'Unauthorized'}, 401
                
                try:
                    # Check user permissions
                    check_permission(request.headers['token'])

                    # Process request logic
                    return process_request(request)
                    
                except AccessDenied:
                    return {'status': 403, 'message': 'Forbidden'}, 403
                except Exception as e:
                        return {'status': 500, 'message': str(e)}, 500
            """

client = OpenAI(base_url="https://api.deepseek.com",
                api_key=os.getenv("DEEPSEEK_API_KEY"))

retrieved_content = get_code_snippet()

question = f"""
Please describe the possible error handling mechanisms based on the following code snippet:
{retrieved_content}
Note: Please provide multiple different analytical perspectives, covering input exceptions, access control, call chains, and related aspects.
"""

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful code analysis assistant"},
        {"role": "user", "content": question}
    ],
    temperature=0.5,
    max_tokens=2048,
    stream=False
)

for i, choice in enumerate(response.choices):
    print(f"Candidate Analysis {i+1}:{choice.message.content.strip()}\n")