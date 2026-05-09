from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
import os

# Example data
examples = [
    {
        "context": "A large manufacturing company's supply chain system experienced delays, causing a 15% drop in production efficiency. Investigation revealed that the main causes were disorganized supplier management and inaccurate inventory forecasting.",
        "answer": """Problem Analysis Report:
                    Core Issue: Low supply chain efficiency
                    Impact Level: 15% reduction in production efficiency
                    Root Causes:
                    - Inadequate supplier management system
                    - Insufficient accuracy in inventory forecasting system

                    Recommended Solutions:
                    1. Optimize the supplier evaluation system
                    2. Introduce an intelligent forecasting system
                    3. Establish a real-time monitoring mechanism"""
    },
    {
        "context": "A tech company's employee turnover rate reached 25%, concentrated mainly in the R&D department, impacting product iteration speed.",
        "answer": """Problem Analysis Report:
                    Core Issue: High employee turnover rate
                    Impact Level: 25% turnover rate
                    Root Causes:
                    - Insufficient competitiveness of salary and benefits
                    - Limited career development opportunities

                    Recommended Solutions:
                    1. Optimize the compensation structure
                    2. Improve the promotion mechanism
                    3. Improve the work environment"""
    }
]

# Create vector database
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
example_texts = [ex["context"] for ex in examples]
db = FAISS.from_texts(example_texts, embeddings)

# User's current issue description
current_issue = """A retail chain company's customer complaint rate has risen 40% over the past three months, mainly around delivery timeliness and product quality, negatively impacting brand reputation."""

# Retrieve the most similar example
docs = db.similarity_search(current_issue, k=1)
most_similar_example = next(ex for ex in examples if ex["context"] == docs[0].page_content)

# Build the prompt
prompt = """Here is a sample business problem analysis:

Example:
Based on the following situation:
{example_context}

{example_answer}

Now, based on the following problem, generate an analysis report in the same format:
{input_context}

Please maintain professionalism and actionability in the analysis.
"""

# Create LLM
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Format prompt and generate response
formatted_prompt = prompt.format(
    example_context=most_similar_example["context"],
    example_answer=most_similar_example["answer"],
    input_context=current_issue
)

print(formatted_prompt)

response = llm.invoke(formatted_prompt)
print(response)
