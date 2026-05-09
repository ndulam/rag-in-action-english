from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
import os

# 1. Load documents
loader = TextLoader("90-Data/black-myth-wukong/settings.txt")
documents = loader.load()

# 2. Split documents
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 3. Create vector database
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
db = FAISS.from_documents(texts, embeddings)

# 4. Retrieve relevant content
query = "What are the characteristics and combat style of the White Bone Demon?"
docs = db.similarity_search(query)
retrieved_content = docs[0].page_content

# 5. Define prompt template
template = """
Based on the following retrieved information:
{context}

Please provide a detailed character analysis report in the following format:

Character Name: [Provide full name]

Background Story: Introduce the character's origin and background, relationships with other characters, and their role in the story.
Skill Highlights: Describe the character's main skills and abilities, special powers, and combat style.
Combat Strategy: Describe the character's main attack methods, defensive mechanisms, special behaviors in battle, counters and weaknesses.

Please provide a thorough analysis based on the retrieved information, ensuring accuracy and coherence.
"""

# Create PromptTemplate and LLM
prompt = PromptTemplate(
    input_variables=["context"],
    template=template
)

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Generate text
formatted_prompt = prompt.format(context=retrieved_content)
response = llm.invoke(formatted_prompt)
print(response)
