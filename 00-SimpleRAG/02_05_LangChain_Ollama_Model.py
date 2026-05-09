"""
Run a large language model locally using Ollama — no OpenAI API key required.

1. Install the Ollama Server:
   - Windows: Visit https://ollama.com/download to download the installer
   - Linux/Mac: Run curl -fsSL https://ollama.com/install.sh | sh

2. Download and run a model:
   - Open a terminal and run the following commands to download a model:
     ollama pull qwen:7b  # Download the Qwen 7B model
     # or
     ollama pull llama2:7b  # Download the Llama2 7B model
     # or
     ollama pull mistral:7b  # Download the Mistral 7B model

3. Set environment variables:
   - Add the following to your .env file:
     OLLAMA_MODEL=qwen:7b  # Or the name of another downloaded model
"""

# 1. Load documents
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Black_Myth:_Wukong",)
)
docs = loader.load()

# 2. Split documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 3. Set up the embedding model
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 4. Create the vector store
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)

# 5. Build the user query
question = "What game scenes are available in Black Myth: Wukong?"

# 6. Search for relevant documents in the vector store and prepare the context
retrieved_docs = vector_store.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 7. Build the prompt template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
                Based on the following context, answer the question. If there is no relevant information in the context,
                please say "I cannot find relevant information in the provided context."
                Context: {context}
                Question: {question}
                Answer:"""
                                          )

# 8. Generate an answer using the large language model
from langchain_ollama import ChatOllama # pip install langchain-ollama
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"))
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer.content)
