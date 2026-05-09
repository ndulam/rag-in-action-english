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

# 2. Split documents
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

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(all_splits)

# 5. Create the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. Create the prompt template
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Based on the following context, answer the question. If there is no relevant information in the context,
please say "I cannot find relevant information in the provided context."
Context: {context}
Question: {question}
Answer:""")

# 7. Set up the language model and output parser
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"))

# 8. Build the LCEL chain
# Pipeline-style data flow, similar to Unix command pipes (|), chains different processing steps together
chain = (
    {
        # Retriever input: question string, output: list of Documents
        # Lambda function input: list of Documents, output: merged text string
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        # RunnablePassthrough input: question string, output: question string passed through unchanged
        "question": RunnablePassthrough()
    }
    # prompt input: dict {"context": text, "question": question}, output: formatted prompt template string
    | prompt
    # llm input: prompt template string, output: ChatMessage object
    | llm
    # StrOutputParser input: ChatMessage object, output: answer text string
    | StrOutputParser()
)

# Inspect the input/output at each stage
question = "test question"

# 1. Retriever stage
retriever_output = retriever.invoke(question)
print("Retriever output:", retriever_output)

# 2. Document merging stage
context = "\n\n".join(doc.page_content for doc in retriever_output)
print("Merged document output:", context)

# 3. Prompt template stage
prompt_output = prompt.invoke({"context": context, "question": question})
print("Prompt template output:", prompt_output)

# 4. LLM stage
llm_output = llm.invoke(prompt_output)
print("LLM output:", llm_output)

# 5. Parser stage
final_output = StrOutputParser().invoke(llm_output)
print("Final output:", final_output)

# 9. Execute the query
question = "What game scenes are available in Black Myth: Wukong?"
response = chain.invoke(question) # Synchronous; can be replaced with async execution
