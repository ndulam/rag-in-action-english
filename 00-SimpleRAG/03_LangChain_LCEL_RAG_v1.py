# 1. Load documents
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
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

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
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-3.5-turbo")

# 8. Build the LCEL chain
# Pipeline-style data flow, similar to Unix command pipes (|), chains different processing steps together
chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
) # Inspect the input/output at each stage

# 9. Execute the query
question = "What game scenes are available in Black Myth: Wukong?"
response = chain.invoke(question) # Synchronous; can be replaced with async execution
print(response)
