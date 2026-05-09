from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Define two prompt templates
combat_template = """You are an expert in Black Myth: Wukong combat techniques.
You excel at answering questions about Black Myth: Wukong combat in a clear and concise manner.
When you don't know the answer to a question, you say so honestly.

Here is a question:
{query}"""

story_template = """You are an expert familiar with the storyline of Black Myth: Wukong.
You excel at breaking down complex plot points and explaining them in detail.
When you don't know the answer to a question, you say so honestly.

Here is a question:
{query}"""

# Initialize the embedding model
embeddings = OpenAIEmbeddings()
prompt_templates = [combat_template, story_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# Define the routing function
def prompt_router(input):
    # Embed the user query
    query_embedding = embeddings.embed_query(input["query"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Select the most similar prompt template
    print("Using combat techniques template" if most_similar == combat_template else "Using storyline template")
    return PromptTemplate.from_template(most_similar)

# Create the processing chain
chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatOpenAI()
    | StrOutputParser()
)

# Example question
print(chain.invoke("How does Wukong defeat enemies in Black Myth: Wukong?"))
