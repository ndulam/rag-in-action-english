import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI as OpenAIClient  # Avoid name conflict with TruLens's OpenAI class

# TruLens is an open-source library for observability and evaluation of deep learning models,
# especially LLM applications. It helps developers track, debug, and evaluate the performance
# of complex applications such as RAG (Retrieval-Augmented Generation).
# - TruSession: Manages evaluation sessions and result storage.
# - Feedback: Defines evaluation metrics such as relevance, faithfulness, etc.
# - TruApp: Wraps your application to make it monitorable by TruLens.
# - instrument: A decorator used to mark specific functions for tracing.
from trulens.core import TruSession, Feedback, Select
from trulens.apps.app import TruApp, instrument
from trulens.providers.openai import OpenAI as TruLensOpenAI
import numpy as np

# Set API key
# os.environ["OPENAI_API_KEY"] = "your_key_here"

# Initialize embedding function
embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get("OPENAI_API_KEY"),
                                             model_name="text-embedding-ada-002")
chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection("Info", embedding_function=embedding_function)

# Add sample data
vector_store.add("starbucks_info", documents=[
    """
    Starbucks Corporation is an American multinational chain of coffeehouses headquartered in Seattle, Washington.
    As the world's largest coffeehouse chain, Starbucks is seen to be the main representation of the United States' second wave of coffee culture.
    """
])

class RAG:
    # @instrument is a decorator provided by TruLens to "instrument" or "equip" a function.
    # Once marked, TruLens records its inputs, outputs, execution time, errors, etc. on every call.
    # This is critical for understanding each step in the RAG pipeline (retrieval, generation).
    @instrument
    def retrieve(self, query: str):
        """Retrieve relevant documents"""
        results = vector_store.query(query_texts=[query], n_results=2)
        return results["documents"][0] if results["documents"] else []

    @instrument
    def generate_completion(self, query: str, context: list):
        """Generate an answer"""
        oai_client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY"))
        context_str = "\n".join(context) if context else "No context available."
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Context: {context_str}\nQuestion: {query}"}]
        ).choices[0].message.content
        return completion

    @instrument
    def query(self, query: str):
        """Full RAG query pipeline"""
        context = self.retrieve(query)
        return self.generate_completion(query, context)

# Initialize TruLens session
# TruSession is the entry point for interacting with the TruLens backend (local SQLite or database).
# It manages and stores all tracing data and evaluation results.
# database_redact_keys=True automatically redacts sensitive info (e.g. API keys) from records.
session = TruSession(database_redact_keys=True)
# Resetting the database clears all previous records, ensuring a clean start for this evaluation.
session.reset_database()

# Initialize TruLens OpenAI provider
# A Provider is the "judge" TruLens uses to execute evaluations. Here we use OpenAI's gpt-4 model.
# These evaluations are performed by asking the LLM questions such as:
# "Is the given answer consistent with the context?"
provider = TruLensOpenAI(model_engine="gpt-4")

# Define evaluation metrics (Feedback Functions)
# Feedback is a core concept in TruLens for defining evaluation dimensions.
# Each Feedback function consists of an evaluator (provider method) and selectors.
# Selectors (.on(...)) precisely specify which part of the app to evaluate (input, output, or intermediate).

# 1. Groundedness
#    - Evaluator: provider.groundedness_measure_with_cot_reasons uses chain-of-thought to judge
#      whether the answer is fully grounded in the provided context.
#    - Selector: .on(Select.RecordCalls.retrieve.rets) specifies the context as the return value of `retrieve`.
#                .on_output() specifies that the content to evaluate is the final output of the RAG app.
f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness") \
    .on(Select.RecordCalls.retrieve.rets).on_output()

# 2. Answer Relevance
#    - Evaluator: provider.relevance_with_cot_reasons judges whether the generated answer is
#      relevant to the original question.
#    - Selector: .on_input() specifies the top-level input (user's query).
#                .on_output() specifies the final output.
f_answer_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance") \
    .on_input().on_output()

# 3. Context Relevance
#    - Evaluator: provider.context_relevance_with_cot_reasons judges how relevant the retrieved
#      context is to the original question.
#    - Selector: .on_input() specifies the top-level input.
#                .on(Select.RecordCalls.retrieve.rets[:]) specifies each element in the list
#                returned by the `retrieve` method.
#    - Aggregator: .aggregate(np.mean) combines all context relevance scores into a single score
#      since the context may contain multiple documents.
f_context_relevance = Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance") \
    .on_input().on(Select.RecordCalls.retrieve.rets[:]).aggregate(np.mean)

# Set up TruApp
# TruApp bundles our RAG application instance with the Feedback function list.
# It creates an "observable" application that can be tracked and evaluated by TruLens.
rag = RAG()
tru_rag = TruApp(
    rag,
    app_name="RAG",
    app_version="base",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
)

# Execute queries and record
# Use the `with tru_rag as recording:` context manager to run the application.
# Inside this block, calls to `rag.query()` and all methods marked with `@instrument` are
# automatically recorded by TruLens.
# The recorded data (app-json) contains the full call chain, inputs/outputs, and intermediate results.
with tru_rag as recording:
    response = rag.query("What wave of coffee culture is Starbucks seen to represent in the United States?")
    print(f"Response: {response}")

# View evaluation results
# get_leaderboard() reads records from the database and displays all evaluation results in a table.
# It shows the average score for each Feedback, giving a quick overview of overall app performance.
# This leaderboard is useful for comparing different app versions
# (e.g., after changing prompts, models, or retrieval strategies).
print(session.get_leaderboard())
