# File: contextual_retrieval.py
# Description: Demonstrates how to implement contextual retrieval using LlamaIndex, with error-handling optimizations.
# Original docs: https://docs.llamaindex.ai/en/stable/examples/cookbooks/contextual_retrieval/

import os
import pandas as pd
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.evaluation import generate_question_context_pairs, RetrieverEvaluator
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from typing import List
import asyncio
from dotenv import load_dotenv
load_dotenv()

# --- Install required libraries ---
# !pip install llama-index llama-index-llms-openai llama-index-embeddings-openai llama-index-postprocessor-cohere-rerank llama-index-retrievers-bm25 pandas

# --- Set API keys ---
# Ensure OPENAI_API_KEY and COHERE_API_KEY are set in environment variables
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["COHERE_API_KEY"] = "your-cohere-api-key"

# --- Set up LLM and Embedding model ---
llm = OpenAI(model="gpt-3.5-turbo")  # Can replace with gpt-4 for better results
embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# --- Sample text data ---
paul_graham_essay_text = """
What I Worked On
February 2021
Before college the two main things I worked on, outside of school, were writing and programming. I wrote short stories and I programmed on an IBM 1401 at my father's company on weekends. I got interested in philosophy in college. I don't know if I would have had I not gone to college, but I suspect not, because the philosophy I encountered in college is quite different from what you'd encounter outside it.
I wanted to study philosophy in college, but my father, who was paying, said I couldn't because it wasn't practical. So I majored in something called "Science and Literature" which was offered by the college of Arts and Sciences but counted as science. I wrote my thesis on the philosophy of mathematics, based on the work of Friedrich Waismann, himself a disciple of Wittgenstein.
I went to grad school in philosophy at Cornell, but after a year I left to study computer science at Harvard, which seemed more exciting. After the first year I started working on what would become Common Lisp. That led to me working with some people starting a company called Lucid. I didn't officially join Lucid, but I spent a lot of time there. In the meantime I also worked on On Lisp. After college I'd occasionally written short essays. In grad school I wrote more of them, and started to publish them on the web, and after that I started to get invited to give talks. In the summer of 1995 I was invited to give a talk at a conference on programming language design. I couldn't take time off because I was running a small consulting business. So I decided to write an essay instead.
"""

# --- Prompt template for generating context per chunk ---
CONTEXT_PROMPT_TEMPLATE = """
Here is a passage from the document:
"{context_str}"

Based on this passage, create a concise sentence that describes the content of this passage.
This will be used to answer questions about the document.
Context: """

# --- Utility functions ---

# Create embedding-based retriever
def create_embedding_retriever(nodes, similarity_top_k=3):
    """Create a vector-embedding-based retriever"""
    # Ensure similarity_top_k does not exceed the number of nodes
    adjusted_top_k = min(similarity_top_k, len(nodes))
    if adjusted_top_k < similarity_top_k:
        print(f"Warning: similarity_top_k adjusted from {similarity_top_k} to {adjusted_top_k} due to node count limit")

    # Ensure at least 1
    adjusted_top_k = max(1, adjusted_top_k)

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    return index.as_retriever(similarity_top_k=adjusted_top_k)

# Create BM25-based retriever
def create_bm25_retriever(nodes, similarity_top_k=3):
    """Create a BM25-algorithm-based retriever"""
    # Convert nodes to TextNode
    text_nodes = [TextNode(text=node.get_content(), id_=node.node_id) for node in nodes if hasattr(node, 'get_content')]

    # Check for valid nodes
    if not text_nodes:
        print("Warning: No valid TextNodes for BM25 retriever")
        text_nodes = [TextNode(text="sample text", id_="sample_id")]

    # Adjust top_k to not exceed corpus size
    adjusted_top_k = min(similarity_top_k, len(text_nodes))
    if adjusted_top_k < similarity_top_k:
        print(f"Warning: similarity_top_k adjusted from {similarity_top_k} to {adjusted_top_k} due to corpus size limit")

    # Ensure at least 1
    adjusted_top_k = max(1, adjusted_top_k)

    return BM25Retriever.from_defaults(nodes=text_nodes, similarity_top_k=adjusted_top_k)

# Hybrid retriever (Embedding + BM25 + Reranker)
class EmbeddingBM25RerankerRetriever(BaseRetriever):
    """Hybrid retriever combining vector retrieval, BM25 retrieval, and a reranker"""
    def __init__(self, embedding_retriever, bm25_retriever, reranker):
        self._embedding_retriever = embedding_retriever
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Get results from both retrievers
        embedding_nodes = self._embedding_retriever.retrieve(query_bundle)
        bm25_nodes = self._bm25_retriever.retrieve(query_bundle)

        # Merge and deduplicate
        all_nodes = {node.node.node_id: node for node in embedding_nodes}
        for node in bm25_nodes:
            if node.node.node_id not in all_nodes:
                all_nodes[node.node.node_id] = node

        # Prepare nodes for reranking
        nodes_for_rerank = list(all_nodes.values())

        # Apply reranker
        if self._reranker and nodes_for_rerank:
            reranked_nodes = self._reranker.postprocess_nodes(
                nodes_for_rerank, query_bundle=query_bundle
            )
            return reranked_nodes
        return nodes_for_rerank

# Evaluation function
async def retrieval_results(retriever, qa_dataset):
    """Evaluate retriever performance"""
    try:
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=retriever
        )
        eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
        return eval_results
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        # Return empty result to avoid interrupting execution
        return []

# Display results
def display_results(name, eval_results):
    """Display evaluation results"""
    if not eval_results:
        # Handle empty results
        return pd.DataFrame({
            "retrievers": [name],
            "hit_rate": [0.0],
            "mrr": [0.0],
            "note": ["Evaluation failed"]
        })

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    if not metric_dicts:
        return pd.DataFrame({
            "retrievers": [name],
            "hit_rate": [0.0],
            "mrr": [0.0],
            "note": ["No evaluation data"]
        })

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean() if "hit_rate" in full_df else 0.0
    mrr = full_df["mrr"].mean() if "mrr" in full_df else 0.0

    # Calculate additional metrics (if present)
    metrics = {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}
    for metric in ["precision", "recall", "ap", "ndcg"]:
        if metric in full_df:
            metrics[metric] = [full_df[metric].mean()]

    return pd.DataFrame(metrics)

# --- Main function ---
async def main():
    print("Starting contextual retrieval experiment")

    # 1. Convert document to Document object
    documents = [Document(text=paul_graham_essay_text)]
    print("Document loaded")

    # 2. Split document into nodes (chunks)
    # Use a larger chunk size to generate enough nodes for the experiment
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} nodes")

    # Debug: show node count — too few may cause BM25 retriever issues
    if len(nodes) < 3:
        print("Warning: fewer than 3 nodes, which may cause retriever evaluation issues")
        # Add extra sample nodes if too few
        while len(nodes) < 3:
            sample_text = f"Sample text {len(nodes)+1}: this is an additional text node for testing."
            sample_node = TextNode(text=sample_text, id_=f"sample_node_{len(nodes)}")
            nodes.append(sample_node)
        print(f"Sample nodes added; total nodes: {len(nodes)}")

    # 3. Generate contextual description for each node
    nodes_contextual = []
    for node in nodes:
        # Simulate LLM-generated context
        simulated_context = f"This passage discusses: {node.get_content()[:50]}..."
        new_metadata = node.metadata.copy() if hasattr(node, 'metadata') else {}
        new_metadata["generated_context"] = simulated_context

        # Create new node with generated context
        contextual_node = TextNode(
            text=node.get_content(),
            metadata=new_metadata,
            id_=node.node_id
        )
        nodes_contextual.append(contextual_node)

    # 4. Set retrieval parameters — ensure they don't exceed node count
    similarity_top_k = min(3, len(nodes))
    print(f"Retrieval parameter similarity_top_k set to {similarity_top_k}")

    # 5. Set up Cohere reranker
    try:
        cohere_rerank = CohereRerank(
            api_key=os.environ.get("COHERE_API_KEY", "your-api-key"),
            model="rerank-english-v3.0",
            top_n=similarity_top_k
        )
        print("Cohere reranker set up successfully")
    except Exception as e:
        print(f"Error setting up Cohere reranker: {str(e)}")
        cohere_rerank = None

    # 6. Create various retrievers
    print("Creating standard retrievers...")
    embedding_retriever = create_embedding_retriever(
        nodes, similarity_top_k=similarity_top_k
    )
    bm25_retriever = create_bm25_retriever(
        nodes, similarity_top_k=similarity_top_k
    )
    embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
        embedding_retriever, bm25_retriever, reranker=cohere_rerank
    )

    print("Creating contextual retrievers...")
    contextual_embedding_retriever = create_embedding_retriever(
        nodes_contextual, similarity_top_k=similarity_top_k
    )
    contextual_bm25_retriever = create_bm25_retriever(
        nodes_contextual, similarity_top_k=similarity_top_k
    )
    contextual_embedding_bm25_retriever_rerank = EmbeddingBM25RerankerRetriever(
        contextual_embedding_retriever,
        contextual_bm25_retriever,
        reranker=cohere_rerank,
    )

    # 7. Create evaluation dataset
    print("Creating evaluation dataset...")
    from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

    fixed_queries = {
        "q1": "What did the author work on before college?",
        "q2": "What major did the author choose in college?",
        "q3": "What did the author do after graduation?"
    }

    # Set relevant document mapping
    relevant_docs_mapping = {}
    if nodes:
        for i, query_id in enumerate(fixed_queries.keys()):
            node_index = min(i, len(nodes)-1)  # Ensure index does not exceed range
            relevant_docs_mapping[query_id] = [nodes[node_index].node_id]

    # Build corpus
    corpus_data = {}
    if nodes:
        corpus_data = {node.node_id: node.get_content() for node in nodes}

    # Create evaluation dataset
    qa_dataset = EmbeddingQAFinetuneDataset(
        queries=fixed_queries,
        corpus=corpus_data,
        relevant_docs=relevant_docs_mapping
    )

    # 8. Evaluate standard retrievers
    print("\n--- Evaluating standard retrievers ---")
    embedding_retriever_results = await retrieval_results(
        embedding_retriever, qa_dataset
    )
    bm25_retriever_results = await retrieval_results(
        bm25_retriever, qa_dataset
    )
    embedding_bm25_retriever_rerank_results = await retrieval_results(
        embedding_bm25_retriever_rerank, qa_dataset
    )

    # 9. Evaluate contextual retrievers
    print("\n--- Evaluating contextual retrievers ---")
    contextual_embedding_retriever_results = await retrieval_results(
        contextual_embedding_retriever, qa_dataset
    )
    contextual_bm25_retriever_results = await retrieval_results(
        contextual_bm25_retriever, qa_dataset
    )
    contextual_embedding_bm25_retriever_rerank_results = await retrieval_results(
        contextual_embedding_bm25_retriever_rerank, qa_dataset
    )

    # 10. Display results
    print("\n--- Results without context ---")
    standard_results = pd.concat(
        [
            display_results("Embedding Retriever", embedding_retriever_results),
            display_results("BM25 Retriever", bm25_retriever_results),
            display_results(
                "Embedding + BM25 + Reranker",
                embedding_bm25_retriever_rerank_results,
            ),
        ],
        ignore_index=True,
        axis=0,
    )
    print(standard_results)

    print("\n--- Results with context ---")
    contextual_results = pd.concat(
        [
            display_results(
                "Contextual Embedding Retriever",
                contextual_embedding_retriever_results,
            ),
            display_results(
                "Contextual BM25 Retriever",
                contextual_bm25_retriever_results
            ),
            display_results(
                "Contextual Embedding + Contextual BM25 + Reranker",
                contextual_embedding_bm25_retriever_rerank_results,
            ),
        ],
        ignore_index=True,
        axis=0,
    )
    print(contextual_results)

if __name__ == "__main__":
    asyncio.run(main())
