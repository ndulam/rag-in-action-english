import os
import asyncio
import random
import nest_asyncio
import numpy as np
import pandas as pd
from collections import defaultdict

# Import LlamaIndex modules
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.evaluation import (
    DatasetGenerator, QueryResponseDataset,
    CorrectnessEvaluator, SemanticSimilarityEvaluator, RelevancyEvaluator, FaithfulnessEvaluator, PairwiseComparisonEvaluator,
    BatchEvalRunner
)
from llama_index.core.evaluation.eval_utils import get_responses, get_results_df

# 1. Configure LLM, Embedding, and text splitter
# --------------------------------------------------
# Set up LLM and Embedding model
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
)
text_splitter = SentenceSplitter()
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
# Global settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

# 2. Load PDF document
# --------------------------------------------------
pdf_path = "/home/huangj2/Documents/rag-in-action/90-Data/complex-pdf/IPCC_AR6_WGII_Chapter03.pdf"
documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

# 3. Split text into nodes
# --------------------------------------------------
# Sliding window nodes
nodes = node_parser.get_nodes_from_documents(documents)
# Base sentence nodes
base_nodes = text_splitter.get_nodes_from_documents(documents)

# 4. Build vector index
# --------------------------------------------------
sentence_index = VectorStoreIndex(nodes)
base_index = VectorStoreIndex(base_nodes)

# 5. Retrieval and QA example
# --------------------------------------------------
# Sliding window retriever
query_engine = sentence_index.as_query_engine(
    # Set top-k retrieval results to 2
    similarity_top_k=2,
    # Use MetadataReplacementPostProcessor to replace metadata with window content
    node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")],
)
window_response = query_engine.query("What are the concerns surrounding the AMOC?")
print("\n[Sliding window retriever result]\n", window_response)
# Get window content of first retrieval result
window = window_response.source_nodes[0].node.metadata["window"]
# Get original sentence text of first retrieval result
sentence = window_response.source_nodes[0].node.metadata["original_text"]
print(f"Window: {window}")
print("------------------")
print(f"Original Sentence: {sentence}")

# Base retriever
base_query_engine = base_index.as_query_engine(similarity_top_k=2)
vector_response = base_query_engine.query("What are the concerns surrounding the AMOC?")
print("\n[Base retriever result]\n", vector_response)

# Print original sentences from all sliding window retriever source_nodes
print("\n[Sliding window retriever source_nodes original sentences]")
for source_node in window_response.source_nodes:
    print(source_node.node.metadata["original_text"])
    print("--------")

# Check whether base retriever source_nodes contain 'AMOC'
print("\n[Base retriever source_nodes: does node contain 'AMOC'?]")
for node in vector_response.source_nodes:
    print("AMOC mentioned?", "AMOC" in node.node.text)
    print("--------")

# Print all base retriever source_nodes node.text
print("\n[Base retriever source_nodes node.text]")
for i, node in enumerate(vector_response.source_nodes):
    print(f"Node {i+1} text:")
    print(node.node.text)
    print("--------")

# 6. Prepare evaluation dataset
# --------------------------------------------------
# Sample a subset of nodes to generate evaluation questions
num_nodes_eval = 30
sample_eval_nodes = random.sample(base_nodes[:200], num_nodes_eval)
# Generate evaluation dataset (uncomment to regenerate)
# dataset_generator = DatasetGenerator(
#     sample_eval_nodes,
#     llm=OpenAI(model="gpt-4"),
#     show_progress=True,
#     num_questions_per_chunk=2,
# )
# eval_dataset = await dataset_generator.agenerate_dataset_from_nodes()
# eval_dataset.save_json("90-Data/complex-pdf/ipcc_eval_qr_dataset.json")

# Load pre-generated evaluation dataset
eval_dataset = QueryResponseDataset.from_json("90-Data/complex-pdf/ipcc_eval_qr_dataset.json")

# 7. Build evaluators
# --------------------------------------------------
evaluator_c = CorrectnessEvaluator(llm=OpenAI(model="gpt-4"))
evaluator_s = SemanticSimilarityEvaluator()
evaluator_r = RelevancyEvaluator(llm=OpenAI(model="gpt-4"))
evaluator_f = FaithfulnessEvaluator(llm=OpenAI(model="gpt-4"))
# pairwise_evaluator = PairwiseComparisonEvaluator(llm=OpenAI(model="gpt-4"))

evaluator_dict = {
    "correctness": evaluator_c,
    "faithfulness": evaluator_f,
    "relevancy": evaluator_r,
    "semantic_similarity": evaluator_s,
}
batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

# 8. Main evaluation workflow
# --------------------------------------------------
async def main():
    max_samples = 30
    eval_qs = eval_dataset.questions
    ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

    # Rebuild retrievers to ensure evaluation consistency
    base_query_engine = base_index.as_query_engine(similarity_top_k=2)
    window_query_engine = sentence_index.as_query_engine(
        similarity_top_k=2,
        node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")],
    )

    # Get model responses
    base_pred_responses = get_responses(
        eval_qs[:max_samples], base_query_engine, show_progress=True
    )
    pred_responses = get_responses(
        eval_qs[:max_samples], window_query_engine, show_progress=True
    )

    # Evaluate
    eval_results = await batch_runner.aevaluate_responses(
        queries=eval_qs[:max_samples],
        responses=pred_responses[:max_samples],
        reference=ref_response_strs[:max_samples],
    )
    base_eval_results = await batch_runner.aevaluate_responses(
        queries=eval_qs[:max_samples],
        responses=base_pred_responses[:max_samples],
        reference=ref_response_strs[:max_samples],
    )
    results_df = get_results_df(
        [eval_results, base_eval_results],
        ["Sentence Window Retriever", "Base Retriever"],
        ["correctness", "relevancy", "faithfulness", "semantic_similarity"],
    )
    print("\n[Evaluation Results]")
    print(results_df)

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
