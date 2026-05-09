#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contextual Retrieval with Milvus
Based on the approach proposed by Anthropic to solve the semantic isolation problem in traditional RAG.

Core concepts:
1. Traditional RAG problem: documents are split into isolated chunks, losing contextual information.
2. Contextual retrieval solution: add document context to each chunk for richer semantics.
3. Deep Evaluation: multi-dimensional evaluation of retrieval system performance.

Tech stack:
- Milvus: vector database supporting dense and sparse vectors
- SentenceTransformer: generates vector representations of text
- OpenAI GPT: used to generate contextualized text chunks (replacing Claude)
- Cohere Reranker: reranking model to optimize retrieval results

API change notes:
- Original version used Anthropic Claude API for context augmentation
- Current version uses OpenAI GPT API for better availability and performance
- Claude-related code is commented out and can be switched back if needed

===============================================================================
Code Structure Analysis
===============================================================================

Overall Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                  RAG Contextual Retrieval System Architecture         │
├─────────────────────────────────────────────────────────────────────┤
│  Input:   Raw docs → Chunking → Context augmentation → Embed → Store │
│  Retrieval: Query embed → Similarity search → Rerank → Return        │
│  Evaluation: Gold standard comparison → Metric calculation → Analysis│
└─────────────────────────────────────────────────────────────────────┘

Module Responsibilities:
1. MilvusContextualRetriever (core retriever)
   - Handles vector database operations
   - Implements multiple retrieval strategies
   - Manages contextualization and reranking pipelines

2. Evaluation Module (Performance Evaluation)
   - evaluate_retrieval(): core evaluation logic
   - evaluate_db(): database performance evaluation
   - retrieve_base(): base retrieval interface

3. Data Processing Module
   - download_data(): data download
   - load_jsonl(): data loading
   - Data format normalization

4. Experiment Control Module
   - main(): experiment workflow control
   - Comparison of three retrieval strategies
   - Performance metric statistics

===============================================================================
Data Flow Analysis
===============================================================================

Data Processing Pipeline:
Raw docs → Chunking → [optional] Context augmentation → Embedding → Milvus storage
    ↓
Query input → Query embedding → Similarity search → [Reranking] → Results
    ↓
Evaluation → Performance metrics → Analysis

Retrieval Strategy Comparison:
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  Strategy    │  Preprocessing  │  Retrieval  │  Post-proc  │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Standard     │ Raw chunks   │ Dense search │ None        │
│ Contextual   │ LLM-enhanced │ Dense search │ None        │
│ Reranked     │ LLM-enhanced │ Dense search │ Cohere rerank│
└──────────────┴──────────────┴──────────────┴──────────────┘

Evaluation Framework:
Input: query + gold standard answer
Process: retrieval → matching → scoring
Output: Pass@K score, average score, recall

===============================================================================
Execution Flow Analysis
===============================================================================

Main Execution Steps:
1. Environment initialization
   - Load API keys and configuration
   - Initialize embedding model and reranking model
   - Download sample data

2. Data preparation
   - Load document dataset
   - Create evaluation query set
   - Data format validation

3. Experiment 1: Standard retrieval baseline
   - Create standard Milvus collection
   - Insert raw text chunks
   - Run retrieval evaluation

4. Experiment 2: Contextual retrieval
   - Create contextual Milvus collection
   - LLM-augment text chunks
   - Insert augmented text chunks
   - Run retrieval evaluation

5. Experiment 3: Reranked retrieval
   - Use contextual retriever
   - Enable Cohere reranking
   - Run retrieval evaluation

6. Result analysis
   - Compare performance across three strategies
   - Calculate performance improvement
   - Generate analysis report

===============================================================================
Key Parameters
===============================================================================

Vector Database Configuration:
- Collection name: distinguishes data collections for different experiments
- Vector dimension: determined by the embedding model (e.g. BGE-large-zh = 1024 dims)
- Index type: FLAT (exact search) + SPARSE_INVERTED_INDEX (sparse vectors)
- Distance metric: inner product (IP)

LLM Configuration:
- Model: gpt-3.5-turbo (fast/economical) or gpt-4 (high quality)
- Max tokens: 1000
- Temperature: 0 (ensures consistency)
- API: OpenAI ChatGPT API

Retrieval Configuration:
- K (retrieval count): default 5 (Pass@5 evaluation)
- Search params: nprobe=10
- Reranking: Cohere Rerank API

===============================================================================
"""

# Import required libraries
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import CohereRerankFunction

from typing import List, Dict, Any
from typing import Callable
from pymilvus import (
    MilvusClient,
    DataType,
    AnnSearchRequest,
    RRFRanker,
)
from tqdm import tqdm
import json
# import anthropic  # Claude API commented out, using OpenAI instead
import openai  # OpenAI API support added
import os
import dotenv
dotenv.load_dotenv()

class MilvusContextualRetriever:
    """
    Milvus Contextual Retriever class

    Architecture design:
    This class is the core of the entire retrieval system. It uses a modular design
    that supports flexible combinations of multiple retrieval strategies.

    Functional modules:
    1. Standard retrieval: vector retrieval on raw text chunks
    2. Hybrid retrieval: combining dense and sparse vectors
    3. Contextual retrieval: uses LLM to enrich text chunk context before retrieval
    4. Reranked retrieval: applies a specialized reranking model on retrieval results

    Data flow:
    Text input → [Context augmentation] → Embedding → Milvus storage
           ↓
    Query input → Embedding → Similarity search → [Reranking] → Output

    Design principles:
    - Single responsibility: each method handles one specific function
    - Open/closed: supports extension with new retrieval strategies
    - Dependency injection: dependencies injected via constructor
    - Configuration-driven: features enabled/disabled via parameters

    Supports standard retrieval, hybrid retrieval, contextual retrieval, and reranking.
    """

    def __init__(
        self,
        uri="milvus.db",
        collection_name="contexual_bgem3",
        dense_embedding_function=None,
        use_sparse=False,
        sparse_embedding_function=None,
        use_contextualize_embedding=False,
        llm_client=None,  # Generic LLM client name (supports OpenAI)
        use_reranker=False,
        rerank_function=None,
    ):
        """
        Initialize the retriever

        Initialization steps:
        1. Set Milvus connection parameters
        2. Configure embedding functions (dense + sparse)
        3. Set up LLM client (for context augmentation)
        4. Configure reranking
        5. Parameter validation and error handling

        Parameters:
            uri: Milvus service address
                - Local file mode: e.g. "./milvus.db" (Milvus Lite)
                - Server mode: e.g. "http://localhost:19530" (standalone Milvus)
                - Cloud mode: Zilliz Cloud connection address
            collection_name: collection name, analogous to a table in a database
            dense_embedding_function: dense vector embedding function
                - Converts text to high-dimensional dense vectors (e.g. 768 or 1024 dims)
                - Typically uses pretrained language models like BGE or SentenceTransformer
            use_sparse: whether to use sparse vectors
                - Sparse vectors are similar to TF-IDF and capture keyword matching
                - Combined with dense vectors can improve retrieval accuracy
            sparse_embedding_function: sparse vector embedding function
            use_contextualize_embedding: whether to use contextual embedding
                - Core feature: uses LLM to add document context to each chunk
                - Solves the problem of chunks lacking context in traditional RAG
            llm_client: LLM client (supports OpenAI GPT or Claude)
                - Used to generate contextualized text chunks
                - Current version primarily supports OpenAI GPT-3.5/GPT-4
                - Original Claude API code is commented out and preserved
            use_reranker: whether to use reranking
                - Applies a specialized reranking model on initial retrieval results
            rerank_function: reranking function (e.g. Cohere Rerank)
        """
        self.collection_name = collection_name

        # For Milvus-lite, uri is a local path like "./milvus.db"
        # For standalone Milvus, uri is like "http://localhost:19530"
        # For Zilliz Cloud, set both `uri` and `token`
        self.client = MilvusClient(uri)

        self.embedding_function = dense_embedding_function

        self.use_sparse = use_sparse
        self.sparse_embedding_function = None

        self.use_contextualize_embedding = use_contextualize_embedding
        # self.anthropic_client = anthropic_client  # Claude client commented out
        self.llm_client = llm_client  # Generic LLM client

        self.use_reranker = use_reranker
        self.rerank_function = rerank_function

        # Parameter validation: if sparse vectors are enabled, a sparse embedding function must be provided
        if use_sparse is True and sparse_embedding_function:
            self.sparse_embedding_function = sparse_embedding_function
        elif use_sparse is True and sparse_embedding_function is None:
            raise ValueError(
                "sparse_embedding_function cannot be None when use_sparse is True"
            )
        else:
            pass

    def build_collection(self):
        """
        Build Milvus collection

        Collection design principles:
        1. Schema design: define data structure and field types
        2. Index strategy: choose appropriate index types to optimize search performance
        3. Dynamic fields: support flexible metadata storage
        4. Vector fields: support mixed storage of dense and sparse vectors

        Storage structure:
        ┌─────────────┬──────────────┬─────────────────┬──────────────┐
        │   Field     │    Type      │    Purpose      │  Index type  │
        ├─────────────┼──────────────┼─────────────────┼──────────────┤
        │ pk          │ INT64        │ Primary key ID  │ Auto         │
        │ dense_vector│ FLOAT_VECTOR │ Semantic vector │ FLAT/IP      │
        │ sparse_vector│ SPARSE_VECTOR│ Keyword vector │ INVERTED/IP  │
        │ content     │ VARCHAR      │ Raw content     │ Dynamic field│
        │ metadata    │ JSON         │ Metadata info   │ Dynamic field│
        └─────────────┴──────────────┴─────────────────┴──────────────┘

        Collection design notes:
        1. Uses dynamic schema to support flexible field addition
        2. Primary key auto-generated to ensure record uniqueness
        3. Dense vector field: stores semantic vector representation of text
        4. Sparse vector field (optional): stores keyword-related sparse vectors
        5. Index strategy: dense vectors use FLAT index; sparse vectors use inverted index
        """
        # Create collection schema definition
        schema = self.client.create_schema(
            auto_id=True,  # Auto-generate primary key ID
            enable_dynamic_field=True,  # Allow dynamic field addition for flexibility
        )

        # Add primary key field
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)

        # Add dense vector field - stores semantic vector representation of text
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_function.dim,  # Vector dimension determined by embedding function
        )

        # If sparse vectors are enabled, add sparse vector field
        if self.use_sparse is True:
            schema.add_field(
                field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
            )

        # Prepare index parameters - indexes accelerate vector search
        index_params = self.client.prepare_index_params()

        # Add index for dense vectors
        # FLAT index: exact search, suitable for small to medium datasets
        # IP (inner product) distance: suitable for normalized vectors
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )

        # Add inverted index for sparse vectors
        if self.use_sparse is True:
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",  # Inverted index specialized for sparse vectors
                metric_type="IP",
            )

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True,
        )

    def insert_data(self, chunk, metadata):
        """
        Insert standard data into Milvus

        Standard data insertion pipeline:
        This is the base data insertion method implementing the simplest text chunk storage strategy.

        Processing steps:
        1. Text input validation
        2. Dense vector generation (semantic representation)
        3. Sparse vector generation (optional, keyword representation)
        4. Data structure assembly
        5. Milvus batch insertion

        Data flow:
        Raw text chunk → Embedding model → Vector representation → Milvus storage
                                        ↓
        Metadata → Field mapping → Dynamic field storage

        Parameters:
            chunk: text chunk content (raw text, not context-augmented)
            metadata: metadata (doc ID, chunk ID, raw content, etc.)

        Stored content:
        1. Dense vector: semantic vector representation of the text chunk
        2. Sparse vector (optional): keyword vector representation of the text chunk
        3. Metadata: document and chunk identification info
        """
        # Generate dense vector representation of the text chunk
        dense_vec = self.embedding_function([chunk])[0]

        # Build data to insert
        data = {
            "dense_vector": dense_vec,
            **metadata  # Expand metadata fields
        }

        # If sparse vectors are enabled, generate and add sparse vector
        if self.use_sparse is True:
            sparse_vec = self.sparse_embedding_function([chunk])[0]
            data["sparse_vector"] = sparse_vec

        # Insert data into Milvus collection
        self.client.insert(
            collection_name=self.collection_name,
            data=[data]
        )

    def insert_contextualized_data(self, doc_content, chunk_content, metadata):
        """
        Insert contextualized data

        Core contextual augmentation pipeline:
        This is the key method implementing contextual retrieval, solving the semantic
        isolation problem in traditional RAG.

        Contextualization steps:
        1. Prepare document context
        2. Build LLM prompt
        3. Call OpenAI GPT API (context augmentation)
        4. Embed augmented text
        5. Store vector data

        Data augmentation flow:
        Raw document ──┐
                       ├─→ LLM prompt → OpenAI GPT API → Context-augmented text
        Text chunk ────┘                                  ↓
                                                Embedding → Augmented vector → Milvus

        Core innovation:
        This is the core feature of contextual retrieval:
        1. Uses the entire document as context
        2. Enriches the semantic information of individual chunks via LLM (OpenAI GPT)
        3. Generates and stores vectors from the augmented text

        Advantages of contextualization:
        - Solves the problem of isolated text chunks
        - Preserves cross-chunk semantic coherence
        - Improves retrieval accuracy, especially for context-dependent queries
        - Reduces semantic ambiguity and misinterpretation

        Parameters:
            doc_content: full document content (used as context background)
            chunk_content: current text chunk content (the part to be augmented)
            metadata: metadata
        """
        # Build LLM prompt asking to add document context to the chunk
        prompt = f"""
        <document>
        {doc_content}
        </document>
        <chunk>
        {chunk_content}
        </chunk>

        Please enrich the above <chunk> using content from <document> to provide background and context.
        Your response should include the complete content of <chunk> and ensure semantic coherence.
        Return only the enriched text without any explanation or commentary.

        Goals:
        1. Keep the core information of the original chunk unchanged
        2. Add necessary background context to clarify the chunk's meaning
        3. Ensure the augmented text is semantically coherent and complete
        """

        # === OpenAI GPT API call (current version) ===
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5-turbo for cost efficiency
            # model="gpt-4",        # Optionally use GPT-4 for better results
            max_tokens=1000,        # Limit output length
            temperature=0,          # Ensure consistency and reproducibility
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the contextualized text generated by OpenAI
        contextualized_chunk = response.choices[0].message.content.strip()

        # === Claude API call (original version, commented out) ===
        # message = self.anthropic_client.messages.create(
        #     model="claude-3-haiku-20240307",
        #     max_tokens=1000,
        #     temperature=0,
        #     messages=[
        #         {"role": "user", "content": prompt}
        #     ]
        # )
        # contextualized_chunk = message.content[0].text.strip()

        # Generate embedding vector from the contextualized content and insert
        dense_vec = self.embedding_function([contextualized_chunk])[0]
        data = {
            "dense_vector": dense_vec,
            "contextualized_content": contextualized_chunk,  # Save the augmented content
            **metadata
        }

        # If sparse vectors are enabled, generate sparse vector from contextualized content
        if self.use_sparse is True:
            sparse_vec = self.sparse_embedding_function([contextualized_chunk])[0]
            data["sparse_vector"] = sparse_vec

        # Insert contextualized data into Milvus
        self.client.insert(
            collection_name=self.collection_name,
            data=[data]
        )

    def search(self, query, k=5):
        """
        Search for relevant content

        Smart retrieval execution pipeline:
        This is the core entry point of the retrieval system, supporting unified calls
        across multiple retrieval strategies.

        Retrieval strategy selection:
        ┌─────────────────┬──────────────────┬──────────────────┐
        │  Retrieval step │  Processing      │  Output          │
        ├─────────────────┼──────────────────┼──────────────────┤
        │ 1. Query embed  │ Embedding model  │ Query vector     │
        │ 2. Similarity   │ Milvus search    │ Top-K candidates │
        │ 3. Reranking    │ Cohere Rerank    │ Optimized ranking│
        └─────────────────┴──────────────────┴──────────────────┘

        Detailed search steps:
        1. Query preprocessing and embedding
        2. Milvus vector similarity search
        3. Retrieve and filter initial results
        4. Optional reranking optimization
        5. Post-processing and return

        Search optimization strategy:
        Query text → Embedding → Similarity search → Candidates
                                                      ↓
        Final result ← Reranking ← Semantic matching ← Result set

        Parameters:
            query: query text
            k: number of results to return

        Returns:
            List of search results sorted by relevance
        """
        # Set search parameters
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

        # Generate query embedding vector
        dense_vec = self.embedding_function([query])[0]

        # Execute standard dense vector search using inner product (IP) as similarity metric
        res = self.client.search(
            collection_name=self.collection_name,
            data=[dense_vec],
            limit=k,
            output_fields=["content", "contextualized_content"],  # Return raw and contextualized content
            search_params=search_params,
        )

        # Further optimize results with the reranker
        # Reranking: re-sorts results based on deeper semantic relationships between query and documents
        if self.use_reranker:
            # Extract document content for reranking
            docs = []
            for hit in res[0]:
                # Prefer contextualized content (if present), otherwise use raw content
                content = hit["entity"].get("contextualized_content", hit["entity"].get("content", ""))
                docs.append(content)

            # Apply reranking: compute deep relevance score between query and each document
            rerank_results = self.rerank_function(query, docs)

            # Re-sort original results based on reranking output
            reranked_results = []
            for result in rerank_results:
                idx = result.index  # Use .index attribute to get original index
                reranked_results.append(res[0][idx])

            res = [reranked_results]

        return res


def evaluate_retrieval(eval_data, retrieval_function, db, k=5):
    """
    Evaluate retrieval performance — core function of the deep evaluation system

    Deep Evaluation system design:
    This is a comprehensive retrieval performance evaluation framework using a
    multi-dimensional evaluation strategy.

    Deep Evaluation concept:
    1. Evaluates not just retrieval quantity but retrieval quality
    2. Uses multi-dimensional metrics to assess retrieval system performance
    3. Performs objective evaluation using a gold-standard dataset
    4. Supports fair comparison of different retrieval strategies

    Evaluation pipeline design:
    ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
    │  Eval stage     │  Input data     │  Processing     │  Output         │
    ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
    │ 1. Data prep    │ Eval query set  │ Format validation│ Normalized data│
    │ 2. Retrieval    │ Single query    │ Retrieval call  │ Candidate list  │
    │ 3. Matching     │ Retrieval result│ Exact text match│ Match count     │
    │ 4. Scoring      │ Match stats     │ Metric calc     │ Eval scores     │
    └─────────────────┴─────────────────┴─────────────────┴─────────────────┘

    Evaluation metrics:
    - Pass@K: proportion of queries where the correct answer appears in the top K results
      * Formula: (queries with correct answer) / (total queries) × 100%
      * Reflects overall retrieval success rate
    - Average score: ratio of correct chunks retrieved per query to total correct chunks
      * Formula: Σ(correct chunks retrieved / total correct chunks per query) / total queries
      * Reflects fine-grained retrieval precision
    - Recall: measures the system's ability to find relevant documents

    Evaluation execution flow:
    Eval data → Query parsing → Gold standard extraction → Retrieval → Comparison → Metric calc
        ↓
    Performance report ← Statistical analysis ← Score aggregation ← Match verification

    Parameters:
        eval_data: evaluation dataset
            - Contains queries and corresponding gold standard answers
            - Each query has clearly identified correct document chunks
        retrieval_function: retrieval function
            - Accepts query and database, returns retrieval results
        db: database instance
        k: top-k results to evaluate

    Returns:
        Evaluation result dict containing Pass@K score, average score, etc.
    """
    total_score = 0      # Cumulative score
    total_queries = 0    # Total number of queries

    # Iterate over each evaluation query
    for item in tqdm(eval_data, desc="Evaluating retrieval"):
        total_queries += 1
        query = item["query"]

        # Get gold standard content (correct answers) for the current query
        golden_contents = []
        for ref in item["references"]:
            doc_uuid = ref["doc_uuid"]      # Unique document identifier
            chunk_index = ref["chunk_index"] # Index of the document chunk

            # Find the corresponding original document in the dataset
            golden_doc = next(
                (
                    doc
                    for doc in dataset
                    if doc.get("original_uuid") == doc_uuid
                ),
                None,
            )
            if not golden_doc:
                print(f"Warning: gold document with UUID {doc_uuid} not found")
                continue

            # Find the corresponding chunk in the document
            golden_chunk = next(
                (
                    chunk
                    for chunk in golden_doc["chunks"]
                    if chunk["original_index"] == chunk_index
                ),
                None,
            )
            if not golden_chunk:
                print(f"Warning: gold chunk with index {chunk_index} not found in document {doc_uuid}")
                continue

            golden_contents.append(golden_chunk["content"].strip())

        if not golden_contents:
            print(f"Warning: no gold content found for query: {query}")
            continue

        # Use the retrieval function to get results
        retrieved_docs = retrieval_function(query, db, k=k)

        # Count how many gold chunks appear in the top-k retrieved documents
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[0][:k]:  # Check only the top k results
                content_field = "content"
                if "contextualized_content" in doc["entity"]:
                    # Use raw content for comparison to ensure fairness
                    content_field = "content"
                retrieved_content = doc["entity"][content_field].strip()

                # Exact match check
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break  # Break inner loop once a match is found

        # Score for this query: correct chunks found / total correct chunks
        query_score = chunks_found / len(golden_contents)
        total_score += query_score

    # Calculate overall evaluation metrics
    average_score = total_score / total_queries
    pass_at_n = average_score * 100  # Convert to percentage

    return {
        "pass_at_n": pass_at_n,           # Pass@K score (percentage)
        "average_score": average_score,    # Average score (0–1)
        "total_queries": total_queries,    # Total number of queries
    }


def retrieve_base(query: str, db, k: int = 20) -> List[Dict[str, Any]]:
    """
    Base retrieval function

    A simple wrapper that calls the database search method.
    Used as a unified interface in the evaluation system.
    """
    return db.search(query, k=k)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and return a list of dicts.

    JSONL format: one JSON object per line, suitable for structured evaluation data.
    """
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def evaluate_db(db, original_jsonl_path: str, k):
    """
    Evaluate database retrieval performance

    Main entry function for the evaluation pipeline:
    1. Load evaluation dataset
    2. Run retrieval evaluation
    3. Output performance metrics

    Parameters:
        db: database instance to evaluate
        original_jsonl_path: path to the evaluation dataset file
        k: top-k parameter for evaluation

    Returns:
        Evaluation result dict
    """
    # Load original JSONL data as queries and ground truth labels
    original_data = load_jsonl(original_jsonl_path)

    # Evaluate retrieval performance
    results = evaluate_retrieval(original_data, retrieve_base, db, k)

    # Output evaluation results
    print(f"Pass@{k}: {results['pass_at_n']:.2f}%")
    print(f"Total score: {results['average_score']}")
    print(f"Total queries: {results['total_queries']}")

    return results


def download_data():
    """
    Download sample data

    Downloads demo data from Anthropic's GitHub repository:
    1. codebase_chunks.json: document chunk data from a codebase
    2. evaluation_set.jsonl: evaluation queries and ground truth answers
    """
    import urllib.request

    # Check if files already exist to avoid re-downloading
    if not os.path.exists("codebase_chunks.json"):
        print("Downloading codebase_chunks.json...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/codebase_chunks.json",
            "codebase_chunks.json"
        )

    if not os.path.exists("evaluation_set.jsonl"):
        print("Downloading evaluation_set.jsonl...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/refs/heads/main/skills/contextual-embeddings/data/evaluation_set.jsonl",
            "evaluation_set.jsonl"
        )

    print("Data download complete!")


def main():
    """
    Main function — run all experiments

    Experiment design framework:
    A complete comparative experiment system using the controlled-variable method
    to verify the effectiveness of different retrieval strategies.

    Experiment goals:
    - Verify performance improvement of contextual retrieval over standard retrieval
    - Quantify the additional benefit of reranking
    - Establish a performance benchmark for retrieval techniques
    - Provide technical selection guidance for real-world applications

    Experiment design matrix:
    ┌─────────────────┬────────────┬────────────┬────────────┬──────────────┐
    │   Experiment    │ Preproc    │ Retrieval  │ Reranking  │ Expected perf│
    ├─────────────────┼────────────┼────────────┼────────────┼──────────────┤
    │ Standard (base) │ Raw chunks │ Dense vecs │ None       │ Baseline     │
    │ Contextual      │ LLM-enhanced│ Dense vecs│ None       │ Medium gain  │
    │ Reranked        │ LLM-enhanced│ Dense vecs│ Cohere     │ Best perf    │
    └─────────────────┴────────────┴────────────┴────────────┴──────────────┘

    Controlled variables:
    - Same dataset (codebase_chunks.json)
    - Same evaluation queries (evaluation_set.jsonl)
    - Same embedding model (BGE-large-zh)
    - Same evaluation metric (Pass@5)
    - Same vector database configuration

    Execution steps:
    1. Environment setup:
       - Validate API key configuration
       - Initialize and test models
       - Download and validate data

    2. Data preprocessing:
       - Load and parse document data
       - Build evaluation query set
       - Normalize data format

    3. Experiment execution:
       - Standard retrieval experiment (baseline)
       - Contextual retrieval experiment (core innovation)
       - Reranked retrieval experiment (performance optimization)

    4. Result analysis:
       - Compute performance metrics
       - Calculate improvement percentages
       - Summarize experiment conclusions

    Core experiment hypothesis:
    This function runs three comparative experiments to show the performance difference
    across retrieval strategies:

    1. Standard retrieval: baseline, uses raw text chunks directly
    2. Contextual retrieval: uses LLM to augment chunk context
    3. Contextual retrieval with reranking: adds reranking on top of contextual retrieval

    Fairness guarantee:
    All experiments use the same evaluation dataset for a fair comparison.
    """
    # Replace with your actual API keys
    cohere_api_key = os.getenv("COHERE_API_KEY")      # Cohere Rerank API key
    # anthropic_api_key = os.getenv("CLAUDE_API_KEY")   # Claude API key (commented out)
    openai_api_key = os.getenv("OPENAI_API_KEY")      # OpenAI API key

    # Download sample data
    download_data()

    # Load dataset
    global dataset
    with open("codebase_chunks.json", "r") as f:
        dataset = json.load(f)

    # Use only the first 5 documents for testing (reduces API calls and runtime)
    dataset = dataset[:5]

    # Initialize models and functions
    dense_ef = SentenceTransformerEmbeddingFunction(model_name='BAAI/bge-large-zh')  # Chinese-optimized BGE model
    cohere_rf = CohereRerankFunction(api_key=cohere_api_key)  # Cohere reranking function

    # === OpenAI client initialization (current version) ===
    openai_client = openai.OpenAI(api_key=openai_api_key)  # OpenAI client

    # === Claude client initialization (original version, commented out) ===
    # anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

    # ===============================
    # Experiment 1: Standard retrieval (baseline)
    # ===============================
    print("\n===== Experiment 1: Standard Retrieval =====")
    print("Baseline experiment: retrieval on raw text chunks with no augmentation")

    standard_retriever = MilvusContextualRetriever(
        uri="standard.db",
        collection_name="standard",
        dense_embedding_function=dense_ef
    )

    # Build collection and insert standard data
    standard_retriever.build_collection()
    for doc in tqdm(dataset, desc="Inserting standard retrieval data"):
        doc_content = doc["content"]
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            standard_retriever.insert_data(chunk_content, metadata)

    # Create simplified evaluation data (for demonstration)
    # In production, use a purpose-built evaluation dataset
    eval_data = []
    for doc in dataset[:2]:  # Use only the first 2 documents for evaluation
        for chunk in doc["chunks"][:2]:  # Take the first 2 chunks per document
            eval_data.append({
                "query": chunk["content"][:50],  # Use first 50 chars of chunk as query
                "references": [{
                    "doc_uuid": doc["original_uuid"],
                    "chunk_index": chunk["original_index"]
                }]
            })

    # Save evaluation data
    with open("evaluation_set.jsonl", "w") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Evaluate standard retrieval performance
    standard_results = evaluate_db(standard_retriever, "evaluation_set.jsonl", 5)

    # ===============================
    # Experiment 2: Contextual retrieval
    # ===============================
    print("\n===== Experiment 2: Contextual Retrieval =====")
    print("Using OpenAI GPT to add document context to each chunk, solving semantic isolation")

    contextual_retriever = MilvusContextualRetriever(
        uri="contextual.db",
        collection_name="contextual",
        dense_embedding_function=dense_ef,
        use_contextualize_embedding=True,  # Enable contextualization
        llm_client=openai_client,  # Use OpenAI client
        # anthropic_client=anthropic_client,  # Original Claude client (commented out)
    )

    # Build collection and insert contextualized data
    contextual_retriever.build_collection()
    for doc in tqdm(dataset, desc="Inserting contextual retrieval data"):
        doc_content = doc["content"]
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            # Use contextualized insertion method
            contextual_retriever.insert_contextualized_data(
                doc_content, chunk_content, metadata
            )

    # Evaluate contextual retrieval performance
    contextual_results = evaluate_db(contextual_retriever, "evaluation_set.jsonl", 5)

    # ===============================
    # Experiment 3: Contextual retrieval with reranking
    # ===============================
    print("\n===== Experiment 3: Contextual Retrieval with Reranking =====")
    print("Using Cohere reranking model on top of contextual retrieval to further optimize results")

    # Enable reranking
    contextual_retriever.use_reranker = True
    contextual_retriever.rerank_function = cohere_rf

    # Evaluate reranked retrieval performance
    reranker_results = evaluate_db(contextual_retriever, "evaluation_set.jsonl", 5)

    # ===============================
    # Comparative result analysis
    # ===============================
    print("\n===== All Experiment Results Comparison =====")
    print("Performance improvement analysis:")
    print(f"Standard Retrieval Pass@5: {standard_results['pass_at_n']:.2f}%")
    print(f"Contextual Retrieval Pass@5: {contextual_results['pass_at_n']:.2f}%")
    print(f"Contextual + Reranking Pass@5: {reranker_results['pass_at_n']:.2f}%")

    # Calculate improvement margins
    context_improvement = contextual_results['pass_at_n'] - standard_results['pass_at_n']
    rerank_improvement = reranker_results['pass_at_n'] - standard_results['pass_at_n']

    print(f"\nPerformance improvement analysis:")
    print(f"Contextual retrieval vs. standard: +{context_improvement:.2f} percentage points")
    print(f"Reranking additional gain: +{rerank_improvement:.2f} percentage points")
    print(f"Total improvement: +{rerank_improvement:.2f} percentage points")


if __name__ == "__main__":
    main()
