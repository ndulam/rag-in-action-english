from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank
import torch

"""
RankLLM Reranking Algorithm Implementation

RankLLM is a reranking method based on Large Language Models (LLM), leveraging the powerful
language understanding capabilities of LLMs for document reranking.

Core principles:
1. Utilize the LLM's deep language understanding capability to judge the relevance between queries and documents
2. Guide the LLM's ranking decisions through prompt engineering
3. Combined with the LLM's reasoning ability, capable of handling complex semantic relationships

Technical characteristics:
- Deep semantic understanding: based on the LLM's powerful language understanding capability
- Strong reasoning ability: can perform complex logical reasoning and semantic matching
- High flexibility: can adapt to different domains and tasks by adjusting prompts
- Good interpretability: LLM can provide reasons and explanations for rankings

Comparison with other methods:
- vs BERT-type models: deeper semantic understanding, can handle more complex reasoning
- vs traditional reranking: can understand context and implicit information
- vs embedding models: considers not just similarity, but also logical relationships

Applicable scenarios:
- Applications requiring extremely high precision
- Queries requiring complex reasoning
- Document retrieval with strong domain expertise
- Reranking tasks requiring interpretability

Notes:
- High computational cost (calling LLM API)
- Relatively high latency
- Requires proper prompt design
"""

print("Initializing RankLLM reranking system...")

# 1. Document loading and preprocessing
print("Loading and preprocessing documents...")
doc_path = "90-Data/shanxi-tourism/YunGangGrottoes.txt"
print(f"Document path: {doc_path}")

print("  Loading document using TextLoader...")
documents = TextLoader(doc_path).load()
print(f"  Successfully loaded document, original document count: {len(documents)}")

print("  Starting document splitting...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 500 characters per document chunk
    chunk_overlap=100     # 100 character overlap between chunks to maintain context continuity
)
texts = text_splitter.split_documents(documents)
print(f"  Number of document chunks after splitting: {len(texts)}")

# Add unique IDs to each document chunk
print("  Adding unique identifiers to document chunks...")
for idx, text in enumerate(texts):
    text.metadata["id"] = idx
    text.metadata["chunk_size"] = len(text.page_content)
print("  Document preprocessing complete")

# 2. Create vector retriever
print(f"\nCreating FAISS vector retriever...")
print("  Loading Chinese embedding model...")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")  # Use Chinese-optimized embedding model
print("  Building FAISS vector index...")
retriever = FAISS.from_documents(texts, embed_model).as_retriever(
    search_kwargs={"k": 20}  # First stage retrieves Top-20 documents
)
print(f"  Vector retriever created, will return Top-20 candidate documents")

# 3. GPU memory optimization (if using GPU)
print(f"\nOptimizing GPU memory usage...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("  GPU cache cleared")
else:
    print("  Currently using CPU mode")

# 4. Configure RankLLM reranker
print(f"\nConfiguring RankLLM reranker...")
print("  RankLLM configuration parameters:")
print("    - top_n: 3 (return top 3 documents finally)")
print("    - model: gpt (using GPT model)")
print("    - gpt_model: gpt-4o-mini (efficient GPT model)")

# Configure OPENAI proxy information
# OPENAI_BASE_URL = "https://vip.apiyi.com/v1"
# OPENAI_API_KEY = ""
# os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

compressor = RankLLMRerank(
    top_n=3,                    # Return the top 3 most relevant documents at the end
    model="gpt",                # Use GPT model for reranking
    gpt_model="gpt-4o-mini"     # Select efficient GPT-4o-mini model
)
print("  RankLLM reranker configuration complete")

# 5. Create contextual compression retriever
print(f"\nCreating contextual compression retriever...")
print("  Combining vector retriever and RankLLM reranker...")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,     # Use RankLLM as compressor (reranker)
    base_retriever=retriever        # Use FAISS as base retriever
)
print("  Retrieval pipeline built: FAISS retrieval -> RankLLM reranking")

# 6. Execute query and reranking
print(f"\nStarting query and reranking execution...")
query = "What are the famous sculptures at Yungang Grottoes?"
print(f"Query: {query}")

print(f"\nPhase 1 - FAISS vector retrieval:")
print("  Retrieving candidate documents based on semantic similarity...")

print(f"\nPhase 2 - RankLLM reranking:")
print("  Calling GPT model for deep semantic reranking...")
print("  Processing (LLM inference may take some time)...")

try:
    compressed_docs = compression_retriever.invoke(query)
    print(f"  RankLLM reranking complete")
    print(f"  Final {len(compressed_docs)} high-quality documents returned")

    # 7. Format and output reranking results
    def pretty_print_docs(docs):
        """
        Document output formatting function

        Function: Display reranked document results in an easy-to-read format

        Parameters:
            docs (list): List of reranked documents
        """
        print(f"\n{'='*60}")
        print(f"RankLLM Reranking Final Results")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Reranked documents (in descending order of relevance):")

        result_parts = []
        for i, doc in enumerate(docs, 1):
            doc_info = f"\nRank {i}:\n"
            doc_info += f"   Document content:\n{doc.page_content}\n"

            # Show document metadata
            if hasattr(doc, 'metadata') and doc.metadata:
                doc_info += f"   Document ID: {doc.metadata.get('id', 'Unknown')}\n"
                doc_info += f"   Content length: {doc.metadata.get('chunk_size', len(doc.page_content))} characters\n"
                if 'source' in doc.metadata:
                    doc_info += f"   Source file: {doc.metadata['source']}\n"

            result_parts.append(doc_info)

        return "\n" + ("-" * 100) + "\n".join(result_parts)

    # Output formatted results
    formatted_result = pretty_print_docs(compressed_docs)
    print(formatted_result)

except Exception as e:
    print(f"  RankLLM reranking failed: {str(e)}")
    print("  Possible causes:")
    print("    - GPT API key not configured or invalid")
    print("    - Network connection issue")
    print("    - API quota exhausted")
    print("    - Document content format issue")
    print("  Recommended checks:")
    print("    - OpenAI API key configuration")
    print("    - Network connection status")
    print("    - Whether document file exists")

# 8. Resource cleanup
print(f"\nCleaning up system resources...")
try:
    # Clean up RankLLM model (if needed)
    if 'compressor' in locals():
        del compressor
        print("  RankLLM model resources released")

    # Clear GPU cache again
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("  GPU cache cleared")

    print("  Resource cleanup complete")
except Exception as e:
    print(f"  Warning during resource cleanup: {str(e)}")

print(f"\nRankLLM Reranking Summary:")
print("- Deep understanding: based on LLM's powerful language understanding capabilities")
print("- Reasoning ability: capable of complex logical reasoning and semantic matching")
print("- High precision: leverages the most advanced language model technology")
print("- Interpretable: LLM can provide reasons and basis for rankings")
print("- High cost: requires LLM API calls, relatively higher cost")
print("- High latency: LLM inference time is relatively longer")
print("- Best practice: suitable for important queries requiring extremely high precision")
print("- Optimization suggestion: properly design prompts to improve reranking effectiveness")
