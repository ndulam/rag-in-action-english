from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

"""
CrossEncoder Reranking Algorithm Implementation

CrossEncoder is a bidirectional encoder reranking model based on BERT, specifically designed
to calculate relevance scores for query-document pairs.

Core principles:
1. Input the query and document as a whole into the BERT model
2. Use the output of the [CLS] token to predict relevance scores
3. Train end-to-end, capable of capturing deep interactions between queries and documents

Differences from other methods:
- Compared to bi-encoder (Bi-Encoder): CrossEncoder can better model the interaction between query and document
- Compared to traditional BM25: can understand semantic similarity, not just relying on keyword matching
- Compared to simple vector similarity: considers positional information and contextual relationships of queries and documents

Advantages:
- High precision: can precisely model the relevance of query-document pairs
- Strong semantic understanding: based on pre-trained language models with powerful semantic understanding
- Good adaptability: can be fine-tuned for specific domains

Disadvantages:
- High computational overhead: requires separate encoding for each query-document pair
- Poor real-time performance: not suitable for the first stage of large-scale retrieval, typically used in the reranking stage
"""

print("Initializing CrossEncoder reranking model...")

# 1. Load pre-trained CrossEncoder model
print("Loading pre-trained model...")
model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Small model trained on MS MARCO dataset
print(f"Using model: {model_name}")
print("  - This model has been fine-tuned on the MS MARCO passage retrieval task")
print("  - Specifically optimized for calculating query-passage relevance scores")
print("  - Balances model size and performance, suitable for production environments")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print("Model loaded successfully")

# 2. Prepare test data
print("\nPreparing test data...")
query = "What are the famous tourist attractions in Shanxi?"
documents = [
    "Mount Wutai is one of the four major Buddhist mountains in China, famous as the sacred site of Manjushri Bodhisattva.",
    "Yungang Grottoes is one of the three major stone grottoes in China, renowned for its exquisite Buddhist sculptures.",
    "Pingyao Ancient City is one of the most intact ancient county cities in China, listed as a World Cultural Heritage site.",
]

print(f"Query: {query}")
print(f"Number of candidate documents: {len(documents)}")
for i, doc in enumerate(documents, 1):
    print(f"  Document {i}: {doc}")

def encode_and_score(query, docs):
    """
    CrossEncoder relevance scoring function

    Function: Calculate the relevance score between the query and each document

    Parameters:
        query (str): User query
        docs (list): List of candidate documents

    Returns:
        list: Relevance scores corresponding to each document

    Workflow:
        1. Concatenate the query and document in "[CLS] query [SEP] document [SEP]" format
        2. Encode through tokenizer, generating input_ids, attention_mask, etc.
        3. Input to BERT model, obtain the output at the [CLS] position
        4. Calculate relevance score through the classification head
        5. Higher score indicates stronger relevance
    """
    print(f"\nStarting relevance score calculation for {len(docs)} documents...")
    scores = []

    for i, doc in enumerate(docs, 1):
        print(f"  Processing document {i}/{len(docs)}...")

        # Combine query and document into BERT input format
        # Format: [CLS] query [SEP] document [SEP]
        inputs = tokenizer(
            query,
            doc,
            return_tensors="pt",           # Return PyTorch tensors
            truncation=True,               # Truncate overly long inputs
            max_length=512,                # BERT maximum input length
            padding="max_length"           # Pad to maximum length
        )

        # Forward pass to calculate relevance score
        with torch.no_grad():  # Disable gradient computation to save memory
            outputs = model(**inputs)
            # Get logits (raw scores), usually the first element is the relevance score
            score = outputs.logits[0][0].item()
            scores.append(score)

        print(f"    Query-document pair relevance score: {score:.4f}")
        print(f"    Input length: {len(inputs['input_ids'][0])} tokens")

    print("Relevance score calculation complete")
    return scores

# 3. Execute CrossEncoder reranking
print(f"\nExecuting CrossEncoder reranking...")
scores = encode_and_score(query, documents)

# 4. Sort documents by score
print(f"\nSorting documents by relevance score...")
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# 5. Output reranking results
print(f"\n{'='*60}")
print(f"CrossEncoder Reranking Results")
print(f"{'='*60}")
print(f"Query: {query}")
print(f"\nSorted results (in descending order of relevance score):")

for rank, (doc, score) in enumerate(ranked_docs, start=1):
    print(f"\nRank {rank}:")
    print(f"   Relevance score: {score:.4f}")
    print(f"   Document content: {doc}")

    # Explain score meaning
    if score > 0:
        relevance_level = "Highly relevant"
    elif score > -2:
        relevance_level = "Moderately relevant"
    else:
        relevance_level = "Low relevance"
    print(f"   Relevance level: {relevance_level}")

print(f"\nCrossEncoder Reranking Summary:")
print("- Deep semantic understanding: captures fine-grained interactions between queries and documents")
print("- Precise relevance modeling: end-to-end training for accurate relevance scores")
print("- Context-aware: considers positional information and contextual relationships of words")
print("- Computationally intensive: each query-document pair requires separate encoding")
print("- Best practice: use for fine reranking of initial retrieval results")
