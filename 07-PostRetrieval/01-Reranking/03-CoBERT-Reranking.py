from transformers import AutoTokenizer, AutoModel
import torch

"""
ColBERT (Contextualized Late Interaction over BERT) Reranking Algorithm Implementation

ColBERT is an innovative architecture that combines BERT's deep semantic understanding
with efficient retrieval, using a "late interaction" mechanism.

Core innovations:
1. Early separate encoding: queries and documents are encoded independently, allowing pre-computation of document embeddings
2. Late fine-grained interaction: fine-grained token-level interaction in the vector space
3. MaxSim operation: for each token in the query, find the most similar token in the document for matching

Technical advantages:
- High efficiency: documents can be pre-encoded and indexed, only the query needs to be encoded at query time
- High precision: preserves token-level fine-grained interactions without losing semantic information
- Scalable: supports efficient retrieval over large-scale document collections

Comparison with other methods:
- vs CrossEncoder: faster, supports pre-computing document embeddings
- vs Bi-Encoder: more fine-grained interactions, considering token-level matching
- vs traditional methods: stronger semantic understanding capability, supports fuzzy matching

Applicable scenarios:
- First stage of large-scale document retrieval
- Applications that need to balance precision and efficiency
- Real-time retrieval systems sensitive to latency
"""

print("Initializing ColBERT reranking model...")

# 1. Load BERT model and tokenizer
print("Loading BERT base model...")
model_name = "bert-base-uncased"  # Base BERT model, can be replaced with ColBERT-specific fine-tuned model
print(f"Using model: {model_name}")
print("  Note: In practical applications, it is recommended to use models specifically fine-tuned for ColBERT")
print("  For example: 'colbert-ir/colbertv2.0' or other ColBERT optimized models")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("Model loaded successfully")

# 2. Prepare test data
print("\nPreparing test data...")
query = "What are the famous tourist attractions in Shanxi?"
documents = [
    "Mount Wutai is one of the four major Buddhist mountains in China, famous as the sacred site of Manjushri Bodhisattva.",
    "Yungang Grottoes is one of the three major stone grottoes in China, renowned for its exquisite Buddhist sculptures.",
    "Pingyao Ancient City is one of the most intact ancient county cities in China, listed as a World Cultural Heritage site."
]

print(f"Query: {query}")
print(f"Number of candidate documents: {len(documents)}")
for i, doc in enumerate(documents, 1):
    print(f"  Document {i}: {doc}")

def encode_text(texts, max_length=128):
    """
    ColBERT text encoding function

    Function: Encode text into contextualized token embeddings

    Parameters:
        texts (list): List of texts to encode
        max_length (int): Maximum sequence length

    Returns:
        torch.Tensor: Embedding tensor of shape [batch_size, seq_len, hidden_size]

    ColBERT encoding characteristics:
        1. Preserve independent embeddings for all tokens (not just [CLS])
        2. Each token has complete contextual information
        3. Prepare for subsequent token-level interactions
    """
    print(f"  Encoding text, max sequence length: {max_length}")

    inputs = tokenizer(
        texts,
        return_tensors="pt",      # Return PyTorch tensors
        padding=True,             # Pad to uniform length
        truncation=True,          # Truncate overly long sequences
        max_length=max_length
    )

    print(f"    Input shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        outputs = model(**inputs)

    # Return hidden states for all tokens (not just [CLS])
    embeddings = outputs.last_hidden_state
    print(f"    Output embedding shape: {embeddings.shape}")

    return embeddings

print(f"\nStarting ColBERT encoding process...")

# 3. Encode queries and documents separately
print(f"\n1. Encoding query...")
query_embeddings = encode_text([query])  # [1, seq_len, hidden_size]

print(f"\n2. Encoding documents...")
doc_embeddings = encode_text(documents)  # [num_docs, seq_len, hidden_size]

def calculate_similarity(query_emb, doc_embs):
    """
    ColBERT similarity calculation function (simplified version)

    Function: Calculate ColBERT similarity scores between query and documents

    Parameters:
        query_emb (torch.Tensor): Query embedding [1, seq_len, hidden_size]
        doc_embs (torch.Tensor): Document embeddings [num_docs, seq_len, hidden_size]

    Returns:
        list: Similarity scores for each document

    ColBERT similarity calculation steps:
        1. For each query token, find the maximum similarity with all document tokens (MaxSim)
        2. Sum the MaxSim scores of all tokens in the query as the final score

    Note: This uses a simplified version (average pooling). The complete ColBERT uses MaxSim operations
    """
    print(f"\n3. Calculating ColBERT similarity...")
    print("  Note: This is a simplified implementation of ColBERT; the complete version uses MaxSim operations")

    # Simplified version: use average pooling instead of complete token-level interactions
    # In the actual ColBERT, precise MaxSim computation would be performed here
    query_emb_pooled = query_emb.mean(dim=1)  # [1, hidden_size]
    doc_embs_pooled = doc_embs.mean(dim=1)    # [num_docs, hidden_size]

    print(f"    Query pooled shape: {query_emb_pooled.shape}")
    print(f"    Document pooled shape: {doc_embs_pooled.shape}")

    # L2 normalization to ensure cosine similarity calculation
    query_emb_norm = query_emb_pooled / query_emb_pooled.norm(dim=1, keepdim=True)
    doc_embs_norm = doc_embs_pooled / doc_embs_pooled.norm(dim=1, keepdim=True)

    # Calculate cosine similarity
    scores = torch.mm(query_emb_norm, doc_embs_norm.t())  # [1, num_docs]

    print(f"    Similarity scores shape: {scores.shape}")

    return scores.squeeze().tolist()

# 4. Calculate similarity scores
scores = calculate_similarity(query_embeddings, doc_embeddings)

# 5. Sort documents
print(f"\nSorting documents by ColBERT similarity score...")
ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

# 6. Output sorted results
print(f"\n{'='*60}")
print(f"ColBERT Reranking Results")
print(f"{'='*60}")
print(f"Query: {query}")
print(f"\nSorted results (in descending order of similarity score):")

for rank, (doc, score) in enumerate(ranked_docs, start=1):
    print(f"\nRank {rank}:")
    print(f"   ColBERT similarity score: {score:.4f}")
    print(f"   Document content: {doc}")

    # Explain score meaning
    if score > 0.8:
        relevance_level = "Highly relevant"
    elif score > 0.6:
        relevance_level = "Moderately relevant"
    else:
        relevance_level = "Low relevance"
    print(f"   Relevance level: {relevance_level}")

print(f"\nColBERT Algorithm Summary:")
print("- Efficient retrieval: supports document pre-encoding, low query latency")
print("- Fine-grained interactions: preserves token-level semantic interaction information")
print("- Scalability: suitable for retrieval over large-scale document collections")
print("- Performance balance: achieves a good balance between precision and efficiency")
print("- Complete implementation: recommended to use specifically fine-tuned ColBERT models")
print("- Optimization suggestion: use MaxSim operations instead of simplified average pooling in practical applications")
