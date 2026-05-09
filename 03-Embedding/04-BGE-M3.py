from FlagEmbedding import BGEM3FlagModel

def main():
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    passage = ["The Monkey King unleashes the Flame Fist, repelling the monster; then activates the Iron Body to withstand the divine weapon attack."]

    # Encode text and obtain sparse and dense embeddings
    passage_embeddings = model.encode(
        passage,
        return_sparse=True,     # Return sparse embeddings
        return_dense=True,      # Return dense embeddings
        return_colbert_vecs=True  # Return multi-vector embeddings
    )
    # Extract sparse, dense, and multi-vector embeddings separately
    dense_vecs = passage_embeddings["dense_vecs"]
    sparse_vecs = passage_embeddings["lexical_weights"]
    colbert_vecs = passage_embeddings["colbert_vecs"]
    # Display examples of sparse and dense embeddings
    print("Dense embedding dimensions:", dense_vecs[0].shape)
    print("Dense embedding first 10 dimensions:", dense_vecs[0][:10])  # Show only the first 10 dimensions

    print("Sparse embedding total length:", len(sparse_vecs[0]))
    print("Sparse embedding first 10 non-zero values:", list(sparse_vecs[0].items())[:10])  # Show only first 10 non-zero values

    print("Multi-vector embedding dimensions:", colbert_vecs[0].shape)
    print("Multi-vector embedding first 2 vectors:", colbert_vecs[0][:2])  # Show only first 2 multi-vector embeddings

if __name__ == '__main__':
    main()
