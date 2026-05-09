from collections import Counter
import math
# Battle logs of the Monkey King
battle_logs = [
    "Monkey King，unleash，Flame Fist，repel，monster；then activate，Iron Body，withstand，divine weapon，attack。",
    "Monster，use，Ice Arrow，attack，Monkey King，but repelled by，Flame Fist，counterattack，defeated。",
    "Monkey King，summon，Flame Fist，and，Devastating Roar，defeat，monster，then，collect，monster，essence。"
]
# Hyperparameters
k1 = 1.5
b = 0.75
# Build vocabulary
vocabulary = set(word for log in battle_logs for word in log.split("，"))
vocab_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
# Compute IDF
N = len(battle_logs)
df = Counter(word for log in battle_logs for word in set(log.split("，")))
idf = {word: math.log((N - df[word] + 0.5) / (df[word] + 0.5) + 1) for word in vocabulary}
# Log length information
avg_log_len = sum(len(log.split("，")) for log in battle_logs) / N
# BM25 sparse embedding generation
def bm25_sparse_embedding(log):
    tf = Counter(log.split("，"))
    log_len = len(log.split("，"))
    embedding = {}
    for word, freq in tf.items():
        if word in vocabulary:
            idx = vocab_to_idx[word]
            score = idf[word] * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * log_len / avg_log_len))
            embedding[idx] = score
    return embedding
# Generate sparse vectors
for log in battle_logs:
    sparse_embedding = bm25_sparse_embedding(log)
print(f"Sparse embedding: {sparse_embedding}")
