import pandas as pd
import numpy as np
import requests
from sklearn.cluster import KMeans

# 1. Configure Jina API
url = 'https://api.jina.ai/v1/embeddings'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer jina_4a0adace937d43299b955eb9146386a54B2Ubzak2NcPXcTekETSbKeDLtep'
}

# 2. Load game description data
df = pd.read_csv("90-Data/chronicles-of-gods/game-description.csv")
texts = df['description'].tolist()

# 3. Get text embeddings
data = {
    "model": "jina-embeddings-v3",
    "task": "text-matching",
    "dimensions": 1024,
    "normalized": True,
    "input": texts
}

response = requests.post(url, headers=headers, json=data)

if response.status_code != 200:
    raise RuntimeError(f"API call failed: {response.status_code} - {response.text}")

embeddings = [item['embedding'] for item in response.json().get('data', [])]
if not embeddings:
    raise RuntimeError("API did not return embeddings")

embeddings = np.array(embeddings)

# 4. Clustering analysis
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(embeddings)

# 5. Print results
print("\nClustering results:")
for i, lbl in enumerate(labels):
    print(f"Cluster {lbl}: {texts[i]}")
