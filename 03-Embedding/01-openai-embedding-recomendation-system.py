import os
import openai
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# Load the user review dataset
df = pd.read_csv("90-Data/chronicles-of-gods/user-reviews.csv")

# Load the game description file
with open("90-Data/chronicles-of-gods/game-instructions.json", "r") as f:
    game_descriptions = json.load(f)

# Define a function to get embeddings
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Get embeddings for all games
unique_games = df['game_title'].unique().tolist()
target_game = "Killing God: Hu Sun"  # Target game name
if target_game not in unique_games:
    unique_games.append(target_game)  # Ensure the target game is in the list
game_embeddings = {}
for game in unique_games:
    description = game_descriptions[game]
    game_embeddings[game] = np.array(get_embedding(description))

# Compute user embedding vectors (average of all game description embeddings the user has reviewed)
user_vectors = {}
for user_id, group in df.groupby("user_id"):
    user_game_vecs = []
    for idx, row in group.iterrows():
        g_title = row['game_title']
        g_vec = game_embeddings[g_title]
        user_game_vecs.append(g_vec)
    user_vectors[user_id] = np.mean(np.array(user_game_vecs), axis=0)

# Get the embedding for the target game
target_vector = game_embeddings[target_game]
# Compute cosine similarity between each user's embedding and the target game's embedding
results = []
for user_id, u_vec in user_vectors.items():
    u_vec_reshaped = u_vec.reshape(1, -1)
    t_vec = target_vector.reshape(1, -1)
    similarity = cosine_similarity(u_vec_reshaped, t_vec)[0,0]
    results.append((user_id, similarity))

# Sort and find the users most likely to enjoy the target game
result_df = pd.DataFrame(results, columns=["user_id", f"similarity_to_{target_game}"])
result_df = result_df.sort_values(by=f"similarity_to_{target_game}", ascending=False)
print(f"\nTop 5 users most likely to enjoy {target_game}:")
print(result_df.head())
