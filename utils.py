import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_genre_preferences(songs_df):
    """
    Calculate genre preferences for each user.
    
    Returns a dictionary where:
    - Keys are user IDs
    - Values are dictionaries mapping genres to (total_rating, avg_rating, count)
    """
    user_genre_preferences = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    
    # Iterate through songs dataframe
    for _, row in songs_df.iterrows():
        # Skip rows with missing genre information
        if not isinstance(row['genre'], str) or not row['genre']:
            continue
            
        # Handle potential string representation of a list
        genre = row['genre']
        if genre.startswith('[') and genre.endswith(']'):
            try:
                # Try to extract the actual genre string
                import ast
                genres = ast.literal_eval(genre)
                if isinstance(genres, list):
                    genre = genres[0]  # Take the first genre if it's a list
            except (ValueError, SyntaxError):
                # If parsing fails, use the original string
                pass
            
        # Process each user's rating
        for user_col in [col for col in songs_df.columns if col.startswith('user_')]:
            # Skip if rating is missing
            if pd.isna(row[user_col]):
                continue
                
            rating = row[user_col]
            
            # Update genre statistics for this user
            current_stats = user_genre_preferences[user_col][genre]
            current_stats[0] += rating  # total rating
            current_stats[2] += 1  # count
            current_stats[1] = current_stats[0] / current_stats[2]  # average rating
            
    return user_genre_preferences

def find_similar_users(target_user, user_genre_preferences, n=3):
    """
    Find the n most similar users to the target user based on genre preferences.
    
    Args:
        target_user: User ID to find similar users for
        user_genre_preferences: Output from calculate_genre_preferences
        n: Number of similar users to return
        
    Returns:
        List of n most similar user IDs
    """
    if target_user not in user_genre_preferences:
        return []
        
    # Get all genres across all users
    all_genres = set()
    for user_prefs in user_genre_preferences.values():
        all_genres.update(user_prefs.keys())
    all_genres = list(all_genres)
    
    # Create vectors of average ratings for each genre
    user_vectors = {}
    for user, prefs in user_genre_preferences.items():
        vector = []
        for genre in all_genres:
            # Use average rating if available, otherwise 0
            avg_rating = prefs[genre][1] if genre in prefs else 0
            vector.append(avg_rating)
        user_vectors[user] = vector
    
    # Skip if target user has no vector
    if target_user not in user_vectors:
        return []
    
    # Calculate cosine similarity between target user and all other users
    target_vector = np.array(user_vectors[target_user])
    similarities = {}
    
    for user, vector in user_vectors.items():
        if user == target_user:
            continue
        user_vector = np.array(vector)
        
        # Calculate cosine similarity manually
        # cos_sim = (A·B) / (||A|| * ||B||)
        dot_product = np.dot(target_vector, user_vector)
        norm_target = np.linalg.norm(target_vector)
        norm_user = np.linalg.norm(user_vector)
        
        # Avoid division by zero
        if norm_target == 0 or norm_user == 0:
            sim = 0
        else:
            sim = dot_product / (norm_target * norm_user)
            
        similarities[user] = sim
    
    # Return top n similar users
    similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:n]
    return [user for user, _ in similar_users]

def find_next_best_genre(target_user, user_genre_preferences, similar_users, k=2):
    """
    Find the next best genre for a user to explore based on similar users' preferences.
    
    Args:
        target_user: User ID to find next best genre for
        user_genre_preferences: Output from calculate_genre_preferences
        similar_users: List of similar user IDs
        k: Number of top genres to exclude from the target user's preferences
        
    Returns:
        Next best genre to explore
    """
    if not similar_users or target_user not in user_genre_preferences:
        return None
    
    # Get target user's top k genres
    target_prefs = user_genre_preferences[target_user]
    target_top_genres = sorted(target_prefs.items(), key=lambda x: x[1][1], reverse=True)[:k]
    target_top_genres = {genre for genre, _ in target_top_genres}
    
    # Collect genre preferences from similar users
    genre_scores = defaultdict(float)
    
    for user in similar_users:
        if user not in user_genre_preferences:
            continue
            
        user_prefs = user_genre_preferences[user]
        for genre, (_, avg_rating, count) in user_prefs.items():
            # Skip genres that are already in target user's top k
            if genre in target_top_genres:
                continue
            
            # Weight by similarity (users are already sorted by similarity)
            # Higher weight for users earlier in the similar_users list
            similarity_weight = 1 - (similar_users.index(user) / len(similar_users))
            genre_scores[genre] += avg_rating * similarity_weight
    
    # Return the highest scoring genre
    if not genre_scores:
        return None
        
    # Sort genres by score in descending order
    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return the best genre (first in the list)
    return sorted_genres[0][0] if sorted_genres else None
