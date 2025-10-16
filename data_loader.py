import pandas as pd
import streamlit as st
from data_processor import DataProcessor
from recommendation_engine import HybridRecommendationEngine

def load_sample_data():
    """
    Load sample data from the extracted CSV files.
    Returns: tuple of (users_df, posts_df, processed_data)
    """
    try:
        users_df = pd.read_csv('users.csv')
        posts_df = pd.read_csv('posts.csv')
        
        processor = DataProcessor()
        processed_data = processor.process_data(users_df, posts_df)
        
        return users_df, posts_df, processed_data, processor
    except Exception as e:
        raise Exception(f"Error loading sample data: {str(e)}")

def generate_recommendations_for_all_users(users_df, posts_df, interaction_matrix, 
                                           n_recommendations=10, 
                                           collab_weight=0.4, 
                                           content_weight=0.4, 
                                           popularity_weight=0.2):
    """
    Generate recommendations for all users in the dataset.
    
    Parameters:
    - users_df: DataFrame with user information
    - posts_df: DataFrame with post information
    - interaction_matrix: Sparse matrix of user-post interactions
    - n_recommendations: Number of recommendations per user
    - collab_weight: Weight for collaborative filtering
    - content_weight: Weight for content-based filtering
    - popularity_weight: Weight for popularity
    
    Returns:
    - recommendations_df: DataFrame with user_id and recommended_post_ids
    - engine: Trained recommendation engine
    """
    engine = HybridRecommendationEngine(
        collab_weight=collab_weight,
        content_weight=content_weight,
        popularity_weight=popularity_weight
    )
    
    engine.fit(users_df, posts_df, interaction_matrix)
    
    recommendations = []
    
    for user_id in users_df['user_id']:
        user_recs = engine.get_recommendations(user_id, n_recommendations)
        recommendations.append({
            'user_id': user_id,
            'recommended_post_ids': ','.join(map(str, user_recs))
        })
    
    recommendations_df = pd.DataFrame(recommendations)
    
    return recommendations_df, engine
