import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import ast
import re

class DataProcessor:
    """
    Data processor for Boomm social platform datasets.
    Handles loading, cleaning, and preprocessing of users.csv and posts.csv.
    """
    
    def __init__(self):
        self.users_df = None
        self.posts_df = None
        self.interaction_matrix = None
    
    def process_data(self, users_df, posts_df):
        """
        Process users and posts datasets.
        
        Parameters:
        - users_df: Raw users DataFrame
        - posts_df: Raw posts DataFrame
        
        Returns:
        - Dictionary containing processed data
        """
        print("Processing users data...")
        processed_users = self._process_users(users_df)
        
        print("Processing posts data...")
        processed_posts = self._process_posts(posts_df)
        
        print("Creating interaction matrix...")
        interaction_matrix = self._create_interaction_matrix(processed_users, processed_posts)
        
        return {
            'users_df': processed_users,
            'posts_df': processed_posts,
            'interaction_matrix': interaction_matrix
        }
    
    def _process_users(self, users_df):
        """Process users dataset."""
        df = users_df.copy()
        
        # Basic info
        print(f"Users dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().to_dict()}")
        
        # Process interested_in column
        if 'interested_in' in df.columns:
            df['interests_list'] = df['interested_in'].apply(self._parse_comma_separated)
            
            # Show most common interests
            all_interests = []
            for interests in df['interests_list'].dropna():
                if isinstance(interests, list):
                    all_interests.extend([interest.strip().lower() for interest in interests])
            
            if all_interests:
                interest_counts = pd.Series(all_interests).value_counts()
                print(f"Most common interests: {interest_counts.head().to_dict()}")
        
        # Handle missing values
        df['name'] = df['name'].fillna('Unknown User')
        
        return df
    
    def _process_posts(self, posts_df):
        """Process posts dataset."""
        df = posts_df.copy()
        
        # Basic info
        print(f"Posts dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().to_dict()}")
        
        # Process like_user_ids column
        if 'like_user_ids' in df.columns:
            df['like_user_ids_list'] = df['like_user_ids'].apply(self._parse_comma_separated_numeric)
            
            # Calculate like counts
            df['like_count'] = df['like_user_ids_list'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        else:
            # If like_user_ids doesn't exist, create empty lists and zero counts
            df['like_user_ids_list'] = [[] for _ in range(len(df))]
            df['like_count'] = 0
        
        # Process is_anonymous column
        if 'is_anonymous' in df.columns:
            # Convert string representations to boolean
            df['is_anonymous'] = df['is_anonymous'].apply(self._parse_boolean)
        else:
            df['is_anonymous'] = False
        
        # Handle missing values
        df['title'] = df['title'].fillna('Untitled Post')
        df['content'] = df['content'].fillna('')
        
        # Show distribution of likes
        if df['like_count'].sum() > 0:
            print(f"Like distribution: min={df['like_count'].min()}, "
                  f"max={df['like_count'].max()}, "
                  f"mean={df['like_count'].mean():.2f}, "
                  f"median={df['like_count'].median():.2f}")
        
        return df
    
    def _create_interaction_matrix(self, users_df, posts_df):
        """Create user-post interaction matrix."""
        # Create mappings
        user_ids = users_df['user_id'].unique()
        post_ids = posts_df['post_id'].unique()
        
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        post_id_to_idx = {post_id: idx for idx, post_id in enumerate(post_ids)}
        
        # Create interaction data
        rows = []
        cols = []
        data = []
        
        for _, post in posts_df.iterrows():
            post_idx = post_id_to_idx[post['post_id']]
            like_user_ids = post.get('like_user_ids_list', [])
            
            if isinstance(like_user_ids, list):
                for user_id in like_user_ids:
                    if user_id in user_id_to_idx:
                        user_idx = user_id_to_idx[user_id]
                        rows.append(user_idx)
                        cols.append(post_idx)
                        data.append(1)  # Binary interaction (liked = 1)
        
        # Create sparse matrix
        interaction_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(len(user_ids), len(post_ids))
        )
        
        print(f"Interaction matrix shape: {interaction_matrix.shape}")
        print(f"Number of interactions: {interaction_matrix.nnz}")
        sparsity = interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1])
        print(f"Matrix sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
        
        return interaction_matrix
    
    def _parse_comma_separated(self, value):
        """Parse comma-separated string into list."""
        if pd.isna(value) or value == '':
            return []
        
        if isinstance(value, str):
            # Remove quotes and split by comma
            value = value.strip('"\'')
            return [item.strip() for item in value.split(',') if item.strip()]
        
        return []
    
    def _parse_comma_separated_numeric(self, value):
        """Parse comma-separated numeric string into list of integers."""
        if pd.isna(value) or value == '':
            return []
        
        if isinstance(value, str):
            try:
                # Remove quotes and split by comma
                value = value.strip('"\'')
                items = [item.strip() for item in value.split(',') if item.strip()]
                return [int(item) for item in items if item.isdigit()]
            except:
                return []
        
        return []
    
    def _parse_boolean(self, value):
        """Parse various boolean representations."""
        if pd.isna(value):
            return False
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            value = value.lower().strip()
            return value in ['true', '1', 'yes', 'y', 't']
        
        if isinstance(value, (int, float)):
            return bool(value)
        
        return False
    
    def get_data_summary(self, users_df, posts_df):
        """Get comprehensive data summary for exploration."""
        summary = {
            'users': {
                'total_count': len(users_df),
                'missing_names': users_df['name'].isnull().sum(),
                'missing_interests': users_df['interested_in'].isnull().sum() if 'interested_in' in users_df.columns else 0,
                'unique_interests': []
            },
            'posts': {
                'total_count': len(posts_df),
                'anonymous_count': posts_df['is_anonymous'].sum() if 'is_anonymous' in posts_df.columns else 0,
                'total_likes': posts_df['like_count'].sum() if 'like_count' in posts_df.columns else 0,
                'avg_likes': posts_df['like_count'].mean() if 'like_count' in posts_df.columns else 0,
                'max_likes': posts_df['like_count'].max() if 'like_count' in posts_df.columns else 0
            }
        }
        
        # Get unique interests
        if 'interests_list' in users_df.columns:
            all_interests = []
            for interests in users_df['interests_list'].dropna():
                if isinstance(interests, list):
                    all_interests.extend([interest.strip().lower() for interest in interests])
            
            summary['users']['unique_interests'] = list(set(all_interests))
        
        return summary
