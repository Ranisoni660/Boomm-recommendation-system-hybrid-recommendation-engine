import base64
import pandas as pd
from io import BytesIO

def create_download_link(df, filename="download.csv", text="Download CSV"):
    """
    Create a download link for a pandas DataFrame.
    
    Parameters:
    - df: pandas DataFrame
    - filename: name for the downloaded file
    - text: link text
    
    Returns:
    - HTML download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def format_recommendations_for_export(recommendations_df):
    """
    Format recommendations DataFrame for CSV export.
    
    Parameters:
    - recommendations_df: DataFrame with user_id and recommended_post_ids columns
    
    Returns:
    - Formatted DataFrame ready for export
    """
    export_df = recommendations_df.copy()
    
    # Ensure recommended_post_ids are properly formatted as comma-separated strings
    export_df['recommended_post_ids'] = export_df['recommended_post_ids'].apply(
        lambda x: x if isinstance(x, str) and x else ''
    )
    
    return export_df[['user_id', 'recommended_post_ids']]

def calculate_recommendation_metrics(recommendations_df, posts_df):
    """
    Calculate various metrics for recommendation quality analysis.
    
    Parameters:
    - recommendations_df: DataFrame with recommendations
    - posts_df: DataFrame with post information
    
    Returns:
    - Dictionary with various metrics
    """
    metrics = {}
    
    # Basic counts
    metrics['total_users'] = len(recommendations_df)
    metrics['total_posts'] = len(posts_df)
    
    # Recommendation coverage
    recommended_posts = set()
    total_recommendations = 0
    
    for recs in recommendations_df['recommended_post_ids']:
        if recs and isinstance(recs, str):
            post_list = [p.strip() for p in recs.split(',') if p.strip()]
            recommended_posts.update(post_list)
            total_recommendations += len(post_list)
    
    metrics['unique_posts_recommended'] = len(recommended_posts)
    metrics['catalog_coverage'] = len(recommended_posts) / len(posts_df) if len(posts_df) > 0 else 0
    metrics['total_recommendations'] = total_recommendations
    metrics['avg_recommendations_per_user'] = total_recommendations / len(recommendations_df) if len(recommendations_df) > 0 else 0
    
    # Popularity bias
    if 'like_count' in posts_df.columns:
        posts_df_str = posts_df.copy()
        posts_df_str['post_id'] = posts_df_str['post_id'].astype(str)
        
        recommended_post_likes = []
        for post_id in recommended_posts:
            post_info = posts_df_str[posts_df_str['post_id'] == post_id]
            if not post_info.empty:
                recommended_post_likes.append(post_info.iloc[0]['like_count'])
        
        if recommended_post_likes:
            metrics['avg_recommended_post_likes'] = sum(recommended_post_likes) / len(recommended_post_likes)
            metrics['avg_all_post_likes'] = posts_df['like_count'].mean()
            metrics['popularity_bias'] = metrics['avg_recommended_post_likes'] / metrics['avg_all_post_likes'] if metrics['avg_all_post_likes'] > 0 else 0
    
    return metrics

def get_sample_recommendations_with_details(recommendations_df, users_df, posts_df, n_samples=5):
    """
    Get sample recommendations with detailed user and post information.
    
    Parameters:
    - recommendations_df: DataFrame with recommendations
    - users_df: DataFrame with user information
    - posts_df: DataFrame with post information
    - n_samples: Number of sample users to show
    
    Returns:
    - List of dictionaries with detailed recommendation information
    """
    samples = []
    
    for idx, row in recommendations_df.head(n_samples).iterrows():
        user_id = row['user_id']
        recommended_post_ids = row['recommended_post_ids'].split(',') if row['recommended_post_ids'] else []
        
        # Get user information
        user_info = users_df[users_df['user_id'] == user_id]
        if user_info.empty:
            continue
            
        user_data = user_info.iloc[0]
        
        # Get recommended post details
        recommended_posts = []
        posts_df_str = posts_df.copy()
        posts_df_str['post_id'] = posts_df_str['post_id'].astype(str)
        
        for post_id in recommended_post_ids[:3]:  # Show top 3 recommendations
            post_id = post_id.strip()
            post_info = posts_df_str[posts_df_str['post_id'] == post_id]
            
            if not post_info.empty:
                post_data = post_info.iloc[0]
                recommended_posts.append({
                    'post_id': post_data['post_id'],
                    'title': post_data['title'],
                    'like_count': post_data.get('like_count', 0),
                    'is_anonymous': post_data.get('is_anonymous', False)
                })
        
        sample = {
            'user_id': user_id,
            'user_name': user_data.get('name', 'Unknown'),
            'user_interests': user_data.get('interests_list', []),
            'total_recommendations': len(recommended_post_ids),
            'sample_posts': recommended_posts
        }
        
        samples.append(sample)
    
    return samples

def validate_csv_format(df, required_columns):
    """
    Validate that a DataFrame has the required columns and format.
    
    Parameters:
    - df: pandas DataFrame to validate
    - required_columns: list of required column names
    
    Returns:
    - tuple (is_valid, error_message)
    """
    # Check if all required columns exist
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for empty DataFrame
    if len(df) == 0:
        return False, "DataFrame is empty"
    
    # Check for required columns having non-null values
    for col in required_columns:
        if df[col].isnull().all():
            return False, f"Column '{col}' contains only null values"
    
    return True, "Valid format"

def clean_text_for_analysis(text):
    """
    Clean text for content analysis.
    
    Parameters:
    - text: input text string
    
    Returns:
    - cleaned text string
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters but keep spaces
    import re
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def calculate_diversity_score(recommendations_df, posts_df):
    """
    Calculate recommendation diversity based on post categories/topics.
    
    Parameters:
    - recommendations_df: DataFrame with recommendations
    - posts_df: DataFrame with post information
    
    Returns:
    - diversity score (higher = more diverse)
    """
    try:
        # Simple diversity measure based on like count distribution
        all_recommended_posts = []
        
        for recs in recommendations_df['recommended_post_ids']:
            if recs and isinstance(recs, str):
                post_list = [p.strip() for p in recs.split(',') if p.strip()]
                all_recommended_posts.extend(post_list)
        
        if not all_recommended_posts:
            return 0.0
        
        # Get like counts for recommended posts
        posts_df_str = posts_df.copy()
        posts_df_str['post_id'] = posts_df_str['post_id'].astype(str)
        
        like_counts = []
        for post_id in all_recommended_posts:
            post_info = posts_df_str[posts_df_str['post_id'] == post_id]
            if not post_info.empty:
                like_counts.append(post_info.iloc[0].get('like_count', 0))
        
        if not like_counts:
            return 0.0
        
        # Calculate coefficient of variation as diversity measure
        import numpy as np
        mean_likes = np.mean(like_counts)
        std_likes = np.std(like_counts)
        
        if mean_likes > 0:
            diversity_score = std_likes / mean_likes
        else:
            diversity_score = 0.0
        
        return min(diversity_score, 1.0)  # Cap at 1.0
        
    except Exception as e:
        print(f"Error calculating diversity score: {e}")
        return 0.0
