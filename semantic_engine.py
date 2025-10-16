import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class SemanticRecommendationEngine:
    """
    Enhanced recommendation system using LSA (Latent Semantic Analysis) for semantic similarity.
    This provides deeper content understanding without requiring sentence-transformers.
    
    LSA captures semantic relationships by decomposing the TF-IDF matrix into latent concepts,
    allowing it to understand that "investing" and "stocks" are semantically related even if
    they don't appear in the same documents.
    """
    
    def __init__(self, n_components=100, collab_weight=0.3, semantic_weight=0.5, popularity_weight=0.2):
        """
        Initialize the semantic recommendation engine.
        
        Parameters:
        - n_components: Number of latent semantic dimensions (LSA components)
        - collab_weight: Weight for collaborative filtering score
        - semantic_weight: Weight for semantic similarity score
        - popularity_weight: Weight for popularity score
        """
        self.n_components = n_components
        self.collab_weight = collab_weight
        self.semantic_weight = semantic_weight
        self.popularity_weight = popularity_weight
        
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.post_semantic_matrix = None
        self.users_df = None
        self.posts_df = None
        self.interaction_matrix = None
        self.popularity_scaler = MinMaxScaler()
        self.post_popularity_scores = None
        
    def fit(self, users_df, posts_df, interaction_matrix):
        """
        Fit the semantic recommendation engine.
        """
        self.users_df = users_df.copy()
        self.posts_df = posts_df.copy()
        self.interaction_matrix = interaction_matrix
        
        print("Training semantic similarity model with LSA...")
        self._fit_semantic_model()
        
        print("Calculating popularity scores...")
        self._calculate_popularity_scores()
        
        print("Semantic recommendation engine training complete!")
        
    def _fit_semantic_model(self):
        """Fit LSA-based semantic similarity model."""
        try:
            self.posts_df['combined_text'] = (
                self.posts_df['title'].fillna('') + ' ' + 
                self.posts_df['content'].fillna('')
            ).str.lower()
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.8
            )
            
            post_texts = self.posts_df['combined_text'].fillna('')
            post_tfidf_matrix = self.tfidf_vectorizer.fit_transform(post_texts)
            
            n_components = min(self.n_components, post_tfidf_matrix.shape[0] - 1, post_tfidf_matrix.shape[1] - 1)
            
            if n_components < 2:
                print(f"Warning: Too few posts for LSA (need at least 3, got {post_tfidf_matrix.shape[0]}). Falling back to simple TF-IDF.")
                self.svd_model = None
                self.post_semantic_matrix = post_tfidf_matrix.toarray()
            else:
                self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
                self.post_semantic_matrix = self.svd_model.fit_transform(post_tfidf_matrix)
                
                explained_variance = self.svd_model.explained_variance_ratio_.sum()
                print(f"LSA explained variance: {explained_variance:.2%}")
            
        except Exception as e:
            print(f"Warning: Semantic model fitting failed: {e}")
            self.tfidf_vectorizer = None
            self.svd_model = None
            self.post_semantic_matrix = None
    
    def _calculate_popularity_scores(self):
        """Calculate and normalize popularity scores."""
        like_counts = self.posts_df['like_count'].values.reshape(-1, 1)
        self.post_popularity_scores = self.popularity_scaler.fit_transform(like_counts).flatten()
    
    def _get_collaborative_scores_simple(self, user_id, candidate_posts):
        """Simple collaborative filtering based on user-user similarity."""
        user_idx = np.where(self.users_df['user_id'] == user_id)[0]
        
        if len(user_idx) == 0:
            return np.zeros(len(candidate_posts))
        
        user_interactions = self.interaction_matrix[user_idx[0], :].toarray().flatten()
        
        scores = []
        for post_id in candidate_posts:
            post_idx = self.posts_df[self.posts_df['post_id'] == post_id].index
            if len(post_idx) > 0 and post_idx[0] < len(user_interactions):
                post_idx = post_idx[0]
                
                post_likers = self.interaction_matrix[:, post_idx].toarray().flatten()
                
                user_similarity = np.dot(user_interactions, post_likers) / (
                    np.linalg.norm(user_interactions) * np.linalg.norm(post_likers) + 1e-10
                )
                scores.append(user_similarity)
            else:
                scores.append(0.0)
        
        return np.array(scores)
    
    def _get_semantic_scores(self, user_id, candidate_posts):
        """Get semantic similarity scores using LSA."""
        if self.svd_model is None or self.post_semantic_matrix is None:
            return np.zeros(len(candidate_posts))
        
        try:
            user_info = self.users_df[self.users_df['user_id'] == user_id]
            
            if user_info.empty:
                return np.zeros(len(candidate_posts))
            
            user_interests = user_info.iloc[0].get('interests_list', [])
            
            user_text = ''
            if isinstance(user_interests, list):
                user_text = ' '.join(user_interests)
            
            user_idx = np.where(self.users_df['user_id'] == user_id)[0]
            if len(user_idx) > 0:
                user_interactions = self.interaction_matrix[user_idx[0], :].toarray().flatten()
                liked_post_indices = np.where(user_interactions > 0)[0]
                
                user_liked_posts = []
                for post_idx in liked_post_indices:
                    if post_idx < len(self.posts_df):
                        post_text = self.posts_df.iloc[post_idx]['combined_text']
                        user_liked_posts.append(post_text)
                
                if user_liked_posts:
                    user_text += ' ' + ' '.join(user_liked_posts)
            
            if not user_text.strip():
                return np.zeros(len(candidate_posts))
            
            user_tfidf = self.tfidf_vectorizer.transform([user_text.lower()])
            user_semantic = self.svd_model.transform(user_tfidf)
            
            candidate_indices = []
            for post_id in candidate_posts:
                post_idx = self.posts_df[self.posts_df['post_id'] == post_id].index
                if len(post_idx) > 0:
                    candidate_indices.append(post_idx[0])
                else:
                    candidate_indices.append(-1)
            
            scores = []
            for idx in candidate_indices:
                if idx >= 0 and idx < self.post_semantic_matrix.shape[0]:
                    similarity = cosine_similarity(user_semantic, self.post_semantic_matrix[idx:idx+1])
                    scores.append(similarity[0][0])
                else:
                    scores.append(0.0)
            
            return np.array(scores)
            
        except Exception as e:
            print(f"Warning: Semantic scoring failed: {e}")
            return np.zeros(len(candidate_posts))
    
    def _get_popularity_scores(self, candidate_posts):
        """Get popularity scores for candidate posts."""
        scores = []
        for post_id in candidate_posts:
            post_idx = self.posts_df[self.posts_df['post_id'] == post_id].index
            if len(post_idx) > 0 and post_idx[0] < len(self.post_popularity_scores):
                scores.append(self.post_popularity_scores[post_idx[0]])
            else:
                scores.append(0.0)
        return np.array(scores)
    
    def get_recommendations(self, user_id, n_recommendations=10):
        """
        Get recommendations for a specific user.
        """
        try:
            user_idx = np.where(self.users_df['user_id'] == user_id)[0]
            
            if len(user_idx) == 0:
                popular_posts = self.posts_df.nlargest(n_recommendations, 'like_count')
                return popular_posts['post_id'].tolist()
            
            user_interactions = self.interaction_matrix[user_idx[0], :].toarray().flatten()
            unrated_post_indices = np.where(user_interactions == 0)[0]
            
            if len(unrated_post_indices) == 0:
                return []
            
            candidate_posts = self.posts_df.iloc[unrated_post_indices]['post_id'].tolist()
            
            collab_scores = self._get_collaborative_scores_simple(user_id, candidate_posts)
            semantic_scores = self._get_semantic_scores(user_id, candidate_posts)
            popularity_scores = self._get_popularity_scores(candidate_posts)
            
            def normalize_scores(scores):
                if len(scores) == 0:
                    return scores
                min_score, max_score = scores.min(), scores.max()
                if max_score > min_score:
                    return (scores - min_score) / (max_score - min_score)
                else:
                    return np.ones_like(scores) * 0.5
            
            collab_scores = normalize_scores(collab_scores)
            semantic_scores = normalize_scores(semantic_scores)
            popularity_scores = normalize_scores(popularity_scores)
            
            final_scores = (
                self.collab_weight * collab_scores +
                self.semantic_weight * semantic_scores +
                self.popularity_weight * popularity_scores
            )
            
            np.random.seed(42)
            final_scores += 0.01 * np.random.random(len(final_scores))
            
            top_indices = np.argsort(final_scores)[-n_recommendations:][::-1]
            recommended_posts = [candidate_posts[i] for i in top_indices]
            
            return recommended_posts
            
        except Exception as e:
            print(f"Error getting recommendations for user {user_id}: {e}")
            popular_posts = self.posts_df.nlargest(n_recommendations, 'like_count')
            return popular_posts['post_id'].tolist()
    
    def get_recommendation_explanation(self, user_id, post_id):
        """Get explanation for why a post was recommended."""
        try:
            collab_score = self._get_collaborative_scores_simple(user_id, [post_id])[0]
            semantic_score = self._get_semantic_scores(user_id, [post_id])[0]
            popularity_score = self._get_popularity_scores([post_id])[0]
            
            post_info = self.posts_df[self.posts_df['post_id'] == post_id]
            post_title = post_info.iloc[0]['title'] if not post_info.empty else "Unknown"
            post_likes = post_info.iloc[0]['like_count'] if not post_info.empty else 0
            
            user_info = self.users_df[self.users_df['user_id'] == user_id]
            user_interests = user_info.iloc[0].get('interests_list', []) if not user_info.empty else []
            
            explanation = {
                'post_id': post_id,
                'post_title': post_title,
                'post_likes': post_likes,
                'user_interests': user_interests,
                'collaborative_score': round(collab_score, 3),
                'semantic_score': round(semantic_score, 3),
                'popularity_score': round(popularity_score, 3),
                'final_score': round(
                    self.collab_weight * collab_score +
                    self.semantic_weight * semantic_score +
                    self.popularity_weight * popularity_score, 3
                ),
                'model_type': 'LSA-Enhanced Semantic'
            }
            
            return explanation
            
        except Exception as e:
            return {'error': f"Could not generate explanation: {e}"}
