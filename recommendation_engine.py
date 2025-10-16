import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Handle surprise import gracefully
try:
    from surprise import Dataset, Reader, KNNWithMeans
    from surprise.model_selection import train_test_split
    SURPRISE_AVAILABLE = True
    print("✅ Surprise library available - collaborative filtering enabled")
except ImportError:
    SURPRISE_AVAILABLE = False
    print("⚠️ Surprise library not available - using content-based recommendations only")

class HybridRecommendationEngine:
    """
    Hybrid recommendation system combining:
    - Collaborative filtering (user-user similarity) - if surprise available
    - Content-based filtering (TF-IDF on post content + user interests)
    - Popularity boost (like counts)
    """
    
    def __init__(self, collab_weight=0.4, content_weight=0.4, popularity_weight=0.2):
        """
        Initialize the hybrid recommendation engine.
        
        Parameters:
        - collab_weight: Weight for collaborative filtering score
        - content_weight: Weight for content-based filtering score
        - popularity_weight: Weight for popularity score
        """
        # Adjust weights if surprise is not available
        if not SURPRISE_AVAILABLE:
            total = content_weight + popularity_weight
            content_weight = content_weight / total if total > 0 else 0.7
            popularity_weight = popularity_weight / total if total > 0 else 0.3
            collab_weight = 0
            print("⚠️ Adjusted weights (collaborative filtering disabled)")
        
        self.collab_weight = collab_weight
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight
        
        # Models and data
        self.collab_model = None
        self.tfidf_vectorizer = None
        self.post_tfidf_matrix = None
        self.users_df = None
        self.posts_df = None
        self.interaction_matrix = None
        self.popularity_scaler = MinMaxScaler()
        
        # Processed data
        self.user_post_ratings = None
        self.post_popularity_scores = None
        
    def fit(self, users_df, posts_df, interaction_matrix):
        """
        Fit the hybrid recommendation engine.
        
        Parameters:
        - users_df: DataFrame with user information
        - posts_df: DataFrame with post information
        - interaction_matrix: Sparse matrix of user-post interactions
        """
        self.users_df = users_df.copy()
        self.posts_df = posts_df.copy()
        self.interaction_matrix = interaction_matrix
        
        if SURPRISE_AVAILABLE:
            print("Training collaborative filtering model...")
            self._fit_collaborative_filtering()
        else:
            print("Collaborative filtering skipped (surprise not available)")
            self._fit_alternative_collaborative()
        
        print("Training content-based filtering model...")
        self._fit_content_based_filtering()
        
        print("Calculating popularity scores...")
        self._calculate_popularity_scores()
        
        print("Hybrid recommendation engine training complete!")
    
    def _fit_alternative_collaborative(self):
        """Alternative collaborative filtering using sklearn when surprise is not available."""
        try:
            # Use k-NN on user interaction matrix
            self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
            user_interactions_dense = self.interaction_matrix.toarray()
            self.knn_model.fit(user_interactions_dense)
            print("✅ Alternative collaborative filtering trained")
        except Exception as e:
            print(f"⚠️ Alternative collaborative filtering failed: {e}")
            self.knn_model = None
    
    def _fit_collaborative_filtering(self):
        """Fit collaborative filtering model using Surprise library."""
        if not SURPRISE_AVAILABLE:
            return
            
        try:
            # Create user-post rating data for Surprise
            ratings_data = []
            
            # Convert interaction matrix to ratings format
            users = self.users_df['user_id'].values
            posts = self.posts_df['post_id'].values
            
            # Get non-zero interactions
            user_indices, post_indices = self.interaction_matrix.nonzero()
            
            for user_idx, post_idx in zip(user_indices, post_indices):
                user_id = users[user_idx]
                post_id = posts[post_idx]
                # Use 1 as rating for liked posts (implicit feedback)
                ratings_data.append([user_id, post_id, 1])
            
            if len(ratings_data) == 0:
                print("⚠️ No interactions found for collaborative filtering")
                return
            
            # Create Surprise dataset
            reader = Reader(rating_scale=(0, 1))
            data = Dataset.load_from_df(pd.DataFrame(ratings_data, columns=['user_id', 'post_id', 'rating']), reader)
            
            # Train collaborative filtering model
            trainset = data.build_full_trainset()
            self.collab_model = KNNWithMeans(k=20, sim_options={'name': 'cosine', 'user_based': True})
            self.collab_model.fit(trainset)
            
            # Store the trainset for predictions
            self.trainset = trainset
            print("✅ Collaborative filtering trained with Surprise")
            
        except Exception as e:
            print(f"⚠️ Collaborative filtering failed: {e}")
            self.collab_model = None
    
    def _fit_content_based_filtering(self):
        """Fit content-based filtering model using TF-IDF."""
        try:
            # Combine post title and content
            self.posts_df['combined_text'] = (
                self.posts_df['title'].fillna('') + ' ' + 
                self.posts_df['content'].fillna('')
            ).str.lower()
            
            # Create TF-IDF matrix for posts
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            post_texts = self.posts_df['combined_text'].fillna('')
            self.post_tfidf_matrix = self.tfidf_vectorizer.fit_transform(post_texts)
            print("✅ Content-based filtering trained")
            
        except Exception as e:
            print(f"⚠️ Content-based filtering failed: {e}")
            self.tfidf_vectorizer = None
            self.post_tfidf_matrix = None
    
    def _calculate_popularity_scores(self):
        """Calculate and normalize popularity scores based on like counts."""
        try:
            like_counts = self.posts_df['like_count'].values.reshape(-1, 1)
            self.post_popularity_scores = self.popularity_scaler.fit_transform(like_counts).flatten()
            print("✅ Popularity scores calculated")
        except Exception as e:
            print(f"⚠️ Popularity calculation failed: {e}")
            self.post_popularity_scores = np.zeros(len(self.posts_df))
    
    def _get_collaborative_scores(self, user_id, candidate_posts):
        """Get collaborative filtering scores for candidate posts."""
        if not SURPRISE_AVAILABLE or self.collab_model is None or self.trainset is None:
            # Use alternative collaborative filtering or return zeros
            if hasattr(self, 'knn_model') and self.knn_model is not None:
                return self._get_alternative_collab_scores(user_id, candidate_posts)
            return np.zeros(len(candidate_posts))
        
        try:
            scores = []
            for post_id in candidate_posts:
                try:
                    prediction = self.collab_model.predict(user_id, post_id)
                    scores.append(prediction.est)
                except:
                    scores.append(0.0)  # Default score for unknown items
            return np.array(scores)
        except:
            return np.zeros(len(candidate_posts))
    
    def _get_alternative_collab_scores(self, user_id, candidate_posts):
        """Alternative collaborative scoring using k-NN."""
        try:
            user_idx = np.where(self.users_df['user_id'] == user_id)[0]
            if len(user_idx) == 0:
                return np.zeros(len(candidate_posts))
            
            user_vector = self.interaction_matrix[user_idx[0], :].toarray()
            
            # Find similar users
            distances, indices = self.knn_model.kneighbors(user_vector)
            
            # Get scores based on similar users' preferences
            scores = []
            for post_id in candidate_posts:
                post_idx = self.posts_df[self.posts_df['post_id'] == post_id].index
                if len(post_idx) > 0:
                    post_vector = self.interaction_matrix[:, post_idx[0]].toarray().flatten()
                    # Score based on similar users' interactions
                    similar_users_interactions = post_vector[indices[0]]
                    score = np.mean(similar_users_interactions)
                    scores.append(score)
                else:
                    scores.append(0.0)
            
            return np.array(scores)
        except:
            return np.zeros(len(candidate_posts))
    
    def _get_content_scores(self, user_id, candidate_posts):
        """Get content-based filtering scores for candidate posts."""
        if self.tfidf_vectorizer is None or self.post_tfidf_matrix is None:
            return np.zeros(len(candidate_posts))
        
        try:
            # Get user's interests and liked posts
            user_info = self.users_df[self.users_df['user_id'] == user_id]
            
            if user_info.empty:
                return np.zeros(len(candidate_posts))
            
            user_interests = user_info.iloc[0].get('interests_list', [])
            
            # Create user profile from interests
            user_text = ''
            if isinstance(user_interests, list):
                user_text = ' '.join(user_interests)
            
            # Also include content from posts the user has liked
            user_liked_posts = []
            user_idx = np.where(self.users_df['user_id'] == user_id)[0]
            if len(user_idx) > 0:
                user_interactions = self.interaction_matrix[user_idx[0], :].toarray().flatten()
                liked_post_indices = np.where(user_interactions > 0)[0]
                
                for post_idx in liked_post_indices:
                    if post_idx < len(self.posts_df):
                        post_text = self.posts_df.iloc[post_idx]['combined_text']
                        user_liked_posts.append(post_text)
            
            # Combine user interests and liked post content
            if user_liked_posts:
                user_text += ' ' + ' '.join(user_liked_posts)
            
            if not user_text.strip():
                return np.zeros(len(candidate_posts))
            
            # Transform user profile
            user_tfidf = self.tfidf_vectorizer.transform([user_text.lower()])
            
            # Get candidate post indices
            candidate_indices = []
            for post_id in candidate_posts:
                post_idx = self.posts_df[self.posts_df['post_id'] == post_id].index
                if len(post_idx) > 0:
                    candidate_indices.append(post_idx[0])
                else:
                    candidate_indices.append(-1)
            
            # Calculate similarity scores
            scores = []
            for idx in candidate_indices:
                if idx >= 0 and idx < self.post_tfidf_matrix.shape[0]:
                    similarity = cosine_similarity(user_tfidf, self.post_tfidf_matrix[idx:idx+1])
                    scores.append(similarity[0][0])
                else:
                    scores.append(0.0)
            
            return np.array(scores)
            
        except Exception as e:
            print(f"⚠️ Content scoring failed: {e}")
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
        
        Parameters:
        - user_id: ID of the user to get recommendations for
        - n_recommendations: Number of recommendations to return
        
        Returns:
        - List of recommended post IDs
        """
        try:
            # Get posts the user hasn't liked
            user_idx = np.where(self.users_df['user_id'] == user_id)[0]
            
            if len(user_idx) == 0:
                # Cold start: recommend most popular posts
                popular_posts = self.posts_df.nlargest(n_recommendations, 'like_count')
                return popular_posts['post_id'].tolist()
            
            user_interactions = self.interaction_matrix[user_idx[0], :].toarray().flatten()
            unrated_post_indices = np.where(user_interactions == 0)[0]
            
            if len(unrated_post_indices) == 0:
                return []  # User has liked all posts
            
            candidate_posts = self.posts_df.iloc[unrated_post_indices]['post_id'].tolist()
            
            # Get scores from each component
            collab_scores = self._get_collaborative_scores(user_id, candidate_posts)
            content_scores = self._get_content_scores(user_id, candidate_posts)
            popularity_scores = self._get_popularity_scores(candidate_posts)
            
            # Normalize scores to [0, 1] range
            def normalize_scores(scores):
                if len(scores) == 0:
                    return scores
                min_score, max_score = scores.min(), scores.max()
                if max_score > min_score:
                    return (scores - min_score) / (max_score - min_score)
                else:
                    return np.ones_like(scores) * 0.5
            
            collab_scores = normalize_scores(collab_scores)
            content_scores = normalize_scores(content_scores)
            popularity_scores = normalize_scores(popularity_scores)
            
            # Combine scores with weights
            final_scores = (
                self.collab_weight * collab_scores +
                self.content_weight * content_scores +
                self.popularity_weight * popularity_scores
            )
            
            # Add small random component for diversity
            np.random.seed(42)  # For reproducibility
            final_scores += 0.01 * np.random.random(len(final_scores))
            
            # Get top N recommendations
            top_indices = np.argsort(final_scores)[-n_recommendations:][::-1]
            recommended_posts = [candidate_posts[i] for i in top_indices]
            
            return recommended_posts
            
        except Exception as e:
            print(f"⚠️ Error getting recommendations for user {user_id}: {e}")
            # Fallback to popular posts
            popular_posts = self.posts_df.nlargest(n_recommendations, 'like_count')
            return popular_posts['post_id'].tolist()
    
    def get_recommendation_explanation(self, user_id, post_id):
        """
        Get explanation for why a post was recommended to a user.
        
        Parameters:
        - user_id: ID of the user
        - post_id: ID of the recommended post
        
        Returns:
        - Dictionary with explanation details
        """
        try:
            # Get individual scores
            collab_score = self._get_collaborative_scores(user_id, [post_id])[0]
            content_score = self._get_content_scores(user_id, [post_id])[0]
            popularity_score = self._get_popularity_scores([post_id])[0]
            
            # Get post details
            post_info = self.posts_df[self.posts_df['post_id'] == post_id]
            post_title = post_info.iloc[0]['title'] if not post_info.empty else "Unknown"
            post_likes = post_info.iloc[0]['like_count'] if not post_info.empty else 0
            
            # Get user interests
            user_info = self.users_df[self.users_df['user_id'] == user_id]
            user_interests = user_info.iloc[0].get('interests_list', []) if not user_info.empty else []
            
            explanation = {
                'post_id': post_id,
                'post_title': post_title,
                'post_likes': post_likes,
                'user_interests': user_interests,
                'collaborative_score': round(collab_score, 3),
                'content_score': round(content_score, 3),
                'popularity_score': round(popularity_score, 3),
                'final_score': round(
                    self.collab_weight * collab_score +
                    self.content_weight * content_score +
                    self.popularity_weight * popularity_score, 3
                )
            }
            
            return explanation
            
        except Exception as e:
            return {'error': f"Could not generate explanation: {e}"}