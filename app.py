import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from recommendation_engine import HybridRecommendationEngine
from semantic_engine import SemanticRecommendationEngine
from data_processor import DataProcessor
from utils import create_download_link

# Page configuration
st.set_page_config(
    page_title="Boomm Post Recommendation System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🚀 Boomm Post Recommendation System")
st.markdown("*A hybrid recommendation engine combining collaborative filtering, content-based analysis, and popularity metrics*")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'recommendations_generated' not in st.session_state:
    st.session_state.recommendations_generated = False

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", [
    "📊 Data Exploration", 
    "🔧 Data Processing", 
    "🤖 Recommendation Engine", 
    "📈 Analysis & Insights",
    "💡 Recommendation Explanations",
    "📋 Export Results"
])

# File upload section
st.sidebar.header("📁 Data Upload")
users_file = st.sidebar.file_uploader("Upload users.csv", type="csv", key="users")
posts_file = st.sidebar.file_uploader("Upload posts.csv", type="csv", key="posts")

# Add sample data loader button
st.sidebar.markdown("---")
if st.sidebar.button("📦 Load Sample Data", help="Load pre-extracted Boomm dataset"):
    if not st.session_state.data_loaded:
        with st.spinner("Loading sample data..."):
            try:
                from data_loader import load_sample_data
                users_df, posts_df, processed_data, processor = load_sample_data()
                
                st.session_state.users_df = processed_data['users_df']
                st.session_state.posts_df = processed_data['posts_df']
                st.session_state.interaction_matrix = processed_data['interaction_matrix']
                st.session_state.processor = processor
                st.session_state.data_loaded = True
                
                st.sidebar.success("✅ Sample data loaded successfully!")
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"❌ Error loading sample data: {str(e)}")

# Load data
if users_file is not None and posts_file is not None and not st.session_state.data_loaded:
    with st.spinner("Loading and processing data..."):
        try:
            # Initialize data processor
            processor = DataProcessor()
            
            # Load data
            users_df = pd.read_csv(users_file)
            posts_df = pd.read_csv(posts_file)
            
            # Process data
            processed_data = processor.process_data(users_df, posts_df)
            
            # Store in session state
            st.session_state.users_df = processed_data['users_df']
            st.session_state.posts_df = processed_data['posts_df']
            st.session_state.interaction_matrix = processed_data['interaction_matrix']
            st.session_state.processor = processor
            st.session_state.data_loaded = True
            
            st.sidebar.success("✅ Data loaded successfully!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")

# Main content based on selected page
if page == "📊 Data Exploration":
    st.header("📊 Data Exploration")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload users.csv and posts.csv files to begin exploration.")
        
        # Show expected data format
        st.subheader("Expected Data Format")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**users.csv format:**")
            st.code("""user_id,name,interested_in
1,John Doe,"finance,investing,stocks"
2,Jane Smith,"crypto,blockchain,trading"
...""")
            
        with col2:
            st.markdown("**posts.csv format:**")
            st.code("""post_id,title,content,is_anonymous,like_user_ids
1,Market Analysis,"Today's market...","False","1,3,5"
2,Crypto News,"Bitcoin update...","True","2,4,6"
...""")
    else:
        users_df = st.session_state.users_df
        posts_df = st.session_state.posts_df
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(users_df))
        with col2:
            st.metric("Total Posts", len(posts_df))
        with col3:
            total_likes = posts_df['like_count'].sum()
            st.metric("Total Likes", total_likes)
        with col4:
            avg_likes = posts_df['like_count'].mean()
            st.metric("Avg Likes/Post", f"{avg_likes:.1f}")
        
        # Data quality overview
        st.subheader("📋 Data Quality Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Users Dataset:**")
            users_info = pd.DataFrame({
                'Column': users_df.columns,
                'Non-Null Count': [users_df[col].count() for col in users_df.columns],
                'Null Count': [users_df[col].isnull().sum() for col in users_df.columns],
                'Data Type': [users_df[col].dtype for col in users_df.columns]
            })
            st.dataframe(users_info)
            
        with col2:
            st.markdown("**Posts Dataset:**")
            posts_info = pd.DataFrame({
                'Column': posts_df.columns,
                'Non-Null Count': [posts_df[col].count() for col in posts_df.columns],
                'Null Count': [posts_df[col].isnull().sum() for col in posts_df.columns],
                'Data Type': [posts_df[col].dtype for col in posts_df.columns]
            })
            st.dataframe(posts_info)
        
        # Visualizations
        st.subheader("📈 Data Distributions")
        
        # Like distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_likes = px.histogram(
                posts_df, 
                x='like_count', 
                nbins=30,
                title='Distribution of Post Likes',
                labels={'like_count': 'Number of Likes', 'count': 'Number of Posts'}
            )
            st.plotly_chart(fig_likes, use_container_width=True)
        
        with col2:
            # Top interests
            if 'interests_list' in users_df.columns:
                all_interests = []
                for interests in users_df['interests_list'].dropna():
                    if isinstance(interests, list):
                        all_interests.extend([interest.strip().lower() for interest in interests])
                
                if all_interests:
                    interest_counts = pd.Series(all_interests).value_counts().head(10)
                    
                    fig_interests = px.bar(
                        x=interest_counts.values,
                        y=interest_counts.index,
                        orientation='h',
                        title='Top 10 User Interests',
                        labels={'x': 'Number of Users', 'y': 'Interest'}
                    )
                    fig_interests.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_interests, use_container_width=True)
        
        # Anonymous posts analysis
        if 'is_anonymous' in posts_df.columns:
            st.subheader("👤 Anonymous vs Non-Anonymous Posts")
            
            anon_stats = posts_df.groupby('is_anonymous').agg({
                'post_id': 'count',
                'like_count': ['mean', 'sum']
            }).round(2)
            anon_stats.columns = ['Post Count', 'Avg Likes', 'Total Likes']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(anon_stats)
            
            with col2:
                fig_anon = px.box(
                    posts_df, 
                    x='is_anonymous', 
                    y='like_count',
                    title='Like Distribution by Anonymous Status'
                )
                st.plotly_chart(fig_anon, use_container_width=True)

elif page == "🔧 Data Processing":
    st.header("🔧 Data Processing")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload and load data first.")
    else:
        users_df = st.session_state.users_df
        posts_df = st.session_state.posts_df
        interaction_matrix = st.session_state.interaction_matrix
        
        st.subheader("✅ Processing Complete")
        st.success("Data has been successfully processed and is ready for recommendation generation.")
        
        # Show processing results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Processed Users", len(users_df))
        with col2:
            st.metric("Processed Posts", len(posts_df))
        with col3:
            sparsity = (interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1])) * 100
            st.metric("Matrix Sparsity", f"{sparsity:.2f}%")
        
        # Show sample processed data
        st.subheader("📋 Sample Processed Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sample Users:**")
            display_users = users_df.head()
            if 'interests_list' in display_users.columns:
                display_users = display_users.copy()
                display_users['interests_preview'] = display_users['interests_list'].apply(
                    lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '') if isinstance(x, list) else 'None'
                )
                st.dataframe(display_users[['user_id', 'name', 'interests_preview']])
            else:
                st.dataframe(display_users)
        
        with col2:
            st.markdown("**Sample Posts:**")
            display_posts = posts_df.head()
            if 'content' in display_posts.columns:
                display_posts = display_posts.copy()
                display_posts['content_preview'] = display_posts['content'].apply(
                    lambda x: (x[:50] + '...') if len(str(x)) > 50 else str(x)
                )
                st.dataframe(display_posts[['post_id', 'title', 'like_count', 'content_preview']])
            else:
                st.dataframe(display_posts)
        
        # Interaction matrix visualization
        st.subheader("🔗 User-Post Interaction Matrix")
        
        # Sample of interaction matrix for visualization
        if interaction_matrix.shape[0] > 0 and interaction_matrix.shape[1] > 0:
            sample_size = min(20, interaction_matrix.shape[0], interaction_matrix.shape[1])
            sample_matrix = interaction_matrix[:sample_size, :sample_size].toarray()
            
            fig_matrix = px.imshow(
                sample_matrix,
                title=f'Sample Interaction Matrix ({sample_size}x{sample_size})',
                labels={'x': 'Posts', 'y': 'Users', 'color': 'Interaction'},
                aspect='auto'
            )
            st.plotly_chart(fig_matrix, use_container_width=True)
        
        st.info("💡 The interaction matrix represents user-post relationships where 1 indicates a user liked a post.")

elif page == "🤖 Recommendation Engine":
    st.header("🤖 Recommendation Engine")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload and load data first.")
    else:
        # Model Selection
        st.subheader("🔬 Model Selection & A/B Testing")
        
        model_type = st.radio(
            "Choose Recommendation Algorithm",
            ["Hybrid (TF-IDF)", "Semantic (LSA-Enhanced)", "A/B Test Both"],
            help="Hybrid uses traditional TF-IDF matching. Semantic uses LSA for deeper understanding. A/B Test compares both."
        )
        
        # Engine configuration
        st.subheader("⚙️ Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        if model_type == "Semantic (LSA-Enhanced)":
            with col1:
                collab_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.3, 0.1)
            with col2:
                semantic_weight = st.slider("Semantic Similarity Weight", 0.0, 1.0, 0.5, 0.1)
            with col3:
                popularity_weight = st.slider("Popularity Weight", 0.0, 1.0, 0.2, 0.1)
            
            total_weight = collab_weight + semantic_weight + popularity_weight
            if total_weight > 0:
                collab_weight /= total_weight
                semantic_weight /= total_weight
                popularity_weight /= total_weight
        else:
            with col1:
                collab_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.4, 0.1)
            with col2:
                content_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.4, 0.1)
            with col3:
                popularity_weight = st.slider("Popularity Weight", 0.0, 1.0, 0.2, 0.1)
            
            total_weight = collab_weight + content_weight + popularity_weight
            if total_weight > 0:
                collab_weight /= total_weight
                content_weight /= total_weight
                popularity_weight /= total_weight
        
        col1, col2 = st.columns(2)
        with col1:
            n_recommendations = st.selectbox("Number of Recommendations per User", [5, 10, 15, 20], index=1)
        with col2:
            min_users_for_recommendations = st.selectbox("Minimum Users to Process", [10, 25, 50, 100], index=2)
        
        # Generate recommendations button
        if st.button("🚀 Generate Recommendations", type="primary"):
            with st.spinner("Generating recommendations... This may take a few minutes."):
                try:
                    user_ids = st.session_state.users_df['user_id'].head(min_users_for_recommendations).tolist()
                    
                    if model_type == "A/B Test Both":
                        st.info("🧪 Running A/B Test with both models...")
                        
                        st.markdown("### Model A: Hybrid (TF-IDF)")
                        engine_a = HybridRecommendationEngine(
                            collab_weight=collab_weight,
                            content_weight=content_weight,
                            popularity_weight=popularity_weight
                        )
                        engine_a.fit(
                            st.session_state.users_df,
                            st.session_state.posts_df,
                            st.session_state.interaction_matrix
                        )
                        
                        recommendations_a = []
                        progress_a = st.progress(0)
                        for i, user_id in enumerate(user_ids):
                            user_recs = engine_a.get_recommendations(user_id, n_recommendations)
                            recommendations_a.append({
                                'user_id': user_id,
                                'recommended_post_ids': ','.join(map(str, user_recs))
                            })
                            progress_a.progress((i + 1) / len(user_ids))
                        
                        st.markdown("### Model B: Semantic (LSA)")
                        engine_b = SemanticRecommendationEngine(
                            n_components=100,
                            collab_weight=0.3,
                            semantic_weight=0.5,
                            popularity_weight=0.2
                        )
                        engine_b.fit(
                            st.session_state.users_df,
                            st.session_state.posts_df,
                            st.session_state.interaction_matrix
                        )
                        
                        recommendations_b = []
                        progress_b = st.progress(0)
                        for i, user_id in enumerate(user_ids):
                            user_recs = engine_b.get_recommendations(user_id, n_recommendations)
                            recommendations_b.append({
                                'user_id': user_id,
                                'recommended_post_ids': ','.join(map(str, user_recs))
                            })
                            progress_b.progress((i + 1) / len(user_ids))
                        
                        st.session_state.recommendations_df_a = pd.DataFrame(recommendations_a)
                        st.session_state.recommendations_df_b = pd.DataFrame(recommendations_b)
                        st.session_state.engine_a = engine_a
                        st.session_state.engine_b = engine_b
                        st.session_state.recommendations_df = st.session_state.recommendations_df_a
                        st.session_state.engine = engine_a
                        st.session_state.ab_test_mode = True
                        st.session_state.recommendations_generated = True
                        
                        st.success("✅ A/B Test complete! View results in Analysis & Insights page.")
                        
                    elif model_type == "Semantic (LSA-Enhanced)":
                        engine = SemanticRecommendationEngine(
                            n_components=100,
                            collab_weight=collab_weight,
                            semantic_weight=semantic_weight,
                            popularity_weight=popularity_weight
                        )
                        
                        engine.fit(
                            st.session_state.users_df,
                            st.session_state.posts_df,
                            st.session_state.interaction_matrix
                        )
                        
                        recommendations = []
                        progress_bar = st.progress(0)
                        
                        for i, user_id in enumerate(user_ids):
                            user_recs = engine.get_recommendations(user_id, n_recommendations)
                            recommendations.append({
                                'user_id': user_id,
                                'recommended_post_ids': ','.join(map(str, user_recs))
                            })
                            progress_bar.progress((i + 1) / len(user_ids))
                        
                        recommendations_df = pd.DataFrame(recommendations)
                        
                        st.session_state.recommendations_df = recommendations_df
                        st.session_state.engine = engine
                        st.session_state.ab_test_mode = False
                        st.session_state.recommendations_generated = True
                        
                        st.success(f"✅ Generated semantic recommendations for {len(recommendations)} users!")
                        st.dataframe(recommendations_df.head(10))
                        
                    else:
                        engine = HybridRecommendationEngine(
                            collab_weight=collab_weight,
                            content_weight=content_weight,
                            popularity_weight=popularity_weight
                        )
                        
                        engine.fit(
                            st.session_state.users_df,
                            st.session_state.posts_df,
                            st.session_state.interaction_matrix
                        )
                        
                        recommendations = []
                        progress_bar = st.progress(0)
                        
                        for i, user_id in enumerate(user_ids):
                            user_recs = engine.get_recommendations(user_id, n_recommendations)
                            recommendations.append({
                                'user_id': user_id,
                                'recommended_post_ids': ','.join(map(str, user_recs))
                            })
                            progress_bar.progress((i + 1) / len(user_ids))
                        
                        recommendations_df = pd.DataFrame(recommendations)
                        
                        st.session_state.recommendations_df = recommendations_df
                        st.session_state.engine = engine
                        st.session_state.ab_test_mode = False
                        st.session_state.recommendations_generated = True
                        
                        st.success(f"✅ Generated recommendations for {len(recommendations)} users!")
                        st.dataframe(recommendations_df.head(10))
                    
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {str(e)}")
                    st.exception(e)
        
        # Show existing recommendations if available
        if st.session_state.recommendations_generated:
            st.subheader("✅ Current Recommendations")
            st.dataframe(st.session_state.recommendations_df.head(10))
            
            # Show recommendation statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Users Processed", len(st.session_state.recommendations_df))
            with col2:
                avg_recs = st.session_state.recommendations_df['recommended_post_ids'].apply(
                    lambda x: len(x.split(',')) if x else 0
                ).mean()
                st.metric("Avg Recommendations/User", f"{avg_recs:.1f}")
            with col3:
                unique_posts = set()
                for recs in st.session_state.recommendations_df['recommended_post_ids']:
                    if recs:
                        unique_posts.update(recs.split(','))
                st.metric("Unique Posts Recommended", len(unique_posts))

elif page == "📈 Analysis & Insights":
    st.header("📈 Analysis & Insights")
    
    if not st.session_state.recommendations_generated:
        st.warning("⚠️ Please generate recommendations first.")
    else:
        recommendations_df = st.session_state.recommendations_df
        users_df = st.session_state.users_df
        posts_df = st.session_state.posts_df
        engine = st.session_state.engine
        
        # A/B Test Comparison Section
        if 'ab_test_mode' in st.session_state and st.session_state.ab_test_mode:
            st.subheader("🧪 A/B Test Comparison")
            
            recs_a = st.session_state.recommendations_df_a
            recs_b = st.session_state.recommendations_df_b
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Model A: Hybrid (TF-IDF)")
                
                posts_a = set()
                for recs in recs_a['recommended_post_ids']:
                    if recs:
                        posts_a.update(recs.split(','))
                
                coverage_a = len(posts_a) / len(posts_df) * 100
                st.metric("Coverage", f"{coverage_a:.1f}%")
                st.metric("Unique Posts", len(posts_a))
                
            with col2:
                st.markdown("### Model B: Semantic (LSA)")
                
                posts_b = set()
                for recs in recs_b['recommended_post_ids']:
                    if recs:
                        posts_b.update(recs.split(','))
                
                coverage_b = len(posts_b) / len(posts_df) * 100
                st.metric("Coverage", f"{coverage_b:.1f}%")
                st.metric("Unique Posts", len(posts_b))
            
            # Overlap analysis
            overlap = len(posts_a & posts_b)
            overlap_pct = (overlap / len(posts_a | posts_b)) * 100 if len(posts_a | posts_b) > 0 else 0
            
            st.markdown(f"""
            **Recommendation Overlap:**  
            - Common posts recommended by both models: **{overlap}** ({overlap_pct:.1f}% of total unique recommendations)
            - Model A exclusive: **{len(posts_a - posts_b)}** posts
            - Model B exclusive: **{len(posts_b - posts_a)}** posts
            
            💡 Higher overlap indicates both models find similar high-quality posts. Lower overlap means models are discovering different valuable content.
            """)
            
            st.markdown("---")
        
        # Recommendation diversity analysis
        st.subheader("🎯 Recommendation Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Coverage analysis
            total_posts = len(posts_df)
            recommended_posts = set()
            
            for recs in recommendations_df['recommended_post_ids']:
                if recs:
                    recommended_posts.update(recs.split(','))
            
            coverage = len(recommended_posts) / total_posts * 100
            
            st.metric("Catalog Coverage", f"{coverage:.1f}%")
            st.caption(f"{len(recommended_posts)} out of {total_posts} posts recommended")
        
        with col2:
            # Diversity score
            rec_lengths = recommendations_df['recommended_post_ids'].apply(
                lambda x: len(x.split(',')) if x else 0
            )
            diversity_score = rec_lengths.std() / rec_lengths.mean() if rec_lengths.mean() > 0 else 0
            
            st.metric("Recommendation Diversity", f"{diversity_score:.3f}")
            st.caption("Lower values indicate more consistent recommendation counts")
        
        # Popular posts analysis
        st.subheader("📊 Popular Posts in Recommendations")
        
        post_recommendation_counts = {}
        for recs in recommendations_df['recommended_post_ids']:
            if recs:
                for post_id in recs.split(','):
                    post_id = post_id.strip()
                    post_recommendation_counts[post_id] = post_recommendation_counts.get(post_id, 0) + 1
        
        if post_recommendation_counts:
            top_recommended = pd.DataFrame([
                {'post_id': post_id, 'recommendation_count': count}
                for post_id, count in sorted(post_recommendation_counts.items(), 
                                           key=lambda x: x[1], reverse=True)[:10]
            ])
            
            # Merge with post details
            top_recommended['post_id'] = top_recommended['post_id'].astype(str)
            posts_df_str = posts_df.copy()
            posts_df_str['post_id'] = posts_df_str['post_id'].astype(str)
            
            top_recommended_detailed = top_recommended.merge(
                posts_df_str[['post_id', 'title', 'like_count']], 
                on='post_id', 
                how='left'
            )
            
            fig_popular = px.bar(
                top_recommended_detailed,
                x='recommendation_count',
                y='title',
                orientation='h',
                title='Top 10 Most Recommended Posts',
                labels={'recommendation_count': 'Times Recommended', 'title': 'Post Title'}
            )
            fig_popular.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_popular, use_container_width=True)
            
            st.dataframe(top_recommended_detailed)
        
        # Sample user recommendations with explanations
        st.subheader("👤 Sample User Recommendations")
        
        sample_users = recommendations_df.head(5)
        
        for idx, row in sample_users.iterrows():
            user_id = row['user_id']
            recommended_post_ids = row['recommended_post_ids'].split(',') if row['recommended_post_ids'] else []
            
            # Get user info
            user_info = users_df[users_df['user_id'] == user_id].iloc[0]
            user_interests = user_info.get('interests_list', []) if 'interests_list' in user_info else []
            
            with st.expander(f"User {user_id} - {user_info.get('name', 'Unknown')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**User Profile:**")
                    st.write(f"**Name:** {user_info.get('name', 'Unknown')}")
                    if isinstance(user_interests, list) and user_interests:
                        st.write(f"**Interests:** {', '.join(user_interests[:5])}")
                    
                    # User's liked posts
                    user_liked_posts = []
                    for _, post in posts_df.iterrows():
                        if 'like_user_ids_list' in post and isinstance(post['like_user_ids_list'], list):
                            if user_id in post['like_user_ids_list']:
                                user_liked_posts.append(post['title'])
                    
                    if user_liked_posts:
                        st.write(f"**Previously Liked:** {len(user_liked_posts)} posts")
                        if user_liked_posts:
                            st.caption(f"Sample: {user_liked_posts[0]}")
                
                with col2:
                    st.markdown("**Recommended Posts:**")
                    for i, post_id in enumerate(recommended_post_ids[:3]):
                        post_info = posts_df[posts_df['post_id'].astype(str) == post_id.strip()]
                        if not post_info.empty:
                            post = post_info.iloc[0]
                            st.write(f"**{i+1}.** {post['title']}")
                            st.caption(f"Likes: {post['like_count']}")
        
        # Interest-based analysis
        if 'interests_list' in users_df.columns:
            st.subheader("🏷️ Interest-Based Recommendation Patterns")
            
            # Calculate recommendation diversity by user interests
            interest_analysis = []
            
            for _, user in users_df.head(20).iterrows():  # Sample analysis
                user_id = user['user_id']
                user_interests = user.get('interests_list', [])
                
                if user_id in recommendations_df['user_id'].values:
                    user_recs = recommendations_df[recommendations_df['user_id'] == user_id]['recommended_post_ids'].iloc[0]
                    if user_recs:
                        rec_post_ids = [pid.strip() for pid in user_recs.split(',')]
                        
                        # Get recommended post titles for content analysis
                        rec_posts = posts_df[posts_df['post_id'].astype(str).isin(rec_post_ids)]
                        
                        interest_analysis.append({
                            'user_id': user_id,
                            'num_interests': len(user_interests) if isinstance(user_interests, list) else 0,
                            'num_recommendations': len(rec_post_ids),
                            'avg_post_likes': rec_posts['like_count'].mean() if not rec_posts.empty else 0
                        })
            
            if interest_analysis:
                interest_df = pd.DataFrame(interest_analysis)
                
                fig_interest = px.scatter(
                    interest_df,
                    x='num_interests',
                    y='avg_post_likes',
                    size='num_recommendations',
                    title='User Interests vs Recommended Post Popularity',
                    labels={
                        'num_interests': 'Number of User Interests',
                        'avg_post_likes': 'Average Likes of Recommended Posts',
                        'num_recommendations': 'Number of Recommendations'
                    }
                )
                st.plotly_chart(fig_interest, use_container_width=True)

elif page == "💡 Recommendation Explanations":
    st.header("💡 Recommendation Explanations")
    
    if not st.session_state.recommendations_generated:
        st.warning("⚠️ Please generate recommendations first.")
    else:
        engine = st.session_state.engine
        users_df = st.session_state.users_df
        posts_df = st.session_state.posts_df
        recommendations_df = st.session_state.recommendations_df
        
        st.markdown("""
        This page provides detailed explanations for why specific posts were recommended to users.
        Select a user to see the scoring breakdown for their top recommendations.
        """)
        
        # User selection
        st.subheader("🔍 Select User to Explain")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_id_options = recommendations_df['user_id'].tolist()
            selected_user_id = st.selectbox(
                "Choose User ID",
                user_id_options,
                format_func=lambda x: f"User {x} - {users_df[users_df['user_id'] == x].iloc[0].get('name', 'Unknown') if not users_df[users_df['user_id'] == x].empty else 'Unknown'}"
            )
        
        with col2:
            num_explain = st.slider("Number of posts to explain", 1, 10, 5)
        
        if selected_user_id:
            # Get user info
            user_info = users_df[users_df['user_id'] == selected_user_id].iloc[0]
            user_interests = user_info.get('interests_list', [])
            
            st.subheader(f"📋 User Profile: {user_info.get('name', 'Unknown')}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**User ID:**")
                st.write(selected_user_id)
            
            with col2:
                st.markdown("**Interests:**")
                if isinstance(user_interests, list) and user_interests:
                    st.write(', '.join(user_interests[:5]) + ('...' if len(user_interests) > 5 else ''))
                else:
                    st.write("No interests specified")
            
            with col3:
                user_liked_count = 0
                for _, post in posts_df.iterrows():
                    if 'like_user_ids_list' in post and isinstance(post['like_user_ids_list'], list):
                        if selected_user_id in post['like_user_ids_list']:
                            user_liked_count += 1
                st.markdown("**Previously Liked Posts:**")
                st.write(f"{user_liked_count} posts")
            
            # Get recommendations for this user
            user_recs = recommendations_df[recommendations_df['user_id'] == selected_user_id]['recommended_post_ids'].iloc[0]
            rec_post_ids = [int(pid.strip()) for pid in user_recs.split(',')][:num_explain]
            
            st.subheader("🎯 Recommendation Explanations")
            
            for i, post_id in enumerate(rec_post_ids):
                # Get explanation
                explanation = engine.get_recommendation_explanation(selected_user_id, post_id)
                
                # Get post info
                post_info = posts_df[posts_df['post_id'] == post_id]
                
                if not post_info.empty:
                    post = post_info.iloc[0]
                    
                    with st.expander(f"#{i+1} - {post['title']}", expanded=(i==0)):
                        # Post details
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**Post Content Preview:**")
                            content_preview = post['content'][:200] + '...' if len(str(post['content'])) > 200 else post['content']
                            st.text(content_preview)
                            
                        with col2:
                            st.markdown("**Post Stats:**")
                            st.write(f"👍 Likes: {post['like_count']}")
                            st.write(f"🎭 Anonymous: {post.get('is_anonymous', False)}")
                        
                        # Scoring breakdown
                        st.markdown("---")
                        st.markdown("**📊 Scoring Breakdown:**")
                        
                        # Create score visualization
                        scores_data = {
                            'Component': ['Collaborative\nFiltering', 'Content-Based\nFiltering', 'Popularity\nScore'],
                            'Score': [
                                explanation.get('collaborative_score', 0),
                                explanation.get('content_score', 0),
                                explanation.get('popularity_score', 0)
                            ],
                            'Weight': ['40%', '40%', '20%']
                        }
                        
                        scores_df = pd.DataFrame(scores_data)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Bar chart of scores
                            fig_scores = px.bar(
                                scores_df,
                                x='Component',
                                y='Score',
                                title='Component Scores (Normalized 0-1)',
                                text='Weight',
                                color='Score',
                                color_continuous_scale='viridis'
                            )
                            fig_scores.update_traces(textposition='outside')
                            fig_scores.update_layout(showlegend=False, height=300)
                            st.plotly_chart(fig_scores, use_container_width=True)
                        
                        with col2:
                            st.markdown("**Final Score:**")
                            st.metric("", f"{explanation.get('final_score', 0):.3f}")
                            
                            st.markdown("**Component Contributions:**")
                            st.write(f"🤝 Collaborative: {explanation.get('collaborative_score', 0):.3f}")
                            st.write(f"📝 Content: {explanation.get('content_score', 0):.3f}")
                            st.write(f"⭐ Popularity: {explanation.get('popularity_score', 0):.3f}")
                        
                        # Explanation text
                        st.markdown("---")
                        st.markdown("**🔍 Why This Was Recommended:**")
                        
                        reasons = []
                        
                        if explanation.get('collaborative_score', 0) > 0.5:
                            reasons.append("✅ **Similar users** who liked posts you enjoyed also liked this post")
                        elif explanation.get('collaborative_score', 0) > 0.3:
                            reasons.append("📊 Users with similar preferences showed **moderate interest** in this post")
                        
                        if explanation.get('content_score', 0) > 0.5:
                            reasons.append("✅ Post content **strongly matches** your interests and previously liked posts")
                        elif explanation.get('content_score', 0) > 0.3:
                            reasons.append("📝 Post content has **some alignment** with your interests")
                        
                        if explanation.get('popularity_score', 0) > 0.7:
                            reasons.append("⭐ This is a **highly popular** post in the community")
                        elif explanation.get('popularity_score', 0) > 0.4:
                            reasons.append("👥 This post has **good engagement** from the community")
                        
                        if not reasons:
                            reasons.append("📌 Recommended based on a **balanced mix** of all factors")
                        
                        for reason in reasons:
                            st.markdown(f"- {reason}")
            
            # Summary section
            st.subheader("📈 Overall Recommendation Strategy")
            st.markdown("""
            **How the hybrid system works:**
            
            1. **Collaborative Filtering (40%)**: Finds users with similar liking patterns and recommends posts they enjoyed
            2. **Content-Based Filtering (40%)**: Matches post content to your interests and previously liked content  
            3. **Popularity Metrics (20%)**: Considers community engagement to surface trending content
            
            Each post receives a score from all three components, which are combined using the weights above to produce a final recommendation score.
            """)

elif page == "📋 Export Results":
    st.header("📋 Export Results")
    
    if not st.session_state.recommendations_generated:
        st.warning("⚠️ Please generate recommendations first.")
    else:
        recommendations_df = st.session_state.recommendations_df
        
        st.subheader("📊 Export Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Users", len(recommendations_df))
        with col2:
            total_recs = recommendations_df['recommended_post_ids'].apply(
                lambda x: len(x.split(',')) if x else 0
            ).sum()
            st.metric("Total Recommendations", total_recs)
        with col3:
            avg_recs = total_recs / len(recommendations_df) if len(recommendations_df) > 0 else 0
            st.metric("Avg Recs/User", f"{avg_recs:.1f}")
        
        # Preview export data
        st.subheader("📋 Export Preview")
        st.dataframe(recommendations_df)
        
        # Download button
        st.subheader("💾 Download Results")
        
        csv_data = recommendations_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendations CSV",
            data=csv_data,
            file_name="boomm_recommendations.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Export statistics
        st.subheader("📈 Export Statistics")
        
        # Recommendation distribution
        rec_counts = recommendations_df['recommended_post_ids'].apply(
            lambda x: len(x.split(',')) if x else 0
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = px.histogram(
                rec_counts,
                nbins=20,
                title='Distribution of Recommendations per User',
                labels={'value': 'Number of Recommendations', 'count': 'Number of Users'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    f"{rec_counts.mean():.1f}",
                    f"{rec_counts.median():.1f}",
                    f"{rec_counts.std():.1f}",
                    f"{rec_counts.min():.0f}",
                    f"{rec_counts.max():.0f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True)
        
        # Additional export options
        st.subheader("🔧 Additional Export Options")
        
        if st.button("📊 Export Detailed Analysis"):
            # Create comprehensive analysis report
            analysis_data = []
            
            for _, row in recommendations_df.iterrows():
                user_id = row['user_id']
                recommended_posts = row['recommended_post_ids'].split(',') if row['recommended_post_ids'] else []
                
                # Get user info
                user_info = st.session_state.users_df[st.session_state.users_df['user_id'] == user_id]
                if not user_info.empty:
                    user_name = user_info.iloc[0].get('name', 'Unknown')
                    user_interests = user_info.iloc[0].get('interests_list', [])
                    
                    for post_id in recommended_posts:
                        post_id = post_id.strip()
                        post_info = st.session_state.posts_df[st.session_state.posts_df['post_id'].astype(str) == post_id]
                        
                        if not post_info.empty:
                            post = post_info.iloc[0]
                            analysis_data.append({
                                'user_id': user_id,
                                'user_name': user_name,
                                'user_interests': ', '.join(user_interests) if isinstance(user_interests, list) else '',
                                'recommended_post_id': post_id,
                                'post_title': post['title'],
                                'post_likes': post['like_count'],
                                'is_anonymous': post.get('is_anonymous', False)
                            })
            
            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                analysis_csv = analysis_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Detailed Analysis CSV",
                    data=analysis_csv,
                    file_name="boomm_detailed_analysis.csv",
                    mime="text/csv"
                )
                
                st.success("✅ Detailed analysis prepared for download!")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ for Boomm Social Platform*")
