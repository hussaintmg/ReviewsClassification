# app.py
import streamlit as st
import re
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        padding: 2rem;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        animation: fadeIn 0.5s;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #ff6a6a, #ff3a3a);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer with caching
@st.cache_resource
def load_models():
    try:
        model = load("logistic_model.pkl")
        vectorizer = load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Please train the model first.")
        return None, None

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub('<.*?>', '', text)  # remove HTML
    text = re.sub('[^a-zA-Z]', ' ', text)  # remove symbols
    text = text.split()
    return " ".join(text)

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for a single text"""
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    return prediction, probability

def analyze_batch_texts(texts, model, vectorizer):
    """Analyze multiple texts"""
    results = []
    for text in texts:
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        pred = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]
        results.append({
            'Review': text[:100] + "..." if len(text) > 100 else text,
            'Sentiment': 'Positive' if pred == 1 else 'Negative',
            'Confidence (%)': round(max(prob) * 100, 2)
        })
    return results

def main():
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.title("🎬 About")
        st.markdown("---")
        st.markdown("""
        ### 📊 Model Information
        - **Model:** Logistic Regression
        - **Features:** TF-IDF with n-grams (1,2)
        - **Accuracy:** 91.29%
        - **Dataset:** 50K IMDB Reviews
        
        ### 💡 How to Use
        1. Enter your movie review
        2. Click "Analyze Sentiment"
        3. Get instant prediction
        """)
    
    # Main content
    st.title("🎬 Movie Review Sentiment Analyzer")
    st.markdown("*Powered by Logistic Regression | 91.29% Accuracy*")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Review", "📊 Batch Analysis", "📈 Model Insights"])
    
    # Tab 1: Single Review
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Use session state for persistent input
            review_text = st.text_area(
                "✍️ Enter your movie review:",
                height=150,
                placeholder="e.g., This movie was incredible! The acting was superb and the story kept me engaged throughout...",
                key="review_input",
                value=st.session_state.get('review_text', '')
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        with col2:
            st.markdown("### 💡 Tips for better results")
            st.info("""
            ✅ Use clear opinions  
            ✅ Mention specific aspects  
            ✅ Write natural language  
            ❌ Avoid mixed sentiments  
            ❌ Avoid very short reviews
            """)
        
        if analyze_btn and review_text:
            if len(review_text.strip()) < 10:
                st.warning("⚠️ Please enter a longer review (at least 10 characters) for better analysis.")
            else:
                with st.spinner("🧠 Analyzing your review..."):
                    prediction, probability = predict_sentiment(review_text, model, vectorizer)
                
                # Display result
                if prediction == 1:
                    st.markdown(f"""
                    <div class="sentiment-positive">
                        <h2>😊 POSITIVE SENTIMENT</h2>
                        <h3>Confidence: {max(probability)*100:.2f}%</h3>
                        <p>This review expresses positive feedback about the movie!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="sentiment-negative">
                        <h2>😞 NEGATIVE SENTIMENT</h2>
                        <h3>Confidence: {max(probability)*100:.2f}%</h3>
                        <p>This review expresses negative feedback about the movie!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display probability chart using matplotlib
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sentiments = ['Positive', 'Negative']
                    probabilities = [probability[1]*100, probability[0]*100]
                    colors = ['#00b09b', '#ff6a6a']
                    bars = ax.bar(sentiments, probabilities, color=colors)
                    ax.set_ylabel('Probability (%)')
                    ax.set_title('Sentiment Probability Distribution')
                    ax.set_ylim([0, 100])
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{prob:.1f}%', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("### 📊 Analysis Details")
                    st.metric("Prediction Confidence", f"{max(probability)*100:.1f}%")
                    st.metric("Positive Probability", f"{probability[1]*100:.1f}%")
                    st.metric("Negative Probability", f"{probability[0]*100:.1f}%")
                    
                    with st.expander("🔍 View cleaned text"):
                        cleaned = clean_text(review_text)
                        st.code(cleaned[:200] + "..." if len(cleaned) > 200 else cleaned)
    
    # Tab 2: Batch Analysis
    with tab2:
        st.subheader("📊 Batch Sentiment Analysis")
        
        upload_option = st.radio("Choose input method:", ["Enter multiple reviews", "Upload CSV file"])
        
        if upload_option == "Enter multiple reviews":
            batch_text = st.text_area(
                "Enter multiple reviews (one per line):",
                height=200,
                placeholder="This movie is great!\nTerrible film, wasted my time.\nPretty good but could be better.",
                help="Enter one review per line"
            )
            
            if st.button("📊 Analyze Batch", type="primary", use_container_width=True):
                if batch_text:
                    reviews = [r.strip() for r in batch_text.split('\n') if r.strip()]
                    if reviews:
                        with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                            results = analyze_batch_texts(reviews, model, vectorizer)
                            
                            # Display summary statistics
                            col1, col2, col3 = st.columns(3)
                            positive_count = sum(1 for r in results if r['Sentiment'] == 'Positive')
                            negative_count = len(results) - positive_count
                            avg_confidence = sum(r['Confidence (%)'] for r in results) / len(results)
                            
                            with col1:
                                st.metric("📊 Total Reviews", len(results))
                            with col2:
                                st.metric("😊 Positive Reviews", positive_count, 
                                         delta=f"{positive_count/len(results)*100:.0f}%")
                            with col3:
                                st.metric("📈 Average Confidence", f"{avg_confidence:.1f}%")
                            
                            # Display results in table
                            results_df = pd.DataFrame(results)
                            st.dataframe(
                                results_df,
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
        
        else:  # CSV upload
            uploaded_file = st.file_uploader("Upload CSV file with reviews", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded file:")
                st.dataframe(df.head())
                
                if 'review' in df.columns:
                    if st.button("📊 Analyze CSV File", type="primary", use_container_width=True):
                        with st.spinner(f"Analyzing {len(df)} reviews..."):
                            results = analyze_batch_texts(df['review'].tolist(), model, vectorizer)
                            results_df = pd.DataFrame(results)
                            
                            # Summary
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Reviews", len(results_df))
                            with col2:
                                positive_pct = (results_df['Sentiment'] == 'Positive').sum() / len(results_df) * 100
                                st.metric("Positive Rate", f"{positive_pct:.1f}%")
                            
                            st.dataframe(results_df, use_container_width=True, hide_index=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="analysis_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                else:
                    st.error("❌ CSV must contain a 'review' column")
    
    # Tab 3: Model Insights
    with tab3:
        st.markdown("### 🎯 Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "91.29%", delta="Excellent")
        with col2:
            st.metric("Precision", "91.5%", delta="High")
        with col3:
            st.metric("Recall", "91.0%", delta="Good")
        with col4:
            st.metric("F1-Score", "91.2%", delta="Strong")
        
        st.markdown("---")
        st.markdown("### ⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **Algorithm Details**
            - Model: Logistic Regression
            - Regularization: C = 2
            - Max Iterations: 3000
            - Random State: 42
            """)
        
        with col2:
            st.info("""
            **Feature Engineering**
            - Vectorizer: TF-IDF
            - Max Features: 30,000
            - n-gram range: (1,2)
            - Min Document Frequency: 2
            - Max Document Frequency: 0.9
            """)
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Reviews", "50,000")
            st.metric("Training Set", "40,000 (80%)")
        with col2:
            st.metric("Test Set", "10,000 (20%)")
            st.metric("Classes", "Positive / Negative")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: rgba(255,255,255,0.7);'>
        Built by Hussain
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()