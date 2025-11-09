import streamlit as st
import pickle
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ========================================
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: white;
        padding: 20px;
        margin-bottom: 30px;
    }
    
    .main-header h1 {
        font-size: 3.5em;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2em;
        opacity: 0.9;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1.1em;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1em;
        font-weight: 600;
        padding: 15px 30px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
        font-size: 1.1em;
        padding: 15px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Result boxes */
    .result-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        margin: 20px 0;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        margin: 20px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ========================================
# LOAD MODELS
# ========================================
@st.cache_resource
def load_models():
    """Load the trained models from pickle files"""
    try:
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('models/logistic_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return tfidf, model
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'models/' directory contains the required .pkl files.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

# ========================================
# TEXT PREPROCESSING
# ========================================
def clean_text(text):
    """Clean and preprocess text (same as training)"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # Remove special characters
    text = ' '.join(text.split())                # Remove extra spaces
    return text

# ========================================
# PREDICTION FUNCTION
# ========================================
def predict_sentiment(text, tfidf, model):
    """
    Predict sentiment of input text
    Returns: sentiment (str), confidence (float), probabilities (dict)
    """
    # Clean text
    cleaned_text = clean_text(text)
    
    # Transform to TF-IDF features
    text_vectorized = tfidf.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    
    # Get probabilities
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # Get confidence (max probability)
    confidence = max(probabilities) * 100
    
    # Create probability dictionary
    prob_dict = {
        'Negative': probabilities[0] * 100,
        'Positive': probabilities[1] * 100
    }
    
    return prediction, confidence, prob_dict

# ========================================
# MAIN APP
# ========================================
def main():
    # Load models
    tfidf, model = load_models()
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎭 Sentiment Analysis App</h1>
            <p>Powered by Machine Learning | Trained on 150,000+ Amazon Reviews</p>
        </div>
    """, unsafe_allow_html=True)
    
    # ========================================
    # SIDEBAR - INFO & STATS
    # ========================================
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
        st.title("📊 Model Info")
        
        st.markdown("---")
        
        st.markdown("### 🤖 Algorithm")
        st.info("**Logistic Regression**")
        
        st.markdown("### 📈 Accuracy")
        st.success("**88.96%** on test data")
        
        st.markdown("### 📚 Training Data")
        st.info("**150,000** Amazon Reviews")
        
        st.markdown("### 🔤 Features")
        st.info("**10,000** TF-IDF features")
        
        st.markdown("---")
        
        st.markdown("### 💡 How it works")
        st.write("""
        1. **Text Preprocessing**: Clean & normalize text
        2. **Feature Extraction**: Convert to TF-IDF vectors
        3. **Classification**: Predict sentiment using ML
        """)
        
        st.markdown("---")
        
        # Analytics if history exists
        if 'history' in st.session_state and len(st.session_state.history) > 0:
            st.markdown("### 📊 Session Analytics")
            total = len(st.session_state.history)
            positive = sum(1 for h in st.session_state.history if h['sentiment'] == 'Positive')
            negative = total - positive
            
            st.metric("Total Analyses", total)
            st.metric("Positive", f"{positive} ({positive/total*100:.1f}%)")
            st.metric("Negative", f"{negative} ({negative/total*100:.1f}%)")
    
    # ========================================
    # MAIN CONTENT AREA
    # ========================================
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Analyze Text", "📝 Batch Analysis", "📊 History", "ℹ️ About"])
    
    # ========================================
    # TAB 1: SINGLE TEXT ANALYSIS
    # ========================================
    with tab1:
        st.markdown("### 🎯 Analyze Single Review")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            user_input = st.text_area(
                "Enter your text here:",
                height=200,
                placeholder="e.g., This product is amazing! I love it so much. Best purchase ever!",
                help="Enter any product review, comment, or feedback to analyze its sentiment"
            )
            
            # Quick examples
            st.markdown("**Quick Examples:**")
            example_col1, example_col2, example_col3 = st.columns(3)
            
            with example_col1:
                if st.button("😊 Positive Example"):
                    user_input = "This product is absolutely amazing! Best purchase I've ever made. Highly recommend to everyone!"
                    st.rerun()
            
            with example_col2:
                if st.button("😞 Negative Example"):
                    user_input = "Terrible quality. Broke after one day. Complete waste of money. Very disappointed."
                    st.rerun()
            
            with example_col3:
                if st.button("😐 Mixed Example"):
                    user_input = "The product is okay. Quality is decent but nothing special. Price could be better."
                    st.rerun()
            
            # Analyze button
            analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")
        
        with col2:
            st.markdown("### 📌 Tips")
            st.info("""
            ✅ Enter honest reviews
            
            ✅ Be descriptive
            
            ✅ Any length works
            
            ✅ Try different tones
            """)
        
        # Analysis results
        if analyze_button and user_input.strip():
            with st.spinner("🔄 Analyzing sentiment..."):
                time.sleep(0.5)  # Small delay for effect
                
                # Get prediction
                sentiment, confidence, prob_dict = predict_sentiment(user_input, tfidf, model)
                
                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Analysis Results")
                
                # Result card
                if sentiment == 'Positive':
                    st.markdown(f"""
                        <div class="result-positive">
                            <div style="font-size: 3em; margin-bottom: 10px;">😊 ✅</div>
                            <div>POSITIVE SENTIMENT</div>
                            <div style="font-size: 0.8em; margin-top: 10px; opacity: 0.9;">
                                Confidence: {confidence:.1f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-negative">
                            <div style="font-size: 3em; margin-bottom: 10px;">😞 ❌</div>
                            <div>NEGATIVE SENTIMENT</div>
                            <div style="font-size: 0.8em; margin-top: 10px; opacity: 0.9;">
                                Confidence: {confidence:.1f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Detailed metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Sentiment</div>
                            <div class="metric-value">{sentiment}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value">{confidence:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    word_count = len(user_input.split())
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Words</div>
                            <div class="metric-value">{word_count}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Probability gauge chart
                st.markdown("### 📊 Confidence Breakdown")
                
                fig = go.Figure(go.Bar(
                    x=[prob_dict['Positive'], prob_dict['Negative']],
                    y=['Positive', 'Negative'],
                    orientation='h',
                    marker=dict(
                        color=['#38ef7d', '#f45c43'],
                        line=dict(color='white', width=2)
                    ),
                    text=[f"{prob_dict['Positive']:.1f}%", f"{prob_dict['Negative']:.1f}%"],
                    textposition='auto',
                    textfont=dict(size=14, color='white', family='Arial Black')
                ))
                
                fig.update_layout(
                    title="Sentiment Probabilities",
                    xaxis_title="Confidence (%)",
                    yaxis_title="Sentiment",
                    height=300,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    xaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Save to history
                if 'history' not in st.session_state:
                    st.session_state.history = []
                
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                    'sentiment': sentiment,
                    'confidence': confidence
                })
        
        elif analyze_button and not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze!")
    
    # ========================================
    # TAB 2: BATCH ANALYSIS
    # ========================================
    with tab2:
        st.markdown("### 📝 Batch Analysis")
        st.info("Analyze multiple reviews at once! Enter one review per line.")
        
        batch_input = st.text_area(
            "Enter multiple reviews (one per line):",
            height=300,
            placeholder="Great product!\nTerrible quality.\nLove it!"
        )
        
        if st.button("🔍 Analyze All", use_container_width=True, type="primary"):
            if batch_input.strip():
                reviews = [r.strip() for r in batch_input.split('\n') if r.strip()]
                
                if len(reviews) > 0:
                    with st.spinner(f"🔄 Analyzing {len(reviews)} reviews..."):
                        results = []
                        for review in reviews:
                            sentiment, confidence, _ = predict_sentiment(review, tfidf, model)
                            results.append({
                                'Review': review[:50] + "..." if len(review) > 50 else review,
                                'Sentiment': sentiment,
                                'Confidence': f"{confidence:.1f}%"
                            })
                        
                        # Display results
                        st.success(f"✅ Analyzed {len(reviews)} reviews!")
                        
                        # Summary
                        positive_count = sum(1 for r in results if r['Sentiment'] == 'Positive')
                        negative_count = len(results) - positive_count
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Reviews", len(results))
                        with col2:
                            st.metric("Positive", f"{positive_count} ({positive_count/len(results)*100:.1f}%)")
                        with col3:
                            st.metric("Negative", f"{negative_count} ({negative_count/len(results)*100:.1f}%)")
                        
                        # Results table
                        st.markdown("### 📊 Detailed Results")
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Pie chart
                        fig = px.pie(
                            values=[positive_count, negative_count],
                            names=['Positive', 'Negative'],
                            title='Sentiment Distribution',
                            color_discrete_sequence=['#38ef7d', '#f45c43']
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Please enter at least one review!")
    
    # ========================================
    # TAB 3: HISTORY
    # ========================================
    with tab3:
        st.markdown("### 📊 Analysis History")
        
        if 'history' in st.session_state and len(st.session_state.history) > 0:
            # Summary metrics
            total = len(st.session_state.history)
            positive = sum(1 for h in st.session_state.history if h['sentiment'] == 'Positive')
            negative = total - positive
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analyses", total)
            with col2:
                st.metric("Positive", f"{positive} ({positive/total*100:.1f}%)", delta="Good" if positive > negative else None)
            with col3:
                st.metric("Negative", f"{negative} ({negative/total*100:.1f}%)", delta="Bad" if negative > positive else None)
            
            # History table
            st.markdown("---")
            df_history = pd.DataFrame(st.session_state.history)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            
            # Clear history button
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("📝 No analysis history yet. Start analyzing some text!")
    
    # ========================================
    # TAB 4: ABOUT
    # ========================================
    with tab4:
        st.markdown("### ℹ️ About This Project")
        
        st.markdown("""
        ## 🎯 Sentiment Analysis Application
        
        This application uses **Machine Learning** to automatically determine whether text expresses 
        a **positive** or **negative** sentiment.
        
        ### 🔧 Technical Details
        
        - **Algorithm**: Logistic Regression
        - **Training Data**: 150,000 Amazon Product Reviews
        - **Accuracy**: 88.96% on test set
        - **Features**: TF-IDF vectorization with 10,000 features
        - **Framework**: Streamlit + Scikit-learn
        
        ### 📊 Model Performance
        
        | Metric | Score |
        |--------|-------|
        | Accuracy | 88.96% |
        | Precision | 89.2% |
        | Recall | 88.7% |
        | F1-Score | 88.9% |
        
        ### 🚀 Use Cases
        
        - 📦 **E-commerce**: Analyze product reviews
        - 💬 **Social Media**: Monitor brand sentiment
        - 📧 **Customer Support**: Prioritize negative feedback
        - 📊 **Market Research**: Understand customer opinions
        
        ### 👨‍💻 Developer
        
        Built with ❤️ using Python, Streamlit, and Scikit-learn
        
        ---
        
        *For questions or feedback, feel free to reach out!*
        """)

# ========================================
# RUN APP
# ========================================
if __name__ == "__main__":
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    main()
