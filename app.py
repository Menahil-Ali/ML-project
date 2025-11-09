import streamlit as st
import pickle
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        font-size: 16px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .positive {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .negative {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #1f1f1f;
        text-align: center;
        font-size: 48px;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Text cleaning function
def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Load models function
@st.cache_resource
def load_models():
    """Load all trained models and vectorizer"""
    try:
        tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
        nb_model = pickle.load(open('models/naive_bayes_model.pkl', 'rb'))
        lr_model = pickle.load(open('models/logistic_regression_model.pkl', 'rb'))
        rf_model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
        return tfidf, nb_model, lr_model, rf_model
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure all model files are in the 'models/' folder.")
        return None, None, None, None

# Prediction function
def predict_sentiment(text, model, tfidf):
    """Predict sentiment using selected model"""
    cleaned = clean_text(text)
    text_vectorized = tfidf.transform([cleaned])
    prediction = model.predict(text_vectorized)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_vectorized)[0]
        confidence = max(proba) * 100
        pos_prob = proba[1] * 100 if prediction == 'Positive' else proba[0] * 100
        neg_prob = proba[0] * 100 if prediction == 'Positive' else proba[1] * 100
    else:
        confidence = None
        pos_prob = None
        neg_prob = None
    
    return prediction, confidence, pos_prob, neg_prob

# Main app
def main():
    # Header
    st.markdown("<h1>🎭 Sentiment Analysis App</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Analyze customer reviews using Machine Learning • Trained on 150,000+ Amazon Reviews</p>", unsafe_allow_html=True)
    
    # Load models
    tfidf, nb_model, lr_model, rf_model = load_models()
    
    if tfidf is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/machine-learning.png", width=150)
        st.title("⚙️ Settings")
        st.markdown("---")
        
        # Model selection
        model_option = st.selectbox(
            "🤖 Select ML Algorithm",
            ["Logistic Regression", "Naive Bayes", "Random Forest"],
            help="Choose which machine learning model to use for prediction"
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 📊 Model Information")
        if model_option == "Logistic Regression":
            st.info("**Best Overall Performance**\n\n✅ Accuracy: ~89%\n⚡ Speed: Fast\n💡 Best for: General use")
        elif model_option == "Naive Bayes":
            st.info("**Fastest Model**\n\n✅ Accuracy: ~87%\n⚡ Speed: Very Fast\n💡 Best for: Real-time analysis")
        else:
            st.info("**Most Robust**\n\n✅ Accuracy: ~88%\n⚡ Speed: Slower\n💡 Best for: Complex patterns")
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📈 Dataset Stats")
        st.metric("Training Samples", "120,000")
        st.metric("Test Samples", "30,000")
        st.metric("Features Used", "10,000")
        
        st.markdown("---")
        st.markdown("### 👨‍💻 About")
        st.markdown("""
        Built with:
        - Streamlit
        - Scikit-learn
        - TF-IDF Vectorization
        - 3 ML Algorithms
        """)
    
    # Select model based on user choice
    if model_option == "Logistic Regression":
        selected_model = lr_model
    elif model_option == "Naive Bayes":
        selected_model = nb_model
    else:
        selected_model = rf_model
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📚 Examples"])
    
    # TAB 1: Single Prediction
    with tab1:
        st.markdown("### 📝 Enter Your Review")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area(
                "Type or paste your product review here:",
                height=150,
                placeholder="Example: This product is amazing! Best purchase I've ever made. Highly recommended!",
                help="Enter any product review to analyze its sentiment"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
            clear_button = st.button("🗑️ Clear", use_container_width=True)
            
            if clear_button:
                st.rerun()
        
        if analyze_button and user_input:
            with st.spinner("🔄 Analyzing sentiment..."):
                prediction, confidence, pos_prob, neg_prob = predict_sentiment(
                    user_input, selected_model, tfidf
                )
                
                # Results
                st.markdown("---")
                st.markdown("### 🎯 Analysis Results")
                
                # Prediction box
                if prediction == "Positive":
                    st.markdown(f"""
                        <div class='prediction-box positive'>
                            😊 POSITIVE SENTIMENT
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='prediction-box negative'>
                            😞 NEGATIVE SENTIMENT
                        </div>
                    """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Model Used", model_option)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    if confidence:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    else:
                        st.metric("Confidence", "N/A")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Text Length", f"{len(user_input)} chars")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Probability chart
                if pos_prob is not None and neg_prob is not None:
                    st.markdown("### 📊 Probability Distribution")
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Positive', 'Negative'],
                            y=[pos_prob, neg_prob],
                            marker_color=['#667eea', '#f5576c'],
                            text=[f'{pos_prob:.1f}%', f'{neg_prob:.1f}%'],
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title="Sentiment Probability",
                        yaxis_title="Probability (%)",
                        height=400,
                        showlegend=False,
                        plot_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif analyze_button and not user_input:
            st.warning("⚠️ Please enter some text to analyze!")
    
    # TAB 2: Batch Analysis
    with tab2:
        st.markdown("### 📊 Analyze Multiple Reviews")
        st.markdown("Enter multiple reviews (one per line) to analyze them all at once.")
        
        batch_input = st.text_area(
            "Enter reviews (one per line):",
            height=200,
            placeholder="Great product!\nTerrible quality, don't buy.\nWorks as expected.\nBest purchase ever!",
            help="Each line will be analyzed separately"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            batch_analyze = st.button("📈 Analyze All", type="primary", use_container_width=True)
        with col2:
            batch_clear = st.button("🗑️ Clear All", use_container_width=True)
        
        if batch_clear:
            st.rerun()
        
        if batch_analyze and batch_input:
            reviews = [r.strip() for r in batch_input.split('\n') if r.strip()]
            
            if len(reviews) > 0:
                with st.spinner(f"🔄 Analyzing {len(reviews)} reviews..."):
                    results = []
                    
                    for review in reviews:
                        pred, conf, _, _ = predict_sentiment(review, selected_model, tfidf)
                        results.append({
                            'Review': review[:100] + '...' if len(review) > 100 else review,
                            'Sentiment': pred,
                            'Confidence': f"{conf:.1f}%" if conf else "N/A"
                        })
                    
                    # Create DataFrame
                    df_results = pd.DataFrame(results)
                    
                    # Summary metrics
                    st.markdown("---")
                    st.markdown("### 📊 Summary Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    positive_count = sum(1 for r in results if r['Sentiment'] == 'Positive')
                    negative_count = len(results) - positive_count
                    
                    with col1:
                        st.metric("Total Reviews", len(reviews))
                    with col2:
                        st.metric("Positive", positive_count, delta=f"{positive_count/len(reviews)*100:.1f}%")
                    with col3:
                        st.metric("Negative", negative_count, delta=f"{negative_count/len(reviews)*100:.1f}%")
                    with col4:
                        sentiment = "Satisfied ✅" if positive_count > negative_count else "Unsatisfied ⚠️"
                        st.metric("Overall", sentiment)
                    
                    # Pie chart
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pie = px.pie(
                            values=[positive_count, negative_count],
                            names=['Positive', 'Negative'],
                            title='Sentiment Distribution',
                            color_discrete_sequence=['#667eea', '#f5576c']
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Results table
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(df_results, use_container_width=True, height=400)
                    
                    # Download button
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # TAB 3: Examples
    with tab3:
        st.markdown("### 📚 Example Reviews")
        st.markdown("Click on any example to see how the model analyzes it.")
        
        examples = {
            "🌟 Highly Positive": [
                "This product is absolutely amazing! Best purchase I've ever made. Highly recommend to everyone!",
                "Exceeded all my expectations! Outstanding quality and fast shipping. Will definitely buy again!",
                "Love it! Perfect in every way. My family is so happy with this purchase."
            ],
            "😊 Positive": [
                "Good product. Does what it's supposed to do. Happy with the purchase.",
                "Nice quality and arrived on time. Would recommend.",
                "Pretty satisfied with this. Works well for the price."
            ],
            "😐 Neutral": [
                "It's okay. Nothing special but gets the job done.",
                "Average product. Neither good nor bad.",
                "Decent for the price. Has some pros and cons."
            ],
            "😞 Negative": [
                "Not satisfied with the quality. Expected better.",
                "Disappointed. Doesn't work as advertised.",
                "Below average. Would not recommend this product."
            ],
            "⛔ Highly Negative": [
                "Terrible quality! Broke after one day. Complete waste of money!",
                "Awful product. Don't buy this. Worst purchase ever!",
                "Horrible experience. Poor quality and terrible customer service."
            ]
        }
        
        for category, reviews in examples.items():
            with st.expander(f"**{category}**"):
                for idx, example in enumerate(reviews):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"*{example}*")
                    with col2:
                        if st.button("Try This", key=f"{category}_{idx}"):
                            pred, conf, _, _ = predict_sentiment(example, selected_model, tfidf)
                            
                            if pred == "Positive":
                                st.success(f"✅ {pred} ({conf:.1f}%)")
                            else:
                                st.error(f"❌ {pred} ({conf:.1f}%)")
                    st.markdown("---")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🎯 Built with Streamlit & Scikit-learn | Trained on 150,000+ Amazon Reviews</p>
            <p>Made with ❤️ for sentiment analysis</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
