import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 1. RENDER DEPLOYMENT OPTIMIZATION ---
# Force the transformer model to use CPU to prevent memory crashes on Render's free tier
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- 2. CONFIGURATION & AI MODEL ---
st.set_page_config(
    page_title="AI E-Commerce Insights",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_hf_model():
    """Load the DistilBERT transformer model from Hugging Face."""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    return pipeline("sentiment-analysis", model=model_name)

# Initialize classifier
classifier = load_hf_model()

@st.cache_data
def fetch_data():
    """Load and clean the scraped JSON data."""
    # Build relative path for cloud compatibility
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'data', 'scraped_data.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df_p = pd.DataFrame(data['products'])
        df_t = pd.DataFrame(data['testimonials'])
        df_r = pd.DataFrame(data['reviews'])
        
        # Data Cleaning: Convert price strings to numbers and dates to objects
        df_p['price'] = pd.to_numeric(df_p['price'].replace(r'[^\d.]', '', regex=True), errors='coerce')
        df_r['date'] = pd.to_datetime(df_r['date'])
        
        return df_p, df_t, df_r
    except Exception as e:
        st.error(f"Error loading data file: {e}")
        return None, None, None

df_prod, df_testi, df_rev = fetch_data()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

if df_prod is not None:

    # --- 4. PAGE: PRODUCTS ---
    if page == "Products":
        st.header("📦 Scraped Product Catalog")
        st.write("Full product listing from multi-page scraping.")
        
        m1, m2 = st.columns(2)
        m1.metric("Total Items", len(df_prod))
        m2.metric("Avg. Price", f"${df_prod['price'].mean():.2f}")
        
        st.divider()
        st.dataframe(df_prod, use_container_width=True, hide_index=True)

    # --- 5. PAGE: TESTIMONIALS ---
    elif page == "Testimonials":
        st.header("💬 Customer Testimonials")
        st.divider()
        
        col_left, col_right = st.columns(2)
        for i, row in df_testi.iterrows():
            target = col_left if i % 2 == 0 else col_right
            with target:
                with st.container(border=True):
                    h1, h2 = st.columns([3, 1])
                    h1.markdown(f"👤 **Customer #{row['id']}**")
                    stars = "⭐" * int(row['rating'])
                    h2.markdown(f"<p style='text-align: right;'>{stars}</p>", unsafe_allow_html=True)
                    st.write(f"_{row['text']}_")

    # --- 6. PAGE: REVIEWS (Core Feature: AI & Visualization) ---
    elif page == "Reviews":
        st.header("⭐ Sentiment Analysis & Monthly Insights")
        
        months_list = ["January", "February", "March", "April", "May", "June", 
                       "July", "August", "September", "October", "November", "December"]
        selected_month = st.selectbox("Select Month (2023):", options=months_list, index=4)

        filtered_df = df_rev[
            (df_rev['date'].dt.year == 2023) & 
            (df_rev['date'].dt.month_name() == selected_month)
        ].copy()

        st.divider()

        if not filtered_df.empty:
            # AI Classification with Spinner
            with st.spinner(f'🤖 AI is classifying {len(filtered_df)} reviews...'):
                raw_texts = filtered_df['text'].tolist()
                ai_results = classifier(raw_texts)
                
                filtered_df['Sentiment'] = [res['label'] for res in ai_results]
                filtered_df['Raw_Score'] = [res['score'] for res in ai_results]
                filtered_df['Confidence'] = filtered_df['Raw_Score'].apply(lambda x: f"{x:.1%}")

            # --- SIDE-BY-SIDE VISUALIZATION ---
            chart_col, table_col = st.columns([1, 2], gap="large")

            with chart_col:
                # Part A: Sentiment Bar Chart
                st.subheader("Sentiment Distribution")
                chart_data = filtered_df.groupby('Sentiment').agg(
                    Count=('Sentiment', 'size'),
                    Avg_Conf=('Raw_Score', 'mean')
                ).reset_index()

                fig = px.bar(
                    chart_data, 
                    x='Sentiment', y='Count', color='Sentiment',
                    text='Count',
                    color_discrete_map={'POSITIVE': '#00875A', 'NEGATIVE': '#DE350B'},
                    hover_data={'Avg_Conf': ':.1%'},
                    height=250
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False, xaxis_title=None, yaxis_title=None,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Part B: Word Cloud
                st.subheader("Key Keywords")
                text_combined = " ".join(review for review in filtered_df['text'])
                if text_combined.strip():
                    wc = WordCloud(
                        background_color="white", 
                        max_words=50, 
                        colormap='Dark2',
                        width=600,
                        height=300
                    ).generate(text_combined)
                    
                    fig_wc, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig_wc)
                
                st.metric("Avg. Model Confidence", f"{filtered_df['Raw_Score'].mean():.1%}")

            with table_col:
                st.subheader("Detailed Reviews")
                
                def sentiment_color(val):
                    bg = '#00875A' if val == 'POSITIVE' else '#DE350B'
                    return f'background-color: {bg}; color: white; font-weight: bold; border-radius: 4px;'

                st.dataframe(
                    filtered_df[['date', 'Sentiment', 'Confidence', 'text']].style.applymap(
                        sentiment_color, subset=['Sentiment']
                    ),
                    use_container_width=True, 
                    hide_index=True,
                    height=600 
                )
        else:
            st.warning(f"No reviews found for {selected_month} 2023.")

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.caption("Homework #3 | Streamlit + Hugging Face + WordCloud")