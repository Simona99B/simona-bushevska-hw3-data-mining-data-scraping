import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AI E-Commerce Insights",
    page_icon="🤖",
    layout="wide"
)

# --- 2. DATA LOADING ---
@st.cache_data
def fetch_data():
    """Load pre-processed JSON data from the data folder."""
    # Build relative path to look inside the /data folder
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'data', 'processed_data.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df_p = pd.DataFrame(data['products'])
        df_t = pd.DataFrame(data['testimonials'])
        df_r = pd.DataFrame(data['reviews'])
        
        # Clean Data Types
        df_p['price'] = pd.to_numeric(df_p['price'].replace(r'[^\d.]', '', regex=True), errors='coerce')
        df_r['date'] = pd.to_datetime(df_r['date'])
        
        if 'Confidence' in df_r.columns:
            df_r['Confidence'] = pd.to_numeric(df_r['Confidence'], errors='coerce')
        
        return df_p, df_t, df_r
    except Exception as e:
        st.error(f"Error loading 'data/processed_data.json': {e}")
        return None, None, None

df_prod, df_testi, df_rev = fetch_data()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

if df_prod is not None:

    # --- 4. PAGE: PRODUCTS ---
    if page == "Products":
        st.header("📦 Product Catalog")
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
                    st.markdown(f"👤 **Customer #{row['id']}**")
                    st.write(f"_{row['text']}_")
                    st.caption(f"Rating: {'⭐' * int(row['rating'])}")

    # --- 6. PAGE: REVIEWS ---
    elif page == "Reviews":
        st.header("⭐ 2023 Monthly Sentiment Insights")
        
        months_list = ["January", "February", "March", "April", "May", "June", 
                       "July", "August", "September", "October", "November", "December"]
        
        # CHANGED: select_slider replaced with selectbox (Dropdown)
        selected_month = st.selectbox("Select Month (2023):", options=months_list, index=4)

        # STRICT FILTER: Ensure we only show 2023 data for the chosen month
        filtered_df = df_rev[
            (df_rev['date'].dt.year == 2023) & 
            (df_rev['date'].dt.month_name() == selected_month)
        ].copy()

        st.divider()

        if not filtered_df.empty:
            chart_col, table_col = st.columns([1, 2], gap="large")

            with chart_col:
                st.subheader("Sentiment Distribution")
                chart_data = filtered_df.groupby('Sentiment').size().reset_index(name='Count')

                fig = px.bar(
                    chart_data, 
                    x='Sentiment', y='Count', color='Sentiment',
                    text='Count',
                    color_discrete_map={'POSITIVE': '#00875A', 'NEGATIVE': '#DE350B'},
                    height=300
                )
                fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)
                
                avg_conf = filtered_df['Confidence'].mean()
                st.metric("Avg. Model Confidence", f"{avg_conf:.1%}")

                st.subheader("Key Keywords")
                text_combined = " ".join(review for review in filtered_df['text'])
                if text_combined.strip():
                    wc = WordCloud(background_color="white", width=400, height=200).generate(text_combined)
                    fig_wc, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig_wc)

            with table_col:
                st.subheader("Review Details")
                
                display_df = filtered_df[['date', 'Sentiment', 'Confidence', 'text']].copy()
                display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

                def sentiment_color(val):
                    bg = '#00875A' if val == 'POSITIVE' else '#DE350B'
                    return f'background-color: {bg}; color: white; font-weight: bold; border-radius: 4px;'

                st.dataframe(
                    display_df.style.applymap(sentiment_color, subset=['Sentiment']),
                    use_container_width=True, 
                    hide_index=True,
                    height=600 
                )
        else:
            st.warning(f"No reviews found for {selected_month} 2023.")

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.caption("Final Submission | Python & Streamlit")import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AI E-Commerce Insights",
    page_icon="🤖",
    layout="wide"
)

# --- 2. DATA LOADING ---
@st.cache_data
def fetch_data():
    """Load pre-processed JSON data from the data folder."""
    # Build relative path to look inside the /data folder
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'data', 'processed_data.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df_p = pd.DataFrame(data['products'])
        df_t = pd.DataFrame(data['testimonials'])
        df_r = pd.DataFrame(data['reviews'])
        
        # Clean Data Types
        df_p['price'] = pd.to_numeric(df_p['price'].replace(r'[^\d.]', '', regex=True), errors='coerce')
        df_r['date'] = pd.to_datetime(df_r['date'])
        
        if 'Confidence' in df_r.columns:
            df_r['Confidence'] = pd.to_numeric(df_r['Confidence'], errors='coerce')
        
        return df_p, df_t, df_r
    except Exception as e:
        st.error(f"Error loading 'data/processed_data.json': {e}")
        return None, None, None

df_prod, df_testi, df_rev = fetch_data()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Products", "Testimonials", "Reviews"])

if df_prod is not None:

    # --- 4. PAGE: PRODUCTS ---
    if page == "Products":
        st.header("📦 Product Catalog")
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
                    st.markdown(f"👤 **Customer #{row['id']}**")
                    st.write(f"_{row['text']}_")
                    st.caption(f"Rating: {'⭐' * int(row['rating'])}")

    # --- 6. PAGE: REVIEWS ---
    elif page == "Reviews":
        st.header("⭐ 2023 Monthly Sentiment Insights")
        
        months_list = ["January", "February", "March", "April", "May", "June", 
                       "July", "August", "September", "October", "November", "December"]
        
        # CHANGED: select_slider replaced with selectbox (Dropdown)
        selected_month = st.selectbox("Select Month (2023):", options=months_list, index=4)

        # STRICT FILTER: Ensure we only show 2023 data for the chosen month
        filtered_df = df_rev[
            (df_rev['date'].dt.year == 2023) & 
            (df_rev['date'].dt.month_name() == selected_month)
        ].copy()

        st.divider()

        if not filtered_df.empty:
            chart_col, table_col = st.columns([1, 2], gap="large")

            with chart_col:
                st.subheader("Sentiment Distribution")
                chart_data = filtered_df.groupby('Sentiment').size().reset_index(name='Count')

                fig = px.bar(
                    chart_data, 
                    x='Sentiment', y='Count', color='Sentiment',
                    text='Count',
                    color_discrete_map={'POSITIVE': '#00875A', 'NEGATIVE': '#DE350B'},
                    height=300
                )
                fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)
                
                avg_conf = filtered_df['Confidence'].mean()
                st.metric("Avg. Model Confidence", f"{avg_conf:.1%}")

                st.subheader("Key Keywords")
                text_combined = " ".join(review for review in filtered_df['text'])
                if text_combined.strip():
                    wc = WordCloud(background_color="white", width=400, height=200).generate(text_combined)
                    fig_wc, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig_wc)

            with table_col:
                st.subheader("Review Details")
                
                display_df = filtered_df[['date', 'Sentiment', 'Confidence', 'text']].copy()
                display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')

                def sentiment_color(val):
                    bg = '#00875A' if val == 'POSITIVE' else '#DE350B'
                    return f'background-color: {bg}; color: white; font-weight: bold; border-radius: 4px;'

                st.dataframe(
                    display_df.style.applymap(sentiment_color, subset=['Sentiment']),
                    use_container_width=True, 
                    hide_index=True,
                    height=600 
                )
        else:
            st.warning(f"No reviews found for {selected_month} 2023.")

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.caption("Final Submission | Python & Streamlit")
