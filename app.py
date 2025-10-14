import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import json
from agents import run_crew, load_airbnb_data

# ตั้งค่า Streamlit page
st.set_page_config(page_title="Airbnb Pricing Analysis", layout="wide")

# CSS สำหรับสไตล์ที่สวยงาม
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stSelectbox { margin-bottom: 20px; }
    .metric-card { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 15px; }
    .header { font-size: 36px; font-weight: bold; color: #333; text-align: center; margin-bottom: 20px; }
    .subheader { font-size: 20px; color: #555; margin-top: 10px; }
    .review-box { background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    @media (max-width: 600px) {
        .header { font-size: 24px; }
        .metric-card { padding: 10px; }
    }
    </style>
""", unsafe_allow_html=True)

# ฟังก์ชันวิเคราะห์ amenities ยอดนิยม
def get_top_amenities(df, neighbourhood, room_type, price, rating_threshold=4.5, price_range=0.2):
    """Analyze top amenities for high-rated listings in a price range."""
    # กรองที่พักที่มีคะแนนดีและอยู่ในช่วงราคา
    price_min = price * (1 - price_range)
    price_max = price * (1 + price_range)
    filtered_df = df[
        (df["neighbourhood_cleansed"] == neighbourhood) &
        (df["room_type"] == room_type) &
        (df["review_scores_rating"] >= rating_threshold) &
        (df["price"].between(price_min, price_max))
    ]
    
    if filtered_df.empty:
        return [], f"No high-rated listings found for {neighbourhood}, {room_type} in price range {price_min:.0f}-{price_max:.0f} THB."
    
    # รวบรวม amenities
    amenities_counter = Counter()
    for amenities in filtered_df["amenities"].dropna():
        try:
            amenities_list = json.loads(amenities) if isinstance(amenities, str) else amenities
            amenities_counter.update([str(a).lower() for a in amenities_list])
        except json.JSONDecodeError:
            continue
    
    # เลือก top 5 amenities
    top_amenities = amenities_counter.most_common(5)
    total_listings = len(filtered_df)
    top_amenities = [(amenity, count, (count / total_listings * 100) if total_listings > 0 else 0) 
                     for amenity, count in top_amenities]
    
    return top_amenities, f"Found {total_listings} high-rated listings in price range {price_min:.0f}-{price_max:.0f} THB."

# โหลดข้อมูล
@st.cache_data
def load_data(file_path="listings.csv"):  # เปลี่ยนเป็น dataset เต็ม
    df, docs, df_reviews = load_airbnb_data(file_path)
    if df is None or df_reviews is None:
        st.error("❌ Failed to load dataset or reviews!")
        return None, None, None
    return df, docs, df_reviews

# ฟังก์ชันรันทุกย่าน
@st.cache_data
def run_all_cached(df, df_reviews, room_type, rating):
    results = []
    progress_bar = st.progress(0)
    neighbourhoods = df["neighbourhood_cleansed"].unique()
    for i, neighbourhood in enumerate(neighbourhoods):
        result = run_crew(neighbourhood, room_type, rating, df, df_reviews)
        if "error" not in result:
            results.append(result)
        else:
            st.warning(f"Failed to analyze {neighbourhood}: {result['error']}")
        progress_bar.progress((i + 1) / len(neighbourhoods))
    return results

# UI หลัก
st.markdown("<div class='header'>Airbnb Pricing Analysis</div>", unsafe_allow_html=True)
st.markdown("Analyze pricing and popular amenities for Airbnb listings in Thailand using AI-driven market and sentiment analysis.")

# โหลด dataset
df, docs, df_reviews = load_data()
if df is not None:
    neighbourhoods = df["neighbourhood_cleansed"].unique().tolist()
    room_types = df["room_type"].unique().tolist()

    # Sidebar สำหรับเลือก input
    st.sidebar.header("Analysis Settings")
    analysis_mode = st.sidebar.radio("Analysis Mode", ["Single Neighbourhood", "All Neighbourhoods"])
    neighbourhood = st.sidebar.selectbox("Select Neighbourhood", neighbourhoods) if analysis_mode == "Single Neighbourhood" else None
    room_type = st.sidebar.selectbox("Select Room Type", room_types)
    rating = st.sidebar.slider("Select Rating", 1.0, 5.0, 3.5, 0.1)
    num_reviews = st.sidebar.slider("Number of Reviews to Display", 1, 10, 3)
    analyze_button = st.sidebar.button("Analyze")

    # Placeholder สำหรับผลลัพธ์
    result_placeholder = st.empty()

    if analyze_button:
        with st.spinner("Analyzing..."):
            if analysis_mode == "Single Neighbourhood":
                if neighbourhood not in neighbourhoods:
                    st.error(f"Neighbourhood '{neighbourhood}' not found in dataset.")
                else:
                    result = run_crew(neighbourhood, room_type, rating, df, df_reviews)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        # แสดงผลลัพธ์ใน metric cards
                        st.markdown(f"<div class='subheader'>Results for {neighbourhood} ({room_type})</div>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.metric(label="Average Price", value=f"{result['avg_price']:.2f} THB")
                            filtered_df = df[(df["neighbourhood_cleansed"] == neighbourhood) & (df["room_type"] == room_type)]
                            if len(filtered_df) < 5:
                                st.warning(f"Only {len(filtered_df)} listings found. Results may be unreliable.")
                            if result["avg_price"] == 1000.0:
                                st.warning("Average price is a default value due to insufficient data.")
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.metric(label="Average Rating", value=f"{result['avg_rating']:.2f}")
                            st.metric(label="Sentiment", value=f"{result['sentiment'].capitalize()} ({int(result['confidence'] * 100)}%)")
                            st.markdown("</div>", unsafe_allow_html=True)
                        with col3:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.metric(label="Amenities Factor", value=f"{result['amenities_factor']:.2f}")
                            if result["amenities_factor"] == 1.0:
                                st.warning("No key amenities (e.g., Wi-Fi, pool) detected.")
                            st.metric(label="Recommended Price", value=f"{result['recommended_price']:.2f} THB")
                            st.markdown("</div>", unsafe_allow_html=True)

                        # วิเคราะห์ amenities ยอดนิยม
                        st.markdown("<div class='subheader'>Popular Amenities in High-Rated Listings</div>", unsafe_allow_html=True)
                        top_amenities, message = get_top_amenities(df, neighbourhood, room_type, result["recommended_price"])
                        if top_amenities:
                            amenities_df = pd.DataFrame(top_amenities, columns=["Amenity", "Count", "Percentage"])
                            st.dataframe(amenities_df[["Amenity", "Percentage"]], use_container_width=True)
                            fig_amenities = px.bar(amenities_df, x="Amenity", y="Percentage", 
                                                   title=f"Top Amenities in High-Rated Listings ({message})",
                                                   color="Amenity", color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig_amenities.update_yaxes(title_text="% of Listings")
                            st.plotly_chart(fig_amenities, use_container_width=True)
                        else:
                            st.info(message)

                        # กราฟเปรียบเทียบราคา
                        price_data = pd.DataFrame({
                            "Price Type": ["Average Price", "Recommended Price"],
                            "Price (THB)": [result["avg_price"], result["recommended_price"]]
                        })
                        fig_price = px.bar(price_data, x="Price Type", y="Price (THB)", title="Price Comparison",
                                           color="Price Type", color_discrete_sequence=["#4CAF50", "#FF5733"])
                        fig_price.update_traces(hovertemplate="Price Type: %{x}<br>Price: %{y:.2f} THB")
                        st.plotly_chart(fig_price, use_container_width=True)

                        # กราฟเปรียบเทียบ rating และ confidence
                        rating_data = pd.DataFrame({
                            "Metric": ["Average Rating", "Sentiment Confidence"],
                            "Value": [result["avg_rating"], result["confidence"] * 100]
                        })
                        fig_rating = px.bar(rating_data, x="Metric", y="Value", title="Rating and Sentiment Confidence",
                                            color="Metric", color_discrete_sequence=["#2196F3", "#FFC107"])
                        fig_rating.update_yaxes(title_text="Value (Rating: 0-5, Confidence: %)")
                        st.plotly_chart(fig_rating, use_container_width=True)

                        # แสดงรีวิวตัวอย่างใน expander
                        st.markdown("<div class='subheader'>Sample Reviews Used for Sentiment Analysis</div>", unsafe_allow_html=True)
                        reviews = df_reviews[df_reviews["listing_id"].isin(
                            df[df["neighbourhood_cleansed"] == result["neighbourhood"]]["id"]
                        )]["comments"].dropna().tolist()[:num_reviews]
                        if reviews:
                            with st.expander("View Sample Reviews"):
                                for i, review in enumerate(reviews):
                                    st.markdown(f"<div class='review-box'>Review {i+1}: {review[:200] + '...' if len(review) > 200 else review}</div>", unsafe_allow_html=True)
                        else:
                            st.info("No reviews available for this neighbourhood. Sentiment based on rating.")

            else:
                # รันทุกย่าน
                results = run_all_cached(df, df_reviews, room_type, rating)
                if results:
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values("recommended_price", ascending=False).head(15)  # จำกัด 15 ย่าน
                    st.markdown("<div class='subheader'>Results for All Neighbourhoods</div>", unsafe_allow_html=True)
                    st.dataframe(results_df[["neighbourhood", "avg_price", "avg_rating", "sentiment", 
                                             "confidence", "amenities_factor", "recommended_price"]].round(2),
                                 use_container_width=True)

                    # กราฟเปรียบเทียบราคาทุกย่าน
                    fig = px.bar(results_df, x="neighbourhood", y=["avg_price", "recommended_price"],
                                 title="Price Comparison Across Top 15 Neighbourhoods",
                                 barmode="group", color_discrete_sequence=["#4CAF50", "#FF5733"])
                    fig.update_traces(hovertemplate="Neighbourhood: %{x}<br>Price: %{y:.2f} THB")
                    st.plotly_chart(fig, use_container_width=True)

                    # ปุ่ม export CSV
                    st.download_button(
                        label="Download Results as CSV",
                        data=results_df.round(2).to_csv(index=False),
                        file_name="airbnb_pricing_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No valid results for any neighbourhood.")

else:
    st.error("Failed to load dataset. Please check the file path.")