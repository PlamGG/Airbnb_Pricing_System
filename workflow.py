import pandas as pd
from data_loader import load_airbnb_data
from typing import TypedDict
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import os
import json
import re
from langgraph.graph import StateGraph, END 

# ตั้งค่า environment สำหรับ Ollama
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434"

# โหลด LLaMA 3
try:
    custom_llm = Ollama(model="llama3:8b", base_url="http://localhost:11434", temperature=0.0)
    print("✅ Ollama server detected and running!")
    print("🎯 LLaMA 3 via Ollama initialized successfully!")
except Exception as e:
    print(f"❌ Error loading Ollama model: {str(e)}")
    custom_llm = None

# ฟังก์ชันโหลด reviews.csv
def load_reviews(file_path="reviews.csv"):
    try:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'utf-8-sig']
        for encoding in encodings:
            try:
                df_reviews = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print(f"Successfully read reviews.csv with encoding: {encoding}")
                return df_reviews
            except UnicodeDecodeError:
                continue
        print(f"❌ Failed to read reviews.csv due to encoding issues")
        return None
    except Exception as e:
        print(f"❌ Error loading reviews.csv: {str(e)}")
        return None

# กำหนด State สำหรับ LangGraph
class PricingState(TypedDict):
    neighbourhood: str
    room_type: str
    rating: float
    df: pd.DataFrame
    df_reviews: pd.DataFrame
    avg_price: float
    avg_rating: float
    sentiment: str
    raw_sentiment_output: str
    amenities_factor: float
    recommended_price: float

# Node สำหรับคำนวณ avg_price
def market_node(state: PricingState):
    df = state["df"]
    filtered_df = df[(df["neighbourhood_cleansed"] == state["neighbourhood"]) & 
                    (df["room_type"] == state["room_type"])]
    avg_price = filtered_df["price"].mean() if not filtered_df.empty else 1000.0  # Default price
    return {"avg_price": avg_price}

# Node สำหรับคำนวณ avg_rating และ sentiment ด้วย LLM
def sentiment_analysis_node(state: PricingState):
    df = state["df"]
    df_reviews = state["df_reviews"]
    avg_rating = df[df["neighbourhood_cleansed"] == state["neighbourhood"]]["review_scores_rating"].mean()
    
    # Log จำนวนข้อมูลที่ไม่ null
    description_count = df[df["neighbourhood_cleansed"] == state["neighbourhood"]]["description"].dropna().shape[0] if "description" in df.columns else 0
    name_count = df[df["neighbourhood_cleansed"] == state["neighbourhood"]]["name"].dropna().shape[0] if "name" in df.columns else 0
    overview_count = df[df["neighbourhood_cleansed"] == state["neighbourhood"]]["neighborhood_overview"].dropna().shape[0] if "neighborhood_overview" in df.columns else 0
    amenities_count = df[df["neighbourhood_cleansed"] == state["neighbourhood"]]["amenities"].dropna().shape[0] if "amenities" in df.columns else 0
    print(f"📊 Description count (non-null) for {state['neighbourhood']}: {description_count}")
    print(f"📊 Name count (non-null) for {state['neighbourhood']}: {name_count}")
    print(f"📊 Neighborhood_overview count (non-null) for {state['neighbourhood']}: {overview_count}")
    print(f"📊 Amenities count (non-null) for {state['neighbourhood']}: {amenities_count}")
    
    # ใช้ reviews.csv ถ้ามี
    if df_reviews is not None:
        listing_ids = df[df["neighbourhood_cleansed"] == state["neighbourhood"]]["id"].tolist()
        reviews = df_reviews[df_reviews["listing_id"].isin(listing_ids)]["comments"].dropna().tolist()[:5]  # จำกัด 5 รีวิว
        if reviews and custom_llm:
            prompt = PromptTemplate(
                input_variables=["reviews"],
                template="Analyze the sentiment of these Airbnb reviews: {reviews}. Return one word: 'positive', 'neutral', or 'negative'."
            )
            try:
                chain = prompt | custom_llm
                raw_sentiment = chain.invoke({"reviews": " ".join([str(r) for r in reviews])}).strip().lower()
                print(f"📝 LLM raw sentiment output: {raw_sentiment}")
                valid_sentiments = ["positive", "neutral", "negative"]
                sentiment = raw_sentiment if raw_sentiment in valid_sentiments else ("positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown")
                print(f"📝 Processed sentiment: {sentiment}")
                return {"avg_rating": avg_rating, "sentiment": sentiment, "raw_sentiment_output": raw_sentiment}
            except Exception as e:
                print(f"❌ LLM error: {str(e)}, using rating-based sentiment")
                sentiment = "positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown"
                return {"avg_rating": avg_rating, "sentiment": sentiment, "raw_sentiment_output": "LLM error"}
    
    # Fallback ถ้าไม่มี reviews.csv
    print("⚠️ No reviews.csv, falling back to description/name/amenities")
    text_column = None
    descriptions = []
    if "description" in df.columns and description_count > 0:
        text_column = "description"
        descriptions = df[df["neighbourhood_cleansed"] == state["neighbourhood"]]["description"].dropna().tolist()
    elif "name" in df.columns and name_count > 0:
        text_column = "name"
        descriptions = df[df["neighbourhood_cleansed"] == state["neighbourhood"]]["name"].dropna().tolist()
    elif "amenities" in df.columns and amenities_count > 0:
        text_column = "amenities"
        descriptions = df[df["neighbourhood_cleansed"] == state["neighbourhood"]]["amenities"].dropna().apply(lambda x: " ".join(json.loads(x) if isinstance(x, str) else x)).tolist()
    
    if not descriptions or not custom_llm:
        print(f"⚠️ No valid text or LLM, using rating-based sentiment")
        sentiment = "positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown"
        return {"avg_rating": avg_rating, "sentiment": sentiment, "raw_sentiment_output": "No valid text or LLM"}
    
    prompt = PromptTemplate(
        input_variables=["descriptions", "column"],
        template="Analyze the overall sentiment of these Airbnb {column} texts: {descriptions}. Return only one word: 'positive', 'neutral', or 'negative'."
    )
    try:
        chain = prompt | custom_llm
        raw_sentiment = chain.invoke({"descriptions": " ".join([str(d) for d in descriptions]), "column": text_column}).strip().lower()
        print(f"📝 LLM raw sentiment output: {raw_sentiment}")
        valid_sentiments = ["positive", "neutral", "negative"]
        sentiment = raw_sentiment if raw_sentiment in valid_sentiments else ("positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown")
        print(f"📝 Processed sentiment: {sentiment}")
        return {"avg_rating": avg_rating, "sentiment": sentiment, "raw_sentiment_output": raw_sentiment}
    except Exception as e:
        print(f"❌ LLM error: {str(e)}, using rating-based sentiment")
        sentiment = "positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown"
        return {"avg_rating": avg_rating, "sentiment": sentiment, "raw_sentiment_output": "LLM error"}

# Node สำหรับคำนวณ amenities_factor
def amenities_node(state: PricingState):
    df = state["df"]
    filtered_df = df[(df["neighbourhood_cleansed"] == state["neighbourhood"]) & 
                    (df["room_type"] == state["room_type"])]
    amenities_factor = 1.0
    if "amenities" in filtered_df.columns:
        amenities_list = filtered_df["amenities"].dropna().tolist()
        print(f"📊 Amenities found for {state['neighbourhood']}: {amenities_list[:2]}")
        for amenities in amenities_list:
            try:
                amenities_json = json.loads(amenities) if isinstance(amenities, str) else amenities
                if any(item.lower() in str(amenity).lower() for amenity in amenities_json for item in ["wi-fi", "wifi", "pool", "air conditioning", "tv", "kitchen"]):
                    amenities_factor = 1.1
                    break
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON in amenities: {amenities}")
                continue
    return {"amenities_factor": amenities_factor}

# Node สำหรับคำนวณ recommended_price
def pricing_node(state: PricingState):
    avg_price = state["avg_price"]
    avg_rating = state["avg_rating"]
    rating = state["rating"]
    amenities_factor = state["amenities_factor"]
    if pd.isna(avg_price) or pd.isna(avg_rating):
        return {"recommended_price": 1000.0}  # Default price
    recommended_price = avg_price * (1 + (rating - 3) * 0.1) * (1.1 if state["sentiment"] == "positive" else 1.0) * amenities_factor
    return {"recommended_price": recommended_price}

# สร้าง LangGraph workflow
workflow = StateGraph(PricingState)
workflow.add_node("market", market_node)
workflow.add_node("sentiment_analysis", sentiment_analysis_node)
workflow.add_node("amenities", amenities_node)
workflow.add_node("pricing", pricing_node)
workflow.set_entry_point("market")
workflow.add_edge("market", "sentiment_analysis")
workflow.add_edge("sentiment_analysis", "amenities")
workflow.add_edge("amenities", "pricing")
workflow.add_edge("pricing", END)
app = workflow.compile()

# ฟังก์ชันรัน workflow
def run_workflow(neighbourhood, room_type, rating, file_path="listings_small.csv", reviews_path="reviews.csv"):
    df, _ = load_airbnb_data(file_path)
    df_reviews = load_reviews(reviews_path)
    
    if df is None or df.empty:
        print("❌ Failed to load dataset")
        return {"error": "Failed to load dataset"}
    
    # สร้าง initial state
    initial_state = {
        "neighbourhood": neighbourhood,
        "room_type": room_type,
        "rating": rating,
        "df": df,
        "df_reviews": df_reviews,
        "avg_price": float('nan'),
        "avg_rating": float('nan'),
        "sentiment": "unknown",
        "raw_sentiment_output": "unknown",
        "amenities_factor": 1.0,
        "recommended_price": float('nan')
    }
    
    # รัน workflow
    try:
        result = app.invoke(initial_state)
        print("\n=== WORKFLOW RESULT ===")
        print(f"Average Price: {result['avg_price']:.2f} THB")
        print(f"Average Rating: {result['avg_rating']:.2f}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Raw Sentiment Output: {result['raw_sentiment_output']}")
        print(f"Amenities Factor: {result['amenities_factor']:.2f}")
        print(f"Recommended Price: {result['recommended_price']:.2f} THB")
        return result
    except Exception as e:
        print(f"❌ Workflow execution failed: {str(e)}")
        return {"error": f"Workflow execution failed: {str(e)}"}

# Main
if __name__ == "__main__":
    test_room_type = "Entire home/apt"
    test_rating = 4.5
    test_neighbourhoods = ["Bang Kapi"]
    for neighbourhood in test_neighbourhoods:
        print(f"\n=== Testing {neighbourhood} ===")
        result = run_workflow(neighbourhood, test_room_type, test_rating)