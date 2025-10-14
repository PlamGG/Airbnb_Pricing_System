import pandas as pd
from data_loader import load_airbnb_data
from typing import TypedDict
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import os
import json
import re
from langgraph.graph import StateGraph, END


os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434"

try:
    custom_llm = Ollama(model="llama3:8b", base_url="http://localhost:11434", temperature=0.0)
    print("Ollama server detected and running!")
    print("LLaMA 3 via Ollama initialized successfully!")
except Exception as e:
    print(f"Error loading Ollama model: {str(e)}")
    custom_llm = None


def load_reviews(file_path="reviews.csv"):
    """Load reviews CSV file with multiple encoding support."""
    try:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'utf-8-sig']
        for encoding in encodings:
            try:
                df_reviews = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print(f"Successfully read reviews file with encoding: {encoding}")
                return df_reviews
            except UnicodeDecodeError:
                continue
        print(f"Failed to read reviews file due to encoding issues")
        return None
    except Exception as e:
        print(f"Error loading reviews file: {str(e)}")
        return None


class PricingState(TypedDict):
    """State definition for LangGraph pricing workflow."""
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


def market_node(state: PricingState):
    """Calculate average price for the given neighbourhood and room type."""
    df = state["df"]
    filtered_df = df[(df["neighbourhood_cleansed"] == state["neighbourhood"]) & 
                    (df["room_type"] == state["room_type"])]
    avg_price = filtered_df["price"].mean() if not filtered_df.empty else 1000.0
    return {"avg_price": avg_price}


def sentiment_analysis_node(state: PricingState):
    """Analyze sentiment using LLM from reviews or listing descriptions."""
    df = state["df"]
    df_reviews = state["df_reviews"]
    avg_rating = df[df["neighbourhood_cleansed"] == state["neighbourhood"]]["review_scores_rating"].mean()
    
    if df_reviews is not None:
        listing_ids = df[df["neighbourhood_cleansed"] == state["neighbourhood"]]["id"].tolist()
        reviews = df_reviews[df_reviews["listing_id"].isin(listing_ids)]["comments"].dropna().tolist()[:5]
        
        if reviews and custom_llm:
            prompt = PromptTemplate(
                input_variables=["reviews"],
                template="Analyze the sentiment of these Airbnb reviews: {reviews}. Return one word: 'positive', 'neutral', or 'negative'."
            )
            try:
                chain = prompt | custom_llm
                raw_sentiment = chain.invoke({"reviews": " ".join([str(r) for r in reviews])}).strip().lower()
                valid_sentiments = ["positive", "neutral", "negative"]
                sentiment = raw_sentiment if raw_sentiment in valid_sentiments else ("positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown")
                return {"avg_rating": avg_rating, "sentiment": sentiment, "raw_sentiment_output": raw_sentiment}
            except Exception as e:
                print(f"LLM error: {str(e)}, using rating-based sentiment")
                sentiment = "positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown"
                return {"avg_rating": avg_rating, "sentiment": sentiment, "raw_sentiment_output": "LLM error"}
    
    text_column = None
    descriptions = []
    
    for col in ["description", "name", "amenities"]:
        if col in df.columns:
            col_data = df[df["neighbourhood_cleansed"] == state["neighbourhood"]][col].dropna()
            if not col_data.empty:
                text_column = col
                if col == "amenities":
                    descriptions = col_data.apply(lambda x: " ".join(json.loads(x) if isinstance(x, str) else x)).tolist()
                else:
                    descriptions = col_data.tolist()
                break
    
    if not descriptions or not custom_llm:
        sentiment = "positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown"
        return {"avg_rating": avg_rating, "sentiment": sentiment, "raw_sentiment_output": "No valid text or LLM"}
    
    prompt = PromptTemplate(
        input_variables=["descriptions", "column"],
        template="Analyze the overall sentiment of these Airbnb {column} texts: {descriptions}. Return only one word: 'positive', 'neutral', or 'negative'."
    )
    
    try:
        chain = prompt | custom_llm
        raw_sentiment = chain.invoke({"descriptions": " ".join([str(d) for d in descriptions]), "column": text_column}).strip().lower()
        valid_sentiments = ["positive", "neutral", "negative"]
        sentiment = raw_sentiment if raw_sentiment in valid_sentiments else ("positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown")
        return {"avg_rating": avg_rating, "sentiment": sentiment, "raw_sentiment_output": raw_sentiment}
    except Exception as e:
        print(f"LLM error: {str(e)}, using rating-based sentiment")
        sentiment = "positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown"
        return {"avg_rating": avg_rating, "sentiment": sentiment, "raw_sentiment_output": "LLM error"}


def amenities_node(state: PricingState):
    """Calculate amenities factor based on available amenities."""
    df = state["df"]
    filtered_df = df[(df["neighbourhood_cleansed"] == state["neighbourhood"]) & 
                    (df["room_type"] == state["room_type"])]
    amenities_factor = 1.0
    
    if "amenities" in filtered_df.columns:
        amenities_list = filtered_df["amenities"].dropna().tolist()
        for amenities in amenities_list:
            try:
                amenities_json = json.loads(amenities) if isinstance(amenities, str) else amenities
                if any(item.lower() in str(amenity).lower() for amenity in amenities_json 
                       for item in ["wi-fi", "wifi", "pool", "air conditioning", "tv", "kitchen"]):
                    amenities_factor = 1.1
                    break
            except json.JSONDecodeError:
                continue
    
    return {"amenities_factor": amenities_factor}


def pricing_node(state: PricingState):
    """Calculate recommended price based on market data, rating, sentiment, and amenities."""
    avg_price = state["avg_price"]
    avg_rating = state["avg_rating"]
    rating = state["rating"]
    amenities_factor = state["amenities_factor"]
    
    if pd.isna(avg_price) or pd.isna(avg_rating):
        return {"recommended_price": 1000.0}
    
    recommended_price = avg_price * (1 + (rating - 3) * 0.1) * (1.1 if state["sentiment"] == "positive" else 1.0) * amenities_factor
    return {"recommended_price": recommended_price}


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


def run_workflow(neighbourhood, room_type, rating, file_path="listings_small.csv", reviews_path="reviews.csv"):
    """Execute the pricing workflow for a given neighbourhood and room type."""
    df, _ = load_airbnb_data(file_path)
    df_reviews = load_reviews(reviews_path)
    
    if df is None or df.empty:
        print("Failed to load dataset")
        return {"error": "Failed to load dataset"}
    
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
        print(f"Workflow execution failed: {str(e)}")
        return {"error": f"Workflow execution failed: {str(e)}"}


if __name__ == "__main__":
    test_room_type = "Entire home/apt"
    test_rating = 4.5
    test_neighbourhoods = ["Bang Kapi"]
    
    for neighbourhood in test_neighbourhoods:
        print(f"\n=== Testing {neighbourhood} ===")
        result = run_workflow(neighbourhood, test_room_type, test_rating)