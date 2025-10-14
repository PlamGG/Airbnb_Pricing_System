import pandas as pd
import numpy as np
import os
import json
import re
from crewai import Agent, Task, Crew
from crewai import LLM
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from data_loader import load_airbnb_data


# ====== ตั้งค่า environment สำหรับ Ollama ======
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434"

def initialize_llm():
    """Initialize LLaMA 3 model for CrewAI and LangChain."""
    try:
        crew_llm = LLM(model="ollama/llama3:8b", base_url="http://localhost:11434", temperature=0.0)
        langchain_llm = Ollama(model="llama3:8b", base_url="http://localhost:11434", temperature=0.0)
        print("✅ Ollama server detected and running!")
        print("🎯 LLaMA 3 initialized successfully!")
        return crew_llm, langchain_llm
    except Exception as e:
        print(f"❌ Error loading Ollama model: {str(e)}")
        return None, None

crew_llm, langchain_llm = initialize_llm()

# ====== สร้าง Agents ======
market_research_agent = Agent(
    role="Market Research Analyst",
    goal="Analyze average prices in a given neighbourhood",
    backstory="Expert in analyzing Airbnb market trends in Thailand.",
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=crew_llm
)

pricing_agent = Agent(
    role="Pricing Specialist",
    goal="Calculate recommended price based on market data, rating, and amenities",
    backstory="Specialist in dynamic pricing for Airbnb hosts.",
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=crew_llm
)

sentiment_agent = Agent(
    role="Sentiment Analyst",
    goal="Analyze customer reviews to determine sentiment",
    backstory="Skilled in NLP and sentiment analysis using review text.",
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=crew_llm
)

# ====== ฟังก์ชันวิเคราะห์ amenities_factor ======
amenities_cache = {}

def calculate_amenities_factor(df, neighbourhood, room_type):
    """Calculate amenities factor based on listings' amenities."""
    cache_key = f"{neighbourhood}_{room_type}"
    if cache_key in amenities_cache:
        print(f"DEBUG: Using cached amenities_factor for {cache_key}")
        return amenities_cache[cache_key]
    
    print(f"DEBUG: Calculating amenities_factor for {cache_key}")
    filtered_df = df[(df["neighbourhood_cleansed"] == neighbourhood) & 
                    (df["room_type"] == room_type)]
    amenities_factor = 1.0
    if "amenities" in filtered_df.columns and not filtered_df.empty:
        amenities_list = filtered_df["amenities"].dropna().tolist()
        print(f"📊 Amenities found for {neighbourhood}: {amenities_list[:2]}")
        for amenities in amenities_list:
            try:
                amenities_json = json.loads(amenities) if isinstance(amenities, str) else amenities
                if any(item.lower() in str(amenity).lower() for amenity in amenities_json 
                       for item in ["wi-fi", "wifi", "pool", "air conditioning", "tv", "kitchen"]):
                    amenities_factor = 1.1
                    break
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON in amenities: {amenities}")
                continue
    amenities_cache[cache_key] = amenities_factor
    return amenities_factor

# ====== ฟังก์ชันวิเคราะห์ sentiment จาก reviews ======
def analyze_reviews(df, df_reviews, neighbourhood):
    """Analyze sentiment of reviews for listings in a neighbourhood."""
    listing_ids = df[df["neighbourhood_cleansed"] == neighbourhood]["id"].tolist()
    reviews = df_reviews[df_reviews["listing_id"].isin(listing_ids)]["comments"].dropna().tolist()[:5]
    avg_rating = df[df["neighbourhood_cleansed"] == neighbourhood]["review_scores_rating"].dropna().mean()
    
    if not reviews or not langchain_llm:
        sentiment = "positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown"
        confidence = 0.8 if avg_rating >= 4 else 0.5
        return {"sentiment": sentiment, "confidence": confidence}

    prompt = PromptTemplate(
        input_variables=["reviews"],
        template="Analyze the sentiment of these Airbnb reviews: {reviews}. Return a JSON object with 'sentiment' ('positive', 'neutral', or 'negative') and 'confidence' (0.0 to 1.0). Example: {{\"sentiment\": \"positive\", \"confidence\": 0.87}}"
    )
    try:
        chain = prompt | langchain_llm
        raw_output = chain.invoke({"reviews": " ".join([str(r) for r in reviews])}).strip()
        print(f"📝 LLM raw sentiment output: {raw_output}")
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            result = json.loads(json_match.group(0))
            sentiment = result.get("sentiment", "neutral").lower()
            confidence = float(result.get("confidence", 0.5))
            valid_sentiments = ["positive", "neutral", "negative"]
            if sentiment not in valid_sentiments:
                sentiment = "neutral"
                confidence = 0.5
            return {"sentiment": sentiment, "confidence": confidence}
        else:
            print(f"⚠️ No JSON found in LLM output: {raw_output}")
            sentiment = "positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown"
            confidence = 0.8 if avg_rating >= 4 else 0.5
            return {"sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        print(f"❌ LLM error in review analysis: {str(e)}")
        sentiment = "positive" if avg_rating >= 4 else "neutral" if not pd.isna(avg_rating) else "unknown"
        confidence = 0.8 if avg_rating >= 4 else 0.5
        return {"sentiment": sentiment, "confidence": confidence}

# ====== สร้าง Tasks ======
def create_tasks(neighbourhood, room_type, rating, df, df_reviews):
    """Create tasks for market analysis, sentiment analysis, and pricing."""
    if df is None or df.empty:
        print("❌ Dataset is None or empty")
        return None

    print(f"DEBUG: Creating tasks for {neighbourhood}, {room_type}, rating {rating}")
    market_task = Task(
        description=f"Analyze average price for {room_type} in {neighbourhood}. Return a dictionary with 'avg_price' and 'amenities_factor'.",
        agent=market_research_agent,
        expected_output="A dictionary with average price and amenities factor",
        callback=lambda output: {
            "avg_price": df[(df["neighbourhood_cleansed"] == neighbourhood) & 
                           (df["room_type"] == room_type)]["price"].dropna().mean()
            if not df[(df["neighbourhood_cleansed"] == neighbourhood) & 
                     (df["room_type"] == room_type)].empty
            else 1000.0,
            "amenities_factor": calculate_amenities_factor(df, neighbourhood, room_type)
        }
    )

    sentiment_task = Task(
        description=f"Analyze sentiment for listings in {neighbourhood} based on reviews. Return a dictionary with 'avg_rating', 'sentiment', and 'confidence'.",
        agent=sentiment_agent,
        expected_output="A dictionary with average rating, sentiment, and confidence",
        callback=lambda output: {
            "avg_rating": df[df["neighbourhood_cleansed"] == neighbourhood]["review_scores_rating"].dropna().mean()
            if not df[df["neighbourhood_cleansed"] == neighbourhood].empty
            else float('nan'),
            **analyze_reviews(df, df_reviews, neighbourhood)
        }
    )

    pricing_task = Task(
        description=f"Calculate recommended price for a {room_type} in {neighbourhood} with rating {rating}. Use sentiment and amenities_factor from tasks. Return a dictionary with 'recommended_price'.",
        agent=pricing_agent,
        expected_output="A dictionary with recommended price",
        context=[market_task, sentiment_task],
        callback=lambda output, df=df, neighbourhood=neighbourhood, room_type=room_type, rating=rating, 
                       market_task=market_task, sentiment_task=sentiment_task, market_output_cache=None: {
            "recommended_price": (
                market_output := market_output_cache if market_output_cache is not None else market_task.callback(None),
                sentiment_output := sentiment_task.callback(None),
                avg_price := market_output.get("avg_price", 1000.0),
                rating_factor := (1 + (rating - 3) * 0.1),
                sentiment_factor := (1.1 if sentiment_output.get("sentiment", "unknown").lower() == "positive" else 1.0),
                amenities_factor := market_output.get("amenities_factor", 1.0),
                recommended_price := avg_price * rating_factor * sentiment_factor * amenities_factor,
                print(f"DEBUG: Pricing Task Inputs - avg_price: {avg_price}, rating_factor: {rating_factor}, "
                      f"sentiment_factor: {sentiment_factor}, amenities_factor: {amenities_factor}"),
                print(f"DEBUG: Pricing Task Source - Using market_output_cache: {amenities_factor}"),
                recommended_price
            )[-1]
        }
    )

    return [market_task, sentiment_task, pricing_task]

# ====== รัน Crew ======
def run_crew(neighbourhood, room_type, rating, df, df_reviews):
    """Run CrewAI to analyze market, sentiment, and calculate recommended price."""
    if crew_llm is None:
        return {"error": "LLM not loaded properly"}

    print(f"DEBUG: Starting run_crew for {neighbourhood}, {room_type}, rating {rating}")
    tasks = create_tasks(neighbourhood, room_type, rating, df, df_reviews)
    if tasks is None:
        return {"error": "Dataset not loaded properly or is empty"}

    crew = Crew(
        agents=[market_research_agent, sentiment_agent, pricing_agent],
        tasks=tasks,
        verbose=True,
        process="sequential"
    )

    try:
        print(f"DEBUG: Starting crew.kickoff()")
        crew.kickoff()
        print(f"DEBUG: Completed crew.kickoff()")

        results = {}
        market_output_cache = None
        for task in crew.tasks:
            try:
                if task.description.startswith("Analyze average price"):
                    market_output_cache = task.callback(None)
                    task_output = market_output_cache
                else:
                    task_output = task.callback(None) if task.callback else {}
                results[task.description] = task_output
                print(f"Task '{task.description}' output: {task_output}")
            except Exception as e:
                print(f"⚠️ Task '{task.description}' failed: {str(e)}")
                results[task.description] = task.callback(None) if task.callback else {"error": str(e)}

        final_result = {
            "neighbourhood": neighbourhood,
            "room_type": room_type,
            "rating": rating,
            "avg_price": results[tasks[0].description].get("avg_price", 1000.0),
            "amenities_factor": results[tasks[0].description].get("amenities_factor", 1.0),
            "avg_rating": results[tasks[1].description].get("avg_rating", float('nan')),
            "sentiment": results[tasks[1].description].get("sentiment", "unknown"),
            "confidence": results[tasks[1].description].get("confidence", 0.5),
            "recommended_price": results[tasks[2].description].get("recommended_price", 1000.0)
        }
        print(f"DEBUG: Final result - {final_result}")
        return final_result

    except Exception as e:
        print(f"❌ Crew execution failed: {str(e)}")
        return {"error": f"Crew execution failed: {str(e)}"}

# ====== Main ======
if __name__ == "__main__":
    file_path = "listings_small.csv"
    df, docs, df_reviews = load_airbnb_data(file_path)
    if df is not None:
        print("Dataset loaded successfully:")
        print(df.head())
        print(f"Loaded {len(docs)} documents")
        print("\nAvailable neighbourhoods:", df['neighbourhood_cleansed'].unique().tolist())
        print("Available room types:", df['room_type'].unique().tolist())
        if df_reviews is not None:
            print("Reviews dataset loaded successfully:")
            print(df_reviews.head())

        test_neighbourhood = "Ratchathewi"
        test_room_type = "Entire home/apt"
        test_rating = 4.5
        result = run_crew(test_neighbourhood, test_room_type, test_rating, df, df_reviews)

        if "error" in result:
            print(result["error"])
        else:
            print(f"\n=== FINAL RESULT ===")
            print(f"Average Price: {result.get('avg_price', 1000.0):.2f} THB")
            print(f"Average Rating: {result.get('avg_rating', float('nan')):.2f}")
            print(f"Sentiment: {result.get('sentiment', 'unknown')} ({int(result.get('confidence', 0.5) * 100)}%)")
            print(f"Amenities Factor: {result.get('amenities_factor', 1.0):.2f}")
            print(f"Recommended Price: {result.get('recommended_price', 1000.0):.2f} THB")