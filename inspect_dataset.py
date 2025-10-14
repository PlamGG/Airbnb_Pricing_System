import pandas as pd
from data_loader import load_airbnb_data
import json


def inspect_dataset(file_path="listings_small.csv"):
    """
    Inspect and analyze the Airbnb dataset structure and content.
    
    This function performs comprehensive data inspection including:
    - Displaying all available columns
    - Checking for missing data in key columns
    - Showing sample data from different neighbourhoods
    - Validating amenities data format
    
    Args:
        file_path (str): Path to the listings CSV file
    """
    # Load the dataset using the data_loader module
    df, _ = load_airbnb_data(file_path)
    
    if df is None or df.empty:
        print("Failed to load dataset")
        return
    
    # Display all column names in the dataset
    print("\n=== All columns in dataset ===")
    print(df.columns.tolist())
    
    # Define columns that are important for analysis
    columns_of_interest = ["name", "description", "neighborhood_overview", "amenities", 
                          "neighbourhood_cleansed", "room_type", "price", 
                          "review_scores_rating"]
    
    # Check which columns are available and which are missing
    available_columns = [col for col in columns_of_interest if col in df.columns]
    missing_columns = [col for col in columns_of_interest if col not in df.columns]
    print("\n=== Available columns from columns_of_interest ===")
    print(available_columns)
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    # Count non-null values for each available column
    print("\n=== Non-null counts for available columns ===")
    non_null_counts = df[available_columns].notnull().sum()
    for col, count in non_null_counts.items():
        print(f"{col}: {count} non-null rows")
    
    # Display total number of rows in the dataset
    print(f"\nTotal rows in dataset: {len(df)}")
    
    # Show sample data for specific neighbourhoods
    neighbourhoods = ["Ratchathewi", "Khlong Toei", "Sukhumvit"]
    print("\n=== Sample row for each neighbourhood ===")
    for neighbourhood in neighbourhoods:
        print(f"\n--- {neighbourhood} ---")
        filtered_df = df[df["neighbourhood_cleansed"] == neighbourhood]
        if not filtered_df.empty:
            # Display first row for this neighbourhood
            sample_row = filtered_df[available_columns].iloc[0]
            for col, value in sample_row.items():
                print(f"{col}: {value}")
        else:
            print(f"No data available for {neighbourhood}")
    
    # Display first 5 rows of the dataset with relevant columns
    print("\n=== Sample 5 rows from dataset ===")
    print(df[available_columns].head().to_string())
    
    # Inspect amenities column format and validate JSON structure
    print("\n=== Inspecting amenities column ===")
    if "amenities" in df.columns:
        amenities_sample = df["amenities"].dropna().head(5).tolist()
        for i, amenities in enumerate(amenities_sample, 1):
            try:
                # Try to parse amenities as JSON
                amenities_json = json.loads(amenities) if isinstance(amenities, str) else amenities
                print(f"Amenities sample {i}: {amenities_json}")
            except json.JSONDecodeError:
                print(f"Amenities sample {i} (invalid JSON): {amenities}")
    else:
        print("No amenities column in dataset")


if __name__ == "__main__":
    inspect_dataset()