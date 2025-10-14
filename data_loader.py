import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter


def load_airbnb_data(file_path, reviews_path="reviews.csv"):
    """
    Load Airbnb listings and reviews data from CSV files.
    
    Args:
        file_path (str): Path to the listings CSV file
        reviews_path (str): Path to the reviews CSV file (default: "reviews.csv")
    
    Returns:
        tuple: (df, documents, df_reviews) - DataFrames and LangChain documents
    """
    try:
        if not pd.io.common.file_exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found. Please ensure the listings file is in the project folder.")
        
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'utf-8-sig']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print(f"Successfully read listings file with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("Failed to read listings file due to encoding issues. Tried encodings: " + ", ".join(encodings))
        
        if df.empty:
            raise ValueError("File is empty or contains no data.")
        
        print(f"Available columns in listings: {df.columns.tolist()}")
        
        def clean_price(price):
            try:
                price_str = str(price).strip()
                cleaned = price_str.replace('$', '').replace(',', '')
                if cleaned and cleaned.replace('.', '', 1).replace('-', '', 1).isdigit():
                    return float(cleaned)
                return float('nan')
            except:
                return float('nan')
        
        df['price'] = df['price'].apply(clean_price)
        
        df_reviews = None
        if pd.io.common.file_exists(reviews_path):
            for encoding in encodings:
                try:
                    df_reviews = pd.read_csv(reviews_path, encoding=encoding, low_memory=False)
                    print(f"Successfully read reviews file with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df_reviews is None:
                print(f"Warning: Failed to read reviews file due to encoding issues")
            elif df_reviews.empty:
                print(f"Warning: Reviews file is empty")
                df_reviews = None
            else:
                print(f"Available columns in reviews: {df_reviews.columns.tolist()}")
        else:
            print(f"Warning: Reviews file not found at {reviews_path}")
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = [
            f"Neighbourhood: {row['neighbourhood_cleansed']}, Room Type: {row['room_type']}, Price: {row['price']}, Rating: {row['review_scores_rating']}" 
            for _, row in df.iterrows() 
            if pd.notna(row['neighbourhood_cleansed']) and pd.notna(row['room_type']) and pd.notna(row['price']) and pd.notna(row['review_scores_rating'])
        ]
        documents = text_splitter.create_documents(texts)
        
        return df, documents, df_reviews
    
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        return None, None, None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty or corrupted.")
        return None, None, None
    except pd.errors.ParserError:
        print(f"Error: File '{file_path}' has invalid format or corrupted data.")
        return None, None, None
    except ValueError as e:
        print(f"Error: {str(e)}")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error loading dataset: {str(e)}")
        return None, None, None


if __name__ == "__main__":
    file_path = "listings_small.csv"
    df, docs, df_reviews = load_airbnb_data(file_path)
    
    if df is not None:
        print("\nDataset loaded successfully")
        print(f"Loaded {len(docs)} documents")
        print(f"\nAvailable neighbourhoods: {len(df['neighbourhood_cleansed'].unique())} unique values")
        print(f"Available room types: {df['room_type'].unique().tolist()}")
        
        if df_reviews is not None:
            print(f"\nReviews dataset loaded: {len(df_reviews)} reviews")
    else:
        print("Failed to load dataset. Please check the file and try again.")