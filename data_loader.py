import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

# โหลด dataset
def load_airbnb_data(file_path, reviews_path="reviews.csv"):
    try:
        # ตรวจสอบว่าไฟล์มีอยู่
        if not pd.io.common.file_exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found. Please ensure 'listings.csv' is in the project folder.")
        
        # พยายามอ่าน CSV ด้วย encoding ต่าง ๆ
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'utf-8-sig']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print(f"Successfully read listings.csv with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("Failed to read listings.csv due to encoding issues. Tried encodings: " + ", ".join(encodings))
        
        # ตรวจสอบว่าไฟล์ว่างหรือไม่
        if df.empty:
            raise ValueError("File is empty or contains no data.")
        
        # แสดงคอลัมน์ทั้งหมด
        print(f"Available columns in listings.csv: {df.columns.tolist()}")
        
        # แปลง price เป็นตัวเลข
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
        
        # พิมพ์ตัวอย่าง price
        print("Sample values in 'price' column (before cleaning):")
        print(df['price'].head(10).tolist())
        print("Sample values in 'price' column (after cleaning):")
        print(df['price'].head(10).tolist())
        
        # โหลด reviews.csv
        df_reviews = None
        if pd.io.common.file_exists(reviews_path):
            for encoding in encodings:
                try:
                    df_reviews = pd.read_csv(reviews_path, encoding=encoding, low_memory=False)
                    print(f"Successfully read reviews.csv with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            if df_reviews is None:
                print(f"⚠️ Failed to read reviews.csv due to encoding issues")
            elif df_reviews.empty:
                print(f"⚠️ reviews.csv is empty")
                df_reviews = None
            else:
                print(f"Available columns in reviews.csv: {df_reviews.columns.tolist()}")
        else:
            print(f"⚠️ reviews.csv not found at {reviews_path}")
        
        # สร้าง text summary สำหรับ LangChain
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = [f"Neighbourhood: {row['neighbourhood_cleansed']}, Room Type: {row['room_type']}, Price: {row['price']}, Rating: {row['review_scores_rating']}" 
                 for _, row in df.iterrows() if pd.notna(row['neighbourhood_cleansed']) and pd.notna(row['room_type']) and pd.notna(row['price']) and pd.notna(row['review_scores_rating'])]
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
        print("Dataset loaded successfully:")
        print(df.head())
        print(f"Loaded {len(docs)} documents")
        print("\nAvailable neighbourhoods:", df['neighbourhood_cleansed'].unique().tolist())
        print("Available room types:", df['room_type'].unique().tolist())
        if df_reviews is not None:
            print("Reviews dataset loaded successfully:")
            print(df_reviews.head())
    else:
        print("Failed to load dataset. Please check the file and try again.")