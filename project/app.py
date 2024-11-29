import os
from dotenv import load_dotenv
from loaders.youtube_loader import load_youtube_texts
from langchain_community.vectorstores import PGVector
from embeddings.youtube_embedding import store_embeddings_pgvector, similarity_search_with_response
from yt_dlp import YoutubeDL

load_dotenv()

DB_CONNECTION_URL_2 = os.getenv("DB_CONNECTION_URL_2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def validate_env_variables():
    if not DB_CONNECTION_URL_2:
        raise ValueError("Database connection URL is missing!")
    print("Database connection URL loaded successfully.")
    
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key is missing!")
    print("Google API key loaded successfully.")
    
    if not GROQ_API_KEY:
        raise ValueError("Groq API key is missing!")
    print("Groq API key loaded successfully.")

validate_env_variables()

def main():
    try:
        youtube_video_url = "https://www.youtube.com/watch?v=1bUy-1hGZpI"

        texts = load_youtube_texts(youtube_video_url)
        print(texts)

        if not texts:
            print("No chunks generated from the video text.")
            return

        store_embeddings_pgvector(texts)

        while True:
            user_query = input("Enter your query (type 'exit' to stop): ")
            if user_query.lower() == "exit":
                print("Exiting the program...")
                break
            print("User Query:", user_query)

            response = similarity_search_with_response(user_query)
            print("\nFinal Response:\n", response)

    except Exception as e:
        print(f"Error in processing: {e}")

if __name__ == "__main__":
    main()
