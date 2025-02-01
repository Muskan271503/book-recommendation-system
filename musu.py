import os
import json
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from requests import get

# Function to search for books using the Google Books API
def search_for_books(api_key, query, max_results=10):
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}&key={api_key}"
        result = get(url)
        result.raise_for_status()  # Raise an exception for HTTP errors
        books = json.loads(result.content).get("items", [])
        return books
    except Exception as e:
        st.error(f"Error searching for books: {e}")
        return []

# Function to extract book details
def extract_book_details(book):
    try:
        volume_info = book.get("volumeInfo", {})
        title = volume_info.get("title", "Unknown Title")
        authors = ", ".join(volume_info.get("authors", ["Unknown Author"]))
        published_date = volume_info.get("publishedDate", "Unknown Date")
        categories = ", ".join(volume_info.get("categories", ["Unknown Category"]))
        description = volume_info.get("description", "No description available")
        page_count = volume_info.get("pageCount", "Unknown Pages")
        average_rating = volume_info.get("averageRating", "No Rating")
        ratings_count = volume_info.get("ratingsCount", "No Ratings")
        book_cover = volume_info.get("imageLinks", {}).get("thumbnail", "https://i.postimg.cc/0QNxYz4V/social.png")
        book_link = volume_info.get("canonicalVolumeLink", volume_info.get("infoLink", "#"))
        
        # Ensure the book link starts with http:// or https://
        if not book_link.startswith("http://") and not book_link.startswith("https://"):
            book_link = f"https://books.google.com/books?id={book_link}"
        
        return [title, authors, published_date, categories, description, page_count, average_rating, ratings_count, book_cover, book_link]
    except Exception as e:
        st.error(f"Error extracting book details: {e}")
        return []

# Function to recommend books based on cosine similarity
def recommend_books(book_title, books_df, similarity_matrix, num_recommendations=5):
    try:
        # Find the index of the selected book
        index = books_df[books_df['Title'].str.lower() == book_title.lower()].index[0]
        
        # Get the similarity scores
        similarity_scores = list(enumerate(similarity_matrix[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Recommend the most similar books
        recommended_books = []
        for i in similarity_scores[1:num_recommendations+1]:  # Exclude the first one (it's the selected book itself)
            recommended_books.append(books_df.iloc[i[0]])
        
        return recommended_books
    except IndexError:
        st.error("Book not found in the dataset.")
        return []
    except Exception as e:
        st.error(f"Error recommending books: {e}")
        return []

# Main function to collect book data and recommend books
def main():
    st.title("📚 Book Recommendation System")
    
    # Input for API Key
    api_key = st.text_input("Enter Google Books API Key:", type="password")
    
    if api_key:
        search_query = st.text_input("Enter book search query:")
        if st.button("Search for Books"):
            if search_query:
                books = search_for_books(api_key, search_query)
                if books:
                    all_books_data = [extract_book_details(book) for book in books]
                    
                    # Convert the book data to a DataFrame
                    books_df = pd.DataFrame(all_books_data, columns=[
                        "Title", "Authors", "Published Date", "Categories", "Description", 
                        "Page Count", "Average Rating", "Ratings Count", "Book Cover", "Book Link"
                    ])
                    
                    # Create combined features for better similarity
                    books_df['Combined Features'] = books_df['Title'].fillna('') + " " + books_df['Authors'].fillna('') + " " + books_df['Description'].fillna('')
                    
                    # Use TF-IDF Vectorizer to convert combined features to vectors for similarity computation
                    tfidf = TfidfVectorizer(stop_words='english')
                    tfidf_matrix = tfidf.fit_transform(books_df['Combined Features'])
                    
                    # Compute cosine similarity between books
                    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
                    
                    # Select a book for recommendation
                    selected_book = st.selectbox("Select a book for recommendations", books_df['Title'].values)
                    
                    # Number of recommendations
                    num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
                    
                    # Recommend books
                    if st.button("Recommend Books"):
                        recommended_books = recommend_books(selected_book, books_df, cosine_sim, num_recommendations)
                        if recommended_books:
                            st.markdown(f"### Books similar to **{selected_book}**:")
                            for idx, book in enumerate(recommended_books):
                                # Display book title as a clickable link
                                st.markdown(f"### {idx+1}. [{book['Title']}]({book['Book Link']})")
                                st.write(f"**Authors**: {book['Authors']}")
                                st.write(f"**Published Date**: {book['Published Date']}")
                                st.write(f"**Categories**: {book['Categories']}")
                                st.write(f"**Rating**: {book['Average Rating']} ({book['Ratings Count']} ratings)")
                                st.write(f"**Description**: {book['Description']}")
                                st.image(book['Book Cover'], width=150)
                                st.markdown("---")
                        else:
                            st.write("No recommendations found.")
                else:
                    st.warning("No books found for the given query.")
            else:
                st.warning("Please enter a search query.")
    else:
        st.warning("Please enter your Google Books API key.")

if __name__ == "__main__":
    main()
