import matplotlib
matplotlib.use('Agg')  # Try using the Cairo backend

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample book data with titles only
book_data = {
    2014: ["The Goldfinch", "All the Light We Cannot See", "Big Little Lies", "The Fault in Our Stars", "The Martian"],
    2015: ["The Girl on the Train", "A Little Life", "The Nightingale", "Go Set a Watchman", "The Revenant"],
    2016: ["The Underground Railroad", "The Sun Down Motel", "The Woman in the Window", "A Man Called Ove", "Homegoing"],
    2017: ["The Handmaid's Tale", "Little Fires Everywhere", "Big Little Lies", "The Silent Patient", "Educated"],
    2018: ["Becoming", "The Tattooist of Auschwitz", "The Night Circus", "Circe", "Where the Crawdads Sing"],
    2019: ["The Silent Patient", "Where the Crawdads Sing", "The Tattooist of Auschwitz", "The Institute", "The Testaments"],
    2020: ["The Vanishing Half", "Where the Crawdads Sing", "The Midnight Library", "A Promised Land", "The Silent Patient"],
    2021: ["Project Hail Mary", "The Paper Palace", "Malibu Rising", "The Last Thing He Told Me", "Klara and the Sun"],
    2022: ["Lessons in Chemistry", "Tomorrow, and Tomorrow, and Tomorrow", "The Measure", "Book Lovers", "Remarkably Bright Creatures"],
    2023: ["Hello Beautiful", "Demon Copperhead", "The Covenant of Water", "The House of Eve", "I Have Some Questions for You"],
    2024: ["The Women", "James", "My Name is Barbra", "House of Cotton", "The Berry Pickers"]
}


GOOGLE_BOOKS_API_URL = "https://www.googleapis.com/books/v1/volumes?q={}"

# Book rating matrix for linear algebra example (random ratings for simplicity)
book_ratings_matrix = {
    "The Vanishing Half": [4.5, 4.0, 4.8, 5.0],
    "Where the Crawdads Sing": [4.0, 4.5, 4.6, 4.9],
    "The Midnight Library": [4.2, 4.0, 4.9, 4.7],
    "A Promised Land": [4.6, 4.4, 4.7, 4.8],
    "The Silent Patient": [4.7, 4.2, 4.9, 4.6],
    "Project Hail Mary": [4.8, 4.6, 4.5, 4.8],
    "The Paper Palace": [4.2, 4.5, 4.3, 4.6],
    "Malibu Rising": [4.5, 4.7, 4.6, 4.9],
    "The Last Thing He Told Me": [4.3, 4.2, 4.8, 4.5],
    "Klara and the Sun": [4.6, 4.8, 4.7, 4.9],
}

users = ["User 1", "User 2", "User 3", "User 4"]  # Simulated users for rating matrix

def fetch_book_details(title):
    """Fetch book details from Google Books API"""
    response = requests.get(GOOGLE_BOOKS_API_URL.format(title))
    data = response.json()
    try:
        book_info = data["items"][0]["volumeInfo"]
        return {
            "title": book_info.get("title", "Unknown Title"),
            "author": ", ".join(book_info.get("authors", ["Unknown Author"])),
            "description": book_info.get("description", "No description available."),
            "image": book_info["imageLinks"]["thumbnail"] if "imageLinks" in book_info else "https://via.placeholder.com/128x192.png?text=No+Image",
            "published_date": book_info.get("publishedDate", "Unknown"),
            "rating": book_info.get("averageRating", "No rating")
        }
    except (KeyError, IndexError):
        return {
            "title": title,
            "author": "Unknown Author",
            "description": "No description available.",
            "image": "https://via.placeholder.com/128x192.png?text=No+Image",
            "published_date": "Unknown",
            "rating": "No rating"
        }

def generate_chart(year):
    """Generate a rating chart (Dummy ratings)"""
    books = book_data.get(year, [])
    ratings = np.random.uniform(4.0, 5.0, len(books))  # Generating random ratings

    df = pd.DataFrame({"title": books, "rating": ratings})

    plt.figure(figsize=(8, 5))
    sns.barplot(x="rating", y="title", data=df, palette="coolwarm")
    plt.xlabel("Rating")
    plt.ylabel("Books")
    plt.title(f"Top 5 Books of {year}")

    chart_path = f"static/chart_{year}.png"
    plt.savefig(chart_path)
    plt.close()  # Close the plot to avoid further issues
    return chart_path

def get_book_similarity(book_title):
    """Calculate similarity of a book with other books using linear algebra (cosine similarity)"""
    if book_title not in book_ratings_matrix:
        return []

    # Convert book ratings to a numpy array for matrix manipulation
    book_vector = np.array(book_ratings_matrix[book_title]).reshape(1, -1)

    # Get the ratings matrix for all books (excluding the selected book)
    all_books = list(book_ratings_matrix.keys())
    ratings_matrix = np.array(list(book_ratings_matrix.values()))

    # Compute cosine similarity between the selected book and all other books
    similarities = cosine_similarity(book_vector, ratings_matrix)[0]
    
    # Create a list of books and their similarity scores
    similarity_scores = [(book, similarity) for book, similarity in zip(all_books, similarities)]
    
    # Sort books based on similarity and return top 5
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
    return similarity_scores

@app.route("/")
def home():
    return render_template("index.html", years=list(book_data.keys()))

@app.route("/books/<int:year>")
def get_books(year):
    books = book_data.get(year, [])
    book_details = [fetch_book_details(book) for book in books]
    chart = generate_chart(year)
    return jsonify({"books": book_details, "chart": chart})

@app.route("/book_similarity/<book_title>")
def get_similar_books(book_title):
    similar_books = get_book_similarity(book_title)
    return jsonify({"book": book_title, "similar_books": similar_books})

if __name__ == "__main__":
    app.run(debug=True)
