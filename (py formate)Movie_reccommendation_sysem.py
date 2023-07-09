import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer

# Read the movie dataset
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge the movie and credits datasets
movies = movies.merge(credits, on='title')

# Select the relevant columns for the recommendation system
movies = movies[['movie_id', 'title', 'overview',
                 'genres', 'keywords', 'cast', 'crew']]

# Data preprocessing
movies.dropna(inplace=True)  # Remove missing data
movies.drop_duplicates(inplace=True)  # Remove duplicates

# Convert stringified lists to lists


def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Extract only the first three cast members
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])

# Extract the director from the crew


def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []


movies['crew'] = movies['crew'].apply(fetch_director)

# Tokenize the movie overview
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Combine all relevant information into tags
movies['tags'] = movies['overview'] + movies['genres'] + \
    movies['keywords'] + movies['cast'] + movies['crew']

# Convert tags to lowercase
movies['tags'] = movies['tags'].apply(lambda x: [word.lower() for word in x])

# Apply stemming to the tags
ps = PorterStemmer()
movies['tags'] = movies['tags'].apply(lambda x: [ps.stem(word) for word in x])

# Convert tags to string for vectorization
movies['tags'] = movies['tags'].apply(lambda x: ' '.join(x))

# Text to vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Save the data for future use
pickle.dump(movies, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
