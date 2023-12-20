import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import missingno as msno

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

pd.set_option('display.max_columns', None)

amazon_titles = pd.read_csv("Amazon Prime Movies and TV Shows/titles.csv")
appletv_titles = pd.read_csv("AppleTV+ Movies and TV Shows/titles.csv")
disney_titles = pd.read_csv("Disney+ Movies and TV Shows/titles.csv")
hbo_titles = pd.read_csv("HBO Max Movies and TV Shows/titles.csv")
netflix_titles = pd.read_csv("Netflix Movies and TV Shows/titles.csv")
paramount_titles = pd.read_csv("Paramount+ Movies and TV Shows/titles.csv")

titles = pd.concat([amazon_titles, appletv_titles, disney_titles, hbo_titles, netflix_titles, paramount_titles], axis=0).reset_index()
titles.drop(['index'], axis=1, inplace=True)

# Seeing if we have duplicates
print("\n", titles[titles.duplicated() == True].head(5))

# Dropping duplicates
titles.drop_duplicates(inplace=True)

print("\n", titles.head(), "\n")

titles.info()

print("\n", titles.isna().sum())

# missing data in the dataset
msno.matrix(titles, sparkline=False, figsize=(30, 10), color=(0.69, 0.23, 0.42))
plt.title('Distribution of Missing Values', fontsize = 40)
plt.show()

# Handling 'genres' and 'production_countries' to handle list values as single value

titles['genres'] = titles['genres'].str.replace(r'[', '').str.replace(r"'", '').str.replace(r']', '')
titles['genre'] = titles['genres'].str.split(',').str[0]

titles['production_countries'] = titles['production_countries'].str.replace(r"[", '').str.replace(r"'", '').str.replace(r"]", '')
titles['production_country'] = titles['production_countries'].str.split(',').str[0]

titles.drop(['genres', 'production_countries'], axis=1, inplace=True)

print("\n", titles.head())

print("\n", titles['genre'].unique())
print("\n", titles['production_country'].unique())

# Fill empty values with NaN

titles['genre'] = titles['genre'].replace('', np.NaN)
titles['production_country'] = titles['production_country'].replace('', np.NaN)

# Handling 'seasons' to handle null values of type 'MOVIE' as 0

print("\n", len(titles.loc[(titles['seasons'].isna()) & (titles['type'] == 'MOVIE')]) == titles.seasons.isna().sum())

titles['seasons'].fillna(0, inplace=True)

print("\n", titles.head())

# Handling the rest of the null values

print("\n", titles.isna().sum(), "\n")

titles.drop(['imdb_id', 'age_certification'], axis=1, inplace=True)

# Get rid of NaN values in the dataset

titles.dropna(inplace=True)

titles.info()

# Now with the clean data, create a Recommendation System

# Content Based Recommender

# Plot description based Recommender
# In this kernel, we will build a recommendation system based on the description of titles. We will calculate pairwise similarity scores for all movies/tv shows based on their descriptions and recommend titles with similar scores.

print("\n", titles['description'].head())

# Adding the Streaming Platform for the titles

lt = []
for i in titles['id']:
    movie_streaming = []
    if i in amazon_titles['id'].values:
        movie_streaming.append('amazon')
    if i in appletv_titles['id'].values:
        movie_streaming.append('appletv')
    if i in disney_titles['id'].values:
        movie_streaming.append('disney+')
    if i in hbo_titles['id'].values:
        movie_streaming.append('hbomax')
    if i in netflix_titles['id'].values:
        movie_streaming.append('netflix')
    if i in paramount_titles['id'].values:
        movie_streaming.append('paramount+')
    lt.append(movie_streaming)

titles['streaming_platform'] = lt
print("\n", titles.head())

# Separating the data in Movies and TV Shows

movies = titles[titles['type'] == 'MOVIE'].copy().reset_index()
movies.drop(['index'], axis=1, inplace=True)

shows = titles[titles['type'] == 'SHOW'].copy().reset_index()
shows.drop(['index'], axis=1, inplace=True)

print("\n", movies.head())
print("\n", shows.head())

# Determine importance of words with the help of the number of instances of the words
# Compute Term Frequency-Inverse Document Frequency

# Define a TF-IDF Vectorizer Object
# This remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix_movies = tfidf.fit_transform(movies['description'])
tfidf_matrix_shows = tfidf.fit_transform(shows['description'])

# Output the shape of tfidf_matrix
print("\n", f'Shape for Movies: {tfidf_matrix_movies.shape}')
print("\n", f'Shape for Shows: {tfidf_matrix_shows.shape}')

# Over 35k different words were used to describe the 13831 titles in our movies dataset
# And 19k different words to describe the 4543 titles in our shows dataset.

# Now, we need to calculate the similarity score a numeric quantity that denotes the similarity between two movies/shows.

# Compute the cosine similarity matrix
cosine_sim_movies = linear_kernel(tfidf_matrix_movies, tfidf_matrix_movies)
cosine_sim_shows = linear_kernel(tfidf_matrix_shows, tfidf_matrix_shows)

# Create a way to identify the index of a movie/show in our data, given its title.
indices_movies = pd.Series(movies.index, index=movies['title'])
indices_shows = pd.Series(shows.index, index=shows['title'])

def get_title(title, indices):
    """
    Function that gets the 'index searcher' and searches the user's title index.
    """

    try:
        index = indices[title]
    except:
        print("\n Title not found")
        return None

    if isinstance(index, np.int64):
        return index

    else:
        rt = 0
        print("Select a title: ")
        for i in range(len(index)):
            print(f"{i} - {movies['title'].iloc[index[i]]}", end=' ')
            print(f"({movies['release_year'].iloc[index[i]]})")

        rt = int(input())
        return index[rt]

# Define functions that accept a movie/show title as input and produce a list of 10 most similar titles.

def get_recommendations_movie(title, cosine_sim = cosine_sim_movies):
    """
    A function that takes a movie title as input and prints on the screen the 10 most similar movies based on the input description
    """

    title = get_title(title, indices_movies)
    if title == None:
        return
    
    idx = indices_movies[title]

    print(f"\n Title: {movies['title'].iloc[idx]} |  Year: {movies['release_year'].iloc[idx]}")

    print('**' * 40)

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    print(movies[['title', 'release_year', 'streaming_platform']].iloc[movie_indices])
    print('**' * 40)

def get_recommendations_show(title, cosine_sim = cosine_sim_shows):
    """
    A function that takes a show title as input and prints on the screen the 10 most similar shows based on the input description.
    """

    title = get_title(title, indices_shows)
    if title == None:
        return

    idx = indices_shows[title]

    print(f"\n Title: {shows['title'].iloc[idx]} |  Year: {shows['release_year'].iloc[idx]}")
    print('**' * 40)

    # Get the pairwise similarity scores of all shows with that show
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the shows based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #Get the scores of the 10 most similar shows
    sim_scores = sim_scores[1:11]

    # Get the show indices
    show_indices = [i[0] for i in sim_scores]

    print(shows[['title', 'release_year', 'streaming_platform']].iloc[show_indices])
    print('**' * 40)

get_recommendations_movie('Rocky')

get_recommendations_show('Narcos')

# Test title for 'Title not found
get_recommendations_movie('This is a Test')

# Test title for 'Title not found
get_recommendations_show('This is a Test')