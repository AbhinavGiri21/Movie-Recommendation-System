from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

movies = pd.read_csv('Dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('Dataset/tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(obj):
    return [i['name'] for i in eval(obj)]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert_name(obj):
    return [i['name'] for i in eval(obj)][:3]

movies['cast'] = movies['cast'].apply(convert_name)

def fetch_director(obj):
    directors = [i['name'] for i in eval(obj) if i['job'] == 'Director']
    return directors if directors else ['Unknown']

movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))
movies['keywords'] = movies['keywords'].apply(lambda x: ' '.join(x))
movies['cast'] = movies['cast'].apply(lambda x: ' '.join(x))
movies['crew'] = movies['crew'].apply(lambda x: ' '.join(x))

movies['tags'] = movies['overview'] + ' ' + movies['genres'] + ' ' + movies['keywords'] + ' ' + movies['cast'] + ' ' + movies['crew']

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()

    movie_index = movies[movies['title'].str.lower() == movie].index

    if len(movie_index) == 0:
        return ['No movie available']
    else:
        movie_index = movie_index[0]  
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_movies = [movies.iloc[i[0]]['title'] for i in movies_list]
        return recommended_movies


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    movie_name = request.form.get('movie_name')
    recommended_movies = recommend(movie_name)
    return render_template('recommendation.html', movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
