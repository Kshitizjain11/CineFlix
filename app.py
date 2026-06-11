import streamlit as st
import requests
from os import getenv as os_getenv
from pathlib import Path
from dotenv import load_dotenv 
from pickle import load as pickle_load
import joblib

load_dotenv()

@st.cache_resource(show_spinner=False)
def get_movies_and_similarity():
    movies = pickle_load(open('movies.pkl', 'rb'))
    similarity_joblib = Path('similarity.joblib')

    if similarity_joblib.exists():
        similarity = joblib.load(similarity_joblib)
    else:
        raise FileNotFoundError('Missing similarity.joblib. Add the compressed similarity artifact to the repo.')

    return movies, similarity


movies, similarity = get_movies_and_similarity()
movies_list = movies['title'].values


def fetch_poster(movie_id):
    api_key = os_getenv('TMDB_API_KEY')
    if not api_key:
        return ""

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(url)
    data = response.json()

    poster_path = data.get('poster_path')
    if not poster_path:
        return ""

    return 'https://image.tmdb.org/t/p/original' + poster_path


def recommend(movie):
    recommended_movies = []
    recommended_movies_posters = []
    movie_index = movies[movies['title']==movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movies.iloc[i[0]].movie_id))
    
    return recommended_movies,recommended_movies_posters



st.title('CinePlix - Movie Recommender System')



selected_movie = st.selectbox('Tell the movie you recently watched?',movies_list)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    cols = st.columns(len(names))

    for col, name, poster in zip(cols, names, posters):
        with col:
            st.text(name)
            if poster:
                st.image(poster)
            else:
                st.caption("Poster unavailable")
                st.image(poster)

