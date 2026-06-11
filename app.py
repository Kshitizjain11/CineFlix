import streamlit as st
from pickle import load as pickle_load

movies = pickle_load(open('movies.pkl','rb'))
movies_list = movies['title'].values
similarity = pickle_load(open('similarity.pkl','rb'))

def recommend(movie):
    recommended_movies = []
    movie_index = movies[movies['title']==movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    
    return recommended_movies


st.title('CinePlix - Movie Recommender System')



selected_movie = st.selectbox('Tell the movie you recently watched?',movies_list)

if st.button('Recommend'):
   recommendations = recommend(selected_movie)
   for i in recommendations:
       st.write(i)