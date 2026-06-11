from pathlib import Path
from pickle import load as pickle_load

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movies_and_similarity(movies_path='movies.pkl', similarity_path='similarity.pkl'):
    movies = pickle_load(open(movies_path, 'rb'))
    similarity_file = Path(similarity_path)

    if similarity_file.exists():
        similarity = pickle_load(open(similarity_file, 'rb'))
    else:
        tags = movies['tags'].fillna('')
        vectors = CountVectorizer(max_features=5000, stop_words='english').fit_transform(tags)
        similarity = cosine_similarity(vectors)

    return movies, similarity
