import numpy as np
import pandas as pd
from os import getenv as os_getenv,path as os_path
from dotenv import load_dotenv
from fastapi import FastAPI,HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional,Dict,Any,List
from pydantic import BaseModel,Field
import httpx
from httpx import H

load_dotenv()

TMDB_API_KEY = os_getenv('TMDB_API_KEY')

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB API Key missing. Put it in the environment config")

app = FastAPI(title="Movie Recommendation System",version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

BASE_DIR = os_path.dirname(os_path.abspath(__file__))
DF_PATH = os_path.join(BASE_DIR,"df.pkl")
INDICES_PATH = os_path.join(BASE_DIR,"indices.pkl")
TFIDF_PATH = os_path.join(BASE_DIR,"tfidf.pkl")
TFIDF_MATRIX_PATH = os_path.join(BASE_DIR,"tfidf_matrix.pkl")

TITLE_TO_IDX : Optional[Dict[str,int]]  = None
df: Optional[pd.DataFrame]
indices_obj : Any = None
tfidf_matrix : Any = None
tfidf_obj : Any = None  

class TMDBMovieCard(BaseModel):
    tmdb_id : int
    title: str
    poster_url : Optional[str] = None
    release_date : Optional[str] = None
    vote_average : Optional[str] = None
    
class TMDBMovieDetails(BaseModel):
    tmdb_id : int
    title: str
    overview : str
    poster_url : Optional[str] = None
    backdrop_url : Optional[str] = None
    genres : List[Dict] = []
   
    
class TFIDFRecItem(BaseModel):
    title : str
    score : float
    tmdb  : Optional[TMDBMovieCard] = None

class SearchBundleResponse(BaseModel):
    query: set
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]

def _norm_title(t : str) -> str:
    return str(t).strip().lower()

def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMAGE_URL}{path}"

async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    TMDB GET Request:
    - TMDB API errors -> 502 with detail
    """
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE_URL}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
        )

    return r.json()

async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out

async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Raw TMDB response for keyword search (MULTIPLE results).
    Streamlit will use this for suggestions and grid.
    """
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None

