# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import uuid
import math
from collections import Counter

# ------------------------
# PATHS
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

RECIPES_PATH = os.path.join(DATA_DIR, "recipes.json")
INGREDIENTS_PATH = os.path.join(DATA_DIR, "ingredients.json")

# ------------------------
# FASTAPI APP
# ------------------------
app = FastAPI(title="SmartChef (TF-IDF Vector Backend)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# HELPERS
# ------------------------
def read_json_file(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json_file(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def normalize_token(s: str) -> str:
    # Normalize ingredient string to a stable token
    return " ".join(s.lower().strip().split())

# ------------------------
# LOAD DATA + NORMALIZE
# ------------------------
def load_ingredients() -> List[str]:
    if not os.path.exists(INGREDIENTS_PATH):
        print("❌ ingredients.json NOT FOUND at:", INGREDIENTS_PATH)
        return []
    data = read_json_file(INGREDIENTS_PATH)
    if isinstance(data, dict):
        items = data.get("ingredients", [])
    elif isinstance(data, list):
        items = data
    else:
        raise HTTPException(status_code=500, detail="ingredients.json unsupported format")
    return [str(x) for x in items]

def load_recipes(normalize_and_save: bool = True) -> List[Dict[str, Any]]:
    if not os.path.exists(RECIPES_PATH):
        print("❌ recipes.json NOT FOUND at:", RECIPES_PATH)
        return []
    raw = read_json_file(RECIPES_PATH)
    if isinstance(raw, dict):
        recipes = raw.get("recipes", [])
        raw_is_dict = True
    elif isinstance(raw, list):
        recipes = raw
        raw_is_dict = False
    else:
        raise HTTPException(status_code=500, detail="recipes.json unsupported format")

    changed = False
    for r in recipes:
        if not isinstance(r, dict):
            raise HTTPException(status_code=500, detail="Each recipe must be an object")
        if "id" not in r or not r.get("id"):
            r["id"] = str(uuid.uuid4())
            changed = True
        if "ingredients" not in r or not isinstance(r["ingredients"], list):
            r["ingredients"] = r.get("ingredients", []) or []
            changed = True
        if "instructions" not in r or not isinstance(r["instructions"], list):
            r["instructions"] = r.get("instructions", []) or []
            changed = True
        if "name" in r and "title" not in r:
            r["title"] = r["name"]
            changed = True

        # Add normalized tokens cache (not saved back unless changed writing)
        norm_tokens = [normalize_token(x) for x in r.get("ingredients", []) if isinstance(x, str) and x.strip()]
        r["_norm_ingredients"] = norm_tokens

    # write back a normalized file if requested and shape changed
    if normalize_and_save and changed:
        if raw_is_dict:
            raw["recipes"] = recipes
            write_json_file(RECIPES_PATH, raw)
        else:
            write_json_file(RECIPES_PATH, {"recipes": recipes})

    return recipes

# Global data
try:
    INGREDIENTS = load_ingredients()
except Exception:
    INGREDIENTS = []

try:
    RECIPES = load_recipes(normalize_and_save=False)
except Exception:
    RECIPES = []

# ------------------------
# TF-IDF / IDF computations (ingredient tokens as "terms")
# ------------------------
IDF: Dict[str, float] = {}
DOC_COUNT = max(1, len(RECIPES))
RECIPE_TFIDF: Dict[str, Dict[str, float]] = {}  # recipe_id -> {token: weight}
RECIPE_NORM: Dict[str, float] = {}  # precomputed vector norms

def build_tfidf_index(recipes: List[Dict[str, Any]]):
    global IDF, DOC_COUNT, RECIPE_TFIDF, RECIPE_NORM
    DOC_COUNT = max(1, len(recipes))
    # document frequency
    df = Counter()
    for r in recipes:
        tokens = set(r.get("_norm_ingredients", []))
        for t in tokens:
            df[t] += 1

    # idf smoothing to avoid zero / divide problems
    IDF = {}
    for term, count in df.items():
        # classic idf with smoothing + add 1 to avoid zero
        IDF[term] = math.log((DOC_COUNT + 1) / (count + 1)) + 1.0

    # default idf for unseen tokens
    default_idf = math.log((DOC_COUNT + 1) / 1) + 1.0

    # compute tf-idf for each recipe
    RECIPE_TFIDF = {}
    RECIPE_NORM = {}
    for r in recipes:
        tid = r["id"]
        tokens = r.get("_norm_ingredients", [])
        # term frequency (tf) — ingredients are usually unique; using count anyway
        tf = Counter(tokens)
        vec = {}
        for term, tf_count in tf.items():
            idf = IDF.get(term, default_idf)
            vec[term] = tf_count * idf
        # store vector and its norm
        RECIPE_TFIDF[tid] = vec
        norm = math.sqrt(sum(v * v for v in vec.values())) if vec else 0.0
        RECIPE_NORM[tid] = norm

# Build index at startup
build_tfidf_index(RECIPES)

# ------------------------
# MODELS
# ------------------------
class MatchRequest(BaseModel):
    ingredients: List[str]

# ------------------------
# VECTOR UTILS
# ------------------------
def build_query_vector(user_ingredients: List[str]) -> Dict[str, float]:
    """
    Returns a TF-IDF vector (dict token->weight) for the user's supplied ingredients.
    We use IDF computed across recipes. Each user ingredient is counted once (tf=1).
    """
    tokens = [normalize_token(x) for x in user_ingredients if isinstance(x, str) and x.strip()]
    tf = Counter(tokens)
    vec = {}
    default_idf = math.log((DOC_COUNT + 1) / 1) + 1.0
    for term, count in tf.items():
        idf = IDF.get(term, default_idf)
        vec[term] = count * idf
    return vec

def cosine_similarity_vec(vec1: Dict[str, float], norm1: Optional[float], vec2: Dict[str, float], norm2: Optional[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    # dot product only over intersection
    dot = 0.0
    for k, v in vec1.items():
        if k in vec2:
            dot += v * vec2[k]
    if (norm1 is None) or norm1 == 0.0:
        norm1 = math.sqrt(sum(v * v for v in vec1.values())) if vec1 else 0.0
    if (norm2 is None) or norm2 == 0.0:
        norm2 = math.sqrt(sum(v * v for v in vec2.values())) if vec2 else 0.0
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)

# ------------------------
# API ROUTES
# ------------------------
@app.get("/")
async def root():
    return {"status": "SmartChef TF-IDF backend running", "recipes": len(RECIPES)}

@app.get("/api/ingredients")
async def api_ingredients():
    return {"ingredients": INGREDIENTS}

@app.get("/api/recipes")
async def api_list_recipes():
    # return recipes but don't include internal _norm fields or tfidf maps
    out = []
    for r in RECIPES:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        out.append(rr)
    return {"recipes": out}

@app.post("/api/recipes/match")
async def api_match(req: MatchRequest, sort: str = Query("tfidf", description="sort=tfidf or sort=match"), top_k: Optional[int] = Query(None, description="limit results")):
    """
    Request body: { "ingredients": ["egg","milk"] }
    Query params:
      - sort: "tfidf" (default) to sort by weighted relevance OR "match" to sort by simple exact-match %.
      - top_k: optional integer to limit returned matches.
    Returns matches with:
      - matchPercentage: simple (#have / #recipe_total) * 100 (integer) — used for UI progress
      - relevanceScore: TF-IDF cosine similarity scaled 0-100 (integer) — use for sorting / badges
      - hasIngredients, missingIngredients
    """
    user_items = [str(x) for x in (req.ingredients or []) if isinstance(x, str) and x.strip()]
    user_set = set(normalize_token(x) for x in user_items)
    if not user_items:
        return {"matches": []}

    # query tf-idf vector
    qvec = build_query_vector(user_items)
    qnorm = math.sqrt(sum(v * v for v in qvec.values())) if qvec else 0.0

    results = []
    for r in RECIPES:
        recipe_tokens_orig = r.get("ingredients", [])
        recipe_tokens = [normalize_token(x) for x in r.get("_norm_ingredients", [])]
        recipe_set = set(recipe_tokens)
        # exact-match counts (for UI progress)
        matched_tokens = [orig for orig, norm in zip(recipe_tokens_orig, recipe_tokens) if norm in user_set]
        # note: matched_tokens currently maps using zip; if duplicate normalized tokens appear it still works
        # fallback: derive matched names by checking normalized membership:
        has_list = [ing for ing in r.get("ingredients", []) if normalize_token(ing) in user_set]
        missing_list = [ing for ing in r.get("ingredients", []) if normalize_token(ing) not in user_set]
        total_count = len(recipe_tokens)
        if total_count == 0:
            match_pct = 0
        else:
            match_pct = int((len(has_list) / total_count) * 100)

        # TF-IDF relevance
        recipe_vec = RECIPE_TFIDF.get(r["id"], {})
        recipe_norm = RECIPE_NORM.get(r["id"], 0.0)
        sim = cosine_similarity_vec(qvec, qnorm, recipe_vec, recipe_norm)
        relevance_score = int(sim * 100)  # 0-100

        # include if either some overlap or some relevance
        if match_pct > 0 or relevance_score > 0:
            results.append({
                "id": r["id"],
                "title": r.get("title"),
                "name": r.get("name"),
                "note": r.get("note", ""),
                "ingredients": r.get("ingredients", []),
                "instructions": r.get("instructions", []),
                "hasIngredients": has_list,
                "missingIngredients": missing_list,
                "matchPercentage": match_pct,        # for progress UI (exact-match %)
                "relevanceScore": relevance_score   # weighted TF-IDF relevance for sorting/badge
            })

    # sort
    if sort == "match":
        results.sort(key=lambda x: x["matchPercentage"], reverse=True)
    else:
        # default sort: TF-IDF relevance then fallback to matchPercentage
        results.sort(key=lambda x: (x["relevanceScore"], x["matchPercentage"]), reverse=True)

    if top_k is not None and isinstance(top_k, int) and top_k > 0:
        results = results[:top_k]

    return {"matches": results}

@app.post("/api/recompute-index")
async def api_recompute_index():
    """
    Rebuild TF-IDF index (call this if you change recipes.json on disk).
    """
    global RECIPES, INGREDIENTS
    RECIPES = load_recipes(normalize_and_save=False)
    build_tfidf_index(RECIPES)
    return {"status": "ok", "recipes": len(RECIPES)}

