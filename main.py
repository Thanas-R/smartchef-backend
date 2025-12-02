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
import difflib
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartchef-backend")

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
app = FastAPI(title="SmartChef (TF-IDF vector backend)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# HELPERS: IO + Normalization
# ------------------------
def read_json_file(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json_file(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

_nonalpha_re = re.compile(r"[^a-z0-9\s]+")

def normalize_token(s: str) -> str:
    """Normalize ingredient-like strings to a stable token.
    Lowercase + strip + remove punctuation (keep spaces) + collapse whitespace.
    """
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = _nonalpha_re.sub("", s)  # remove punctuation
    s = " ".join(s.split())
    return s

# ------------------------
# LOAD DATA + NORMALIZE
# ------------------------
def load_ingredients() -> List[str]:
    if not os.path.exists(INGREDIENTS_PATH):
        logger.warning("ingredients.json not found at: %s", INGREDIENTS_PATH)
        return []
    data = read_json_file(INGREDIENTS_PATH)
    if isinstance(data, dict):
        items = data.get("ingredients", [])
    elif isinstance(data, list):
        items = data
    else:
        raise HTTPException(status_code=500, detail="ingredients.json unsupported format")
    # normalize and dedupe
    normalized = []
    seen = set()
    for it in items:
        tok = normalize_token(str(it))
        if tok and tok not in seen:
            normalized.append(tok)
            seen.add(tok)
    return normalized

def load_recipes(normalize_and_save: bool = True) -> List[Dict[str, Any]]:
    if not os.path.exists(RECIPES_PATH):
        logger.warning("recipes.json not found at: %s", RECIPES_PATH)
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

        # add normalized tokens placeholder (we'll canonicalize using ingredient vocab later)
        norm_tokens = [normalize_token(x) for x in r.get("ingredients", []) if isinstance(x, str) and x.strip()]
        r["_norm_ingredients_raw"] = norm_tokens  # raw normalized prior to canonicalization

    # save back minimal normalizations if requested (only IDs/titles were added)
    if normalize_and_save and changed:
        if raw_is_dict:
            raw["recipes"] = recipes
            write_json_file(RECIPES_PATH, raw)
        else:
            write_json_file(RECIPES_PATH, {"recipes": recipes})

    return recipes

# Global data
try:
    INGREDIENT_VOCAB = load_ingredients()  # normalized tokens list, canonical forms
except Exception:
    INGREDIENT_VOCAB = []

try:
    RECIPES = load_recipes(normalize_and_save=False)
except Exception:
    RECIPES = []

# Build quick lookup set for exact membership
INGREDIENT_VOCAB_SET = set(INGREDIENT_VOCAB)

# ------------------------
# FUZZY / CANONICALIZATION
# ------------------------
def canonicalize_token(token: str, cutoff: float = 0.77) -> str:
    """
    Map a normalized token to a canonical ingredient token from INGREDIENT_VOCAB using fuzzy matching.
    If no good match is found, return the normalized token itself.
    cutoff: difflib similarity threshold (0..1)
    """
    token = normalize_token(token)
    if not token:
        return token
    if token in INGREDIENT_VOCAB_SET:
        return token
    # try close matches by difflib
    if INGREDIENT_VOCAB:
        matches = difflib.get_close_matches(token, INGREDIENT_VOCAB, n=1, cutoff=cutoff)
        if matches:
            return matches[0]
    # nothing found -> return original normalized token
    return token

def canonicalize_recipe_tokens(recipes: List[Dict[str, Any]]):
    """Fill each recipe's `_norm_ingredients` using canonicalization (fuzzy)"""
    for r in recipes:
        raw_tokens = r.get("_norm_ingredients_raw", [])
        canon = [canonicalize_token(t) for t in raw_tokens if t]
        r["_norm_ingredients"] = canon

# fill recipe canonical tokens at startup
canonicalize_recipe_tokens(RECIPES)

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
        IDF[term] = math.log((DOC_COUNT + 1) / (count + 1)) + 1.0

    # default idf for unseen tokens
    default_idf = math.log((DOC_COUNT + 1) / 1) + 1.0

    # compute tf-idf for each recipe
    RECIPE_TFIDF = {}
    RECIPE_NORM = {}
    for r in recipes:
        tid = r["id"]
        tokens = r.get("_norm_ingredients", [])
        tf = Counter(tokens)
        vec = {}
        for term, tf_count in tf.items():
            idf = IDF.get(term, default_idf)
            vec[term] = tf_count * idf
        RECIPE_TFIDF[tid] = vec
        RECIPE_NORM[tid] = math.sqrt(sum(v * v for v in vec.values())) if vec else 0.0

# build index at startup
build_tfidf_index(RECIPES)

# ------------------------
# MODELS
# ------------------------
class MatchRequest(BaseModel):
    ingredients: List[str]

# ------------------------
# VECTOR UTILITIES
# ------------------------
def build_query_vector(user_ingredients: List[str]) -> Dict[str, float]:
    """
    Returns a TF-IDF vector (dict token->weight) for the user's supplied ingredients.
    We canonicalize user tokens using the same mapping as recipe tokens.
    Each user ingredient counts once (tf=1).
    """
    tokens_raw = [normalize_token(x) for x in user_ingredients if isinstance(x, str) and x.strip()]
    tokens = [canonicalize_token(t) for t in tokens_raw if t]
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
    return {"status": "SmartChef TF-IDF backend running", "recipes_count": len(RECIPES)}

@app.get("/api/ingredients")
async def api_ingredients():
    # return the canonical ingredient vocab (readable)
    return {"ingredients": INGREDIENT_VOCAB}

@app.get("/api/recipes")
async def api_list_recipes():
    # return recipes but hide internal _ fields
    out = []
    for r in RECIPES:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        out.append(rr)
    return {"recipes": out}

@app.post("/api/recipes/match")
async def api_match(
    req: MatchRequest,
    sort: str = Query("tfidf", description="sort=tfidf or sort=match"),
    top_k: Optional[int] = Query(None, description="limit results")
):
    """
    Body: { "ingredients": ["egg","milk"] }
    Query params:
      - sort: "tfidf" (default) sorts by weighted relevance OR "match" sorts by exact-match %.
      - top_k: optional integer limit.
    Returns:
      - matchPercentage: exact (#have / #recipe_total)*100 (integer) for UI progress
      - relevanceScore: TF-IDF cosine similarity scaled 0-100 (integer) for sorting/badge
      - vectorScore: raw sim float 0..1 (for debug if needed)
      - hasIngredients / missingIngredients (original display strings)
    """
    user_items = [str(x) for x in (req.ingredients or []) if isinstance(x, str) and x.strip()]
    if not user_items:
        return {"matches": []}

    # build TF-IDF query vector (canonicalized)
    qvec = build_query_vector(user_items)
    qnorm = math.sqrt(sum(v * v for v in qvec.values())) if qvec else 0.0

    # user_set normalized canonical tokens for exact-match membership
    user_set = set(canonicalize_token(normalize_token(x)) for x in user_items)

    results = []
    for r in RECIPES:
        recipe_orig_ings = r.get("ingredients", [])
        recipe_norm_tokens = r.get("_norm_ingredients", [])

        # exact-match lists (original strings preserved)
        has_list = [orig for orig, norm in zip(recipe_orig_ings, recipe_norm_tokens) if norm in user_set]
        # fallback: if mismatch in length between orig/list use membership check
        if len(has_list) == 0:
            # fallback membership
            has_list = [ing for ing in recipe_orig_ings if canonicalize_token(normalize_token(ing)) in user_set]

        missing_list = [ing for ing in recipe_orig_ings if canonicalize_token(normalize_token(ing)) not in user_set]

        total_count = len(recipe_norm_tokens)
        match_pct = int((len(has_list) / total_count) * 100) if total_count > 0 else 0

        # TF-IDF relevance
        recipe_vec = RECIPE_TFIDF.get(r["id"], {})
        recipe_norm = RECIPE_NORM.get(r["id"], 0.0)
        sim = cosine_similarity_vec(qvec, qnorm, recipe_vec, recipe_norm)
        relevance_score = int(sim * 100)  # 0..100 integer for UI/badge

        # include if any overlap or if tfidf relevance (use small threshold if needed)
        if match_pct > 0 or relevance_score > 0:
            results.append({
                "id": r["id"],
                "title": r.get("title"),
                "name": r.get("name"),
                "note": r.get("note", ""),
                "ingredients": recipe_orig_ings,
                "instructions": r.get("instructions", []),
                "hasIngredients": has_list,
                "missingIngredients": missing_list,
                "matchPercentage": match_pct,        # simple exact-match % (for progress bar)
                "relevanceScore": relevance_score,   # weighted TF-IDF relevance for sorting/badge
                "vectorScore": round(sim, 4)         # raw sim 0..1 (debug / fine-grain)
            })

    # sorting
    if sort == "match":
        results.sort(key=lambda x: x["matchPercentage"], reverse=True)
    else:
        results.sort(key=lambda x: (x["relevanceScore"], x["matchPercentage"]), reverse=True)

    if top_k is not None and isinstance(top_k, int) and top_k > 0:
        results = results[:top_k]

    return {"matches": results}

@app.get("/api/suggest")
async def api_suggest(q: str = Query("", min_length=1), top_k: int = Query(8)):
    """
    Suggest ingredient tokens (canonical) given an input string.
    Useful for autocomplete / fixing typos client-side.
    """
    qn = normalize_token(q)
    if not qn:
        return {"suggestions": []}
    # prefix matches (best)
    prefix = [w for w in INGREDIENT_VOCAB if w.startswith(qn)]
    # close matches next (difflib)
    close = difflib.get_close_matches(qn, INGREDIENT_VOCAB, n=top_k, cutoff=0.6)
    # combine preserving order and uniqueness
    seen = set()
    out = []
    for w in prefix + close:
        if w not in seen:
            seen.add(w)
            out.append(w)
            if len(out) >= top_k:
                break
    return {"suggestions": out}

@app.post("/api/recompute-index")
async def api_recompute_index():
    """
    Rebuild TF-IDF index. Call this when you change recipes.json or ingredients.json on disk.
    """
    global RECIPES, INGREDIENT_VOCAB, INGREDIENT_VOCAB_SET
    RECIPES = load_recipes(normalize_and_save=False)
    INGREDIENT_VOCAB = load_ingredients()
    INGREDIENT_VOCAB_SET = set(INGREDIENT_VOCAB)
    canonicalize_recipe_tokens(RECIPES)
    build_tfidf_index(RECIPES)
    return {"status": "ok", "recipes": len(RECIPES), "ingredients": len(INGREDIENT_VOCAB)}
