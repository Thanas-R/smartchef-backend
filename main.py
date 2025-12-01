# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import uuid
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel  # fast for TF-IDF dot-product
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RECIPES_PATH = os.path.join(DATA_DIR, "recipes.json")
INGREDIENTS_PATH = os.path.join(DATA_DIR, "ingredients.json")

app = FastAPI(title="SmartChef - TFIDF Vector Backend")

# CORS - allow all for dev; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals for embeddings cache
_VECTORIZER: Optional[TfidfVectorizer] = None
_VECTORS = None
_RECIPES: List[Dict[str, Any]] = []
_RECIPES_MTIME = 0.0
_LOCK = threading.RLock()

# ---------- Utility IO ----------
def read_json_file(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail=f"File not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading {path}: {str(e)}")

def write_json_file(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving {path}: {str(e)}")

# ---------- Load & normalize recipes ----------
def load_recipes(normalize_and_save: bool = True) -> List[Dict[str, Any]]:
    """
    Accepts either top-level list or {"recipes": [...]}
    Ensures each recipe has id, title, ingredients (list), instructions (list)
    """
    raw = read_json_file(RECIPES_PATH)
    changed = False

    if isinstance(raw, dict):
        recipes = raw.get("recipes", [])
    elif isinstance(raw, list):
        recipes = raw
    else:
        raise HTTPException(status_code=500, detail="recipes.json has unsupported format")

    for r in recipes:
        if not isinstance(r, dict):
            raise HTTPException(status_code=500, detail="Each recipe must be an object")
        if "id" not in r or not r.get("id"):
            r["id"] = str(uuid.uuid4())
            changed = True
        if "name" in r and "title" not in r:
            r["title"] = r.get("name")
            changed = True
        if "ingredients" not in r or not isinstance(r["ingredients"], list):
            r["ingredients"] = r.get("ingredients", []) if r.get("ingredients") else []
            changed = True
        # instructions could be string or list; normalize to list of steps
        if "instructions" in r:
            if isinstance(r["instructions"], str):
                # naive split by newlines
                r["instructions"] = [line.strip() for line in r["instructions"].splitlines() if line.strip()]
                changed = True
            elif not isinstance(r["instructions"], list):
                r["instructions"] = []
                changed = True
        else:
            r["instructions"] = []

    if changed and normalize_and_save:
        # write back as {"recipes": [...] } to maintain consistent shape
        write_json_file(RECIPES_PATH, {"recipes": recipes})

    return recipes

# ---------- Load ingredients ----------
def load_ingredients() -> List[str]:
    data = read_json_file(INGREDIENTS_PATH)
    if isinstance(data, dict):
        return data.get("ingredients", [])
    if isinstance(data, list):
        return data
    raise HTTPException(status_code=500, detail="ingredients.json has unsupported format")

# ---------- Build documents and embeddings ----------
def build_document_text(recipe: Dict[str, Any]) -> str:
    parts = []
    # title / name
    if recipe.get("title"):
        parts.append(str(recipe.get("title")))
    if recipe.get("note"):
        parts.append(str(recipe.get("note")))
    # ingredients joined
    if recipe.get("ingredients"):
        parts.append(" ".join([str(i) for i in recipe["ingredients"]]))
    # optionally include instructions text (join)
    if recipe.get("instructions"):
        if isinstance(recipe["instructions"], list):
            parts.append(" ".join(recipe["instructions"]))
        else:
            parts.append(str(recipe["instructions"]))
    return " ".join(parts)

def compute_embeddings(force: bool = False):
    """
    Load recipes.json, compute TF-IDF vectorizer + matrix; cache globals.
    Will skip recompute if file not changed unless force=True.
    """
    global _VECTORIZER, _VECTORS, _RECIPES, _RECIPES_MTIME
    with _LOCK:
        if not os.path.exists(RECIPES_PATH):
            raise HTTPException(status_code=500, detail=f"Missing {RECIPES_PATH}")

        mtime = os.path.getmtime(RECIPES_PATH)
        if not force and _RECIPES_mtime_valid(mtime):
            # already up-to-date
            return

        recipes = load_recipes(normalize_and_save=True)
        documents = [build_document_text(r).lower() for r in recipes]

        # TF-IDF vectorizer (English stopwords)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        vectors = vectorizer.fit_transform(documents)  # sparse matrix (n_recipes x n_features)

        _VECTORIZER = vectorizer
        _VECTORS = vectors
        _RECIPES = recipes
        _RECIPES_MTIME = mtime

def _RECIPES_mtime_valid(mtime: float) -> bool:
    global _RECIPES_MTIME
    return _RECIPES_MTIME and abs(_RECIPES_MTIME - mtime) < 0.001

# Ensure initial index build at startup
@app.on_event("startup")
def startup_event():
    try:
        compute_embeddings(force=True)
        # also ensure ingredients file exists (but don't crash if missing)
        try:
            load_ingredients()
        except HTTPException:
            pass
    except Exception as e:
        # Log to console and continue — endpoints will return 500 if files missing
        print("Startup indexing error:", str(e))

# ---------- Request models ----------
class MatchRequest(BaseModel):
    ingredients: List[str]

# ---------- Endpoints ----------
@app.get("/")
async def root():
    return {"status": "SmartChef TF-IDF backend running", "recipes_indexed": len(_RECIPES)}

@app.get("/api/ingredients")
async def api_ingredients():
    return {"ingredients": load_ingredients()}

@app.get("/api/recipes")
async def api_list_recipes(limit: int = Query(100, ge=1, le=1000)):
    # return basic recipe metadata (not necessarily full instructions to save bandwidth)
    compute_embeddings()  # lazy refresh if file changed
    return {"recipes": [
        {
            "id": r.get("id"),
            "title": r.get("title") or r.get("name"),
            "note": r.get("note", ""),
            "ingredients": r.get("ingredients", []),
            "hasInstructions": bool(r.get("instructions"))
        } for r in _RECIPES[:limit]
    ]}

@app.get("/api/recipes/{recipe_id}")
async def api_get_recipe(recipe_id: str):
    compute_embeddings()
    for r in _RECIPES:
        if str(r.get("id")) == str(recipe_id):
            return r
    raise HTTPException(status_code=404, detail="Recipe not found")

@app.post("/api/recipes/match")
async def api_match(req: MatchRequest, top_k: int = Query(12, ge=1, le=100), similarity_threshold: float = Query(0.20, ge=0.0, le=1.0)):
    """
    Match by user's ingredients list.
    - We build a query string from ingredients and compute TF-IDF cosine similarity against recipe docs.
    - Additionally compute ingredient overlap (has/missing) to show exact matches.
    - We only return recipes with at least one ingredient match OR similarity above threshold.
    """
    compute_embeddings()
    user_ings = [i.strip().lower() for i in req.ingredients if isinstance(i, str) and i.strip()]
    if not user_ings:
        return {"matches": []}

    query_text = " ".join(user_ings)
    q_vec = _VECTORIZER.transform([query_text])
    # linear_kernel is an optimized dot product for sparse TF-IDF
    sim = linear_kernel(q_vec, _VECTORS).flatten()  # numpy array

    results = []
    user_set = set(user_ings)
    for idx, r in enumerate(_RECIPES):
        recipe_ings = [str(x).strip().lower() for x in r.get("ingredients", []) if isinstance(x, str)]
        has = [orig for orig in r.get("ingredients", []) if isinstance(orig, str) and orig.strip().lower() in user_set]
        missing = [orig for orig in r.get("ingredients", []) if isinstance(orig, str) and orig.strip().lower() not in user_set]
        ingredient_pct = int((len(has) / max(1, len(recipe_ings))) * 100) if recipe_ings else 0
        similarity = float(sim[idx])  # 0..1

        # filter: either some ingredient matches OR similarity above threshold
        if len(has) > 0 or similarity >= similarity_threshold:
            combined_score = (similarity * 0.6) + (ingredient_pct / 100.0 * 0.4)
            results.append({
                "id": r.get("id"),
                "title": r.get("title") or r.get("name"),
                "name": r.get("name"),
                "note": r.get("note", ""),
                "ingredients": r.get("ingredients", []),
                "instructions": r.get("instructions", []),
                "hasIngredients": has,
                "missingIngredients": missing,
                "matchPercentageIngredients": ingredient_pct,
                "similarityScore": round(similarity, 4),
                "combinedScore": round(combined_score, 4)
            })

    # sort by combinedScore desc, then similarity
    results.sort(key=lambda x: (x["combinedScore"], x["similarityScore"]), reverse=True)
    return {"matches": results[:top_k]}

@app.get("/api/recipes/search")
async def api_search(q: str = Query(...), top_k: int = Query(12, ge=1, le=200), similarity_threshold: float = Query(0.15, ge=0.0, le=1.0)):
    """
    Generic semantic search: return recipes most similar to the provided query text.
    """
    compute_embeddings()
    query = q.strip().lower()
    if not query:
        return {"matches": []}

    q_vec = _VECTORIZER.transform([query])
    sim = linear_kernel(q_vec, _VECTORS).flatten()

    results = []
    for idx, r in enumerate(_RECIPES):
        s = float(sim[idx])
        if s >= similarity_threshold:
            results.append({
                "id": r.get("id"),
                "title": r.get("title") or r.get("name"),
                "note": r.get("note", ""),
                "similarityScore": round(s, 4),
                "ingredients": r.get("ingredients", []),
            })

    results.sort(key=lambda x: x["similarityScore"], reverse=True)
    return {"matches": results[:top_k]}

# Admin helper: reindex on demand (no auth by default — you may secure this)
@app.post("/api/admin/reindex")
async def api_reindex(force: bool = Query(False)):
    compute_embeddings(force=force)
    return {"status": "reindexed", "recipes_indexed": len(_RECIPES)}
