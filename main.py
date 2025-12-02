from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, uuid, math
from difflib import SequenceMatcher

# -----------------------------------------------------
# PATHS
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

RECIPES_PATH = os.path.join(DATA_DIR, "recipes.json")
INGREDIENTS_PATH = os.path.join(DATA_DIR, "ingredients.json")

# -----------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------
app = FastAPI(title="SmartChef Simple Vector Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------
def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())

def fuzzy_match(a, b):
    """Return similarity between 0–1"""
    return SequenceMatcher(None, a, b).ratio()

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
raw_ing = load_json(INGREDIENTS_PATH)
INGREDIENTS = raw_ing["ingredients"] if isinstance(raw_ing, dict) else raw_ing

raw_recipes = load_json(RECIPES_PATH)
RECIPES = raw_recipes["recipes"] if isinstance(raw_recipes, dict) else raw_recipes

# Ensure IDs + normalize fields
for r in RECIPES:
    r.setdefault("id", str(uuid.uuid4()))
    r.setdefault("ingredients", [])
    r.setdefault("instructions", [])
    r.setdefault("title", r.get("name", ""))

# -----------------------------------------------------
# SIMPLE VECTOR EMBEDDING
# -----------------------------------------------------
def text_to_vector(tokens):
    vec = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec

def cosine(v1, v2):
    dot = sum(v1[w] * v2[w] for w in v1 if w in v2)
    mag1 = math.sqrt(sum(v * v for v in v1.values()))
    mag2 = math.sqrt(sum(v * v for v in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

# Precompute recipe vectors
for r in RECIPES:
    normalized = [normalize(x) for x in r["ingredients"]]
    r["_vec"] = text_to_vector(normalized)

# -----------------------------------------------------
# REQUEST MODEL
# -----------------------------------------------------
class MatchRequest(BaseModel):
    ingredients: list[str]

# -----------------------------------------------------
# MATCH ROUTE
# -----------------------------------------------------
@app.post("/api/recipes/match")
def match_recipes(req: MatchRequest):
    if not req.ingredients:
        return {"matches": []}

    # User input normalization + fuzzy correction
    user_norm = []
    for u in req.ingredients:
        u_n = normalize(u)
        # Find closest ingredient from list (fuzzy)
        best = sorted(
            INGREDIENTS,
            key=lambda x: fuzzy_match(u_n, normalize(x)),
            reverse=True,
        )[0]
        if fuzzy_match(u_n, normalize(best)) >= 0.65:
            user_norm.append(normalize(best))
        else:
            user_norm.append(u_n)

    qvec = text_to_vector(user_norm)

    results = []
    for r in RECIPES:
        sim = cosine(qvec, r["_vec"])
        relevance = int(sim * 100)

        if relevance == 0:
            continue

        has = [ing for ing in r["ingredients"] if normalize(ing) in user_norm]
        missing = [ing for ing in r["ingredients"] if normalize(ing) not in user_norm]

        results.append({
            "id": r["id"],
            "title": r["title"],
            "ingredients": r["ingredients"],
            "instructions": r["instructions"],
            "hasIngredients": has,
            "missingIngredients": missing,
            "relevancePercentage": relevance
        })

    results.sort(key=lambda x: x["relevancePercentage"], reverse=True)

    return {"matches": results}

# -----------------------------------------------------
# BASIC ROUTES
# -----------------------------------------------------
@app.get("/")
def root():
    return {"status": "Simple SmartChef backend running"}

@app.get("/api/ingredients")
def get_ingredients():
    return {"ingredients": INGREDIENTS}

@app.get("/api/recipes")
def get_recipes():
    return {"recipes": RECIPES}
