from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import math

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load recipes.json
# ----------------------------
with open("recipes.json", "r") as f:
    recipes = json.load(f)

# --------------------------------------------
# SIMPLE VECTOR EMBEDDING (PURE PYTHON)
# --------------------------------------------

def build_vocab(recipes):
    """Create a vocabulary of all unique ingredients"""
    vocab = {}
    idx = 0
    for r in recipes:
        for ing in r["ingredients"]:
            ing = ing.lower().strip()
            if ing not in vocab:
                vocab[ing] = idx
                idx += 1
    return vocab


def vectorize(ingredients, vocab):
    """Turn ingredient list into a vector of 0s and 1s"""
    vector = [0] * len(vocab)
    for ing in ingredients:
        ing = ing.lower().strip()
        if ing in vocab:
            vector[vocab[ing]] = 1
    return vector


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)


# --------------------------------------------
# Build the vocabulary and recipe vectors at startup
# --------------------------------------------
vocab = build_vocab(recipes)
recipe_vectors = [vectorize(r["ingredients"], vocab) for r in recipes]


# --------------------------------------------
# ROUTES
# --------------------------------------------

@app.get("/")
def home():
    return {"status": "SmartChef backend running successfully!"}


@app.get("/match")
def match(ingredients: str):
    """
    Example call: /match?ingredients=egg,tomato,bread
    """
    user_ings = [i.strip().lower() for i in ingredients.split(",")]
    user_vec = vectorize(user_ings, vocab)

    # Compare similarity with every recipe
    results = []
    for recipe, rec_vec in zip(recipes, recipe_vectors):
        score = cosine_similarity(user_vec, rec_vec)

        results.append({
            "name": recipe["name"],
            "note": recipe.get("note", ""),
            "ingredients": recipe["ingredients"],
            "score": round(score * 100, 2)  # convert to %
        })

    # Sort from highest match to lowest
    results = sorted(results, key=lambda r: r["score"], reverse=True)

    return results[:5]   # Top 5 results
