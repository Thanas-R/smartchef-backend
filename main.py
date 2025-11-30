from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from pydantic import BaseModel
import google.generativeai as genai

# --- FIXED PATH HANDLING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPES_PATH = os.path.join(BASE_DIR, "data", "recipes.json")
INGREDIENTS_PATH = os.path.join(BASE_DIR, "data", "ingredients.json")

# --- Setup Gemini ---
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

app = FastAPI()

# Allow Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------- LOAD HELPERS ----------------------
def load_json(path):
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail=f"File missing: {path}")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail=f"Could not load {path}")


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------- INGREDIENT LIST ----------------------
@app.get("/api/ingredients")
async def get_ingredients():
    data = load_json(INGREDIENTS_PATH)
    return { "ingredients": data.get("ingredients", []) }


# ---------------------- RECIPE MATCHING ----------------------
class MatchRequest(BaseModel):
    ingredients: list[str]


@app.post("/api/recipes/match")
async def recipe_match(req: MatchRequest):
    data = load_json(RECIPES_PATH)
    recipes = data.get("recipes", [])

    matches = []

    for r in recipes:
        recipe_ingredients = [i.lower() for i in r.get("ingredients", [])]
        user_ing = [u.lower() for u in req.ingredients]

        has = [i for i in recipe_ingredients if i in user_ing]
        missing = [i for i in recipe_ingredients if i not in user_ing]

        match_percentage = int((len(has) / len(recipe_ingredients)) * 100)

        matches.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "title": r.get("name"),
            "note": r.get("note"),
            "ingredients": recipe_ingredients,
            "instructions": r.get("instructions", []),
            "hasIngredients": has,
            "missingIngredients": missing,
            "matchPercentage": match_percentage
        })

    matches = sorted(matches, key=lambda x: x["matchPercentage"], reverse=True)

    return { "matches": matches }


# ---------------------- AI INSTRUCTION GENERATION ----------------------
class GenerateInstructions(BaseModel):
    recipe_id: str
    recipe_name: str
    ingredients: list[str]


@app.post("/api/generate-instructions")
async def generate_instructions(req: GenerateInstructions):
    try:
        data = load_json(RECIPES_PATH)
        recipes = data.get("recipes", [])

        recipe = next((r for r in recipes if r["id"] == req.recipe_id), None)
        if not recipe:
            r
