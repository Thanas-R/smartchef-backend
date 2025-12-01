# SmartChef - TF-IDF Backend

This backend uses scikit-learn TF-IDF vectors to match ingredients / search recipes.

## Run locally
1. Create `data/recipes.json` and `data/ingredients.json`.
2. `pip install -r requirements.txt`
3. `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

## Endpoints
- GET `/` health
- GET `/api/ingredients` -> { ingredients: [...] }
- GET `/api/recipes` -> { recipes: [...] } (metadata)
- GET `/api/recipes/{id}` -> full recipe object
- POST `/api/recipes/match` -> { ingredients: [...] } -> matches with has/missing + scores
- GET `/api/recipes/search?q=...` -> semantic search
- POST `/api/admin/reindex` -> reindex recipes.json

## Deployment
Use Render or any host. Ensure `startCommand` is `uvicorn main:app --host 0.0.0.0 --port $PORT`.
