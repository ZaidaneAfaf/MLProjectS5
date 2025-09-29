from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes import router
import os

app = FastAPI()
app.include_router(router)

# 🔹 Monter le dossier static pour servir index.html
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Endpoint racine pour rediriger vers le HTML
@app.get("/")
async def root():
    return {"message": "API Iris Prediction Running 🚀"}

# Optionnel : rediriger /index.html vers le fichier statique
@app.get("/index.html")
async def index_html():
    return StaticFiles(directory=STATIC_DIR)
