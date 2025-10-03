from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from routes import router
import os

app = FastAPI()

# =========================
# CORS (front et debug local)
# =========================
# Mets ici l'URL publique de ton front Azure
FRONT_URL_AZURE = "https://app-jenkis-ml-front.azurewebsites.net"

ALLOWED_ORIGINS = [
    FRONT_URL_AZURE,      # Front en production (Azure)
    "http://localhost:8502",  # Front local (Streamlit)
]

# Pour élargir via variable d'env (ex: "https://x.y;https://z.t")
extra_origins = os.getenv("EXTRA_CORS_ORIGINS", "")
if extra_origins:
    ALLOWED_ORIGINS.extend([o.strip() for o in extra_origins.split(";") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # en test, tu peux mettre ["*"] puis restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Routes applicatives
# =========================
app.include_router(router)

# =========================
# Fichiers statiques / index
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Racine → sert index.html si présent
@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "API Iris Prediction Running 🚀"}

# Accès direct à /index.html
@app.get("/index.html")
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"error": "index.html not found"}, status_code=404)

# =========================
# Healthcheck (Dockerfile)
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# (Optionnel) Version simple
# =========================
@app.get("/version")
def version():
    return {
        "app": "iris-backend",
        "env": {
            "WEBSITE_SITE_NAME": os.getenv("WEBSITE_SITE_NAME"),
            "WEBSITES_PORT": os.getenv("WEBSITES_PORT"),
        }
    }
