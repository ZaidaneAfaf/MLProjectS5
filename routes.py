# routes.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

# Initialisation
router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Charger le modèle
model = joblib.load("models/iris_model.pkl")

# Route de test HTML
@router.get("/test", response_class=HTMLResponse)
async def say_hello(request: Request):
    return templates.TemplateResponse("test.html", {"request": request, "name": "Douaa"})

# Route API de prédiction
@router.post("/predict")
async def predict(features: dict):
    try:
        data = np.array([[ 
            features["SepalLengthCm"],
            features["SepalWidthCm"],
            features["PetalLengthCm"],
            features["PetalWidthCm"]
        ]])
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data).max()  # score de confiance
        return JSONResponse({
            "prediction": prediction,
            "confidence": round(float(proba), 3)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
