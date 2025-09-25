# routes.py - VERSION CORRIGÉE
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import os

router = APIRouter()

# Chargement du modèle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "iris_model.pkl")
model = joblib.load(MODEL_PATH)

@router.post("/predict")
async def predict(features: dict):
    try:
        # Ordre exact des features
        data = np.array([[
            features["SepalLengthCm"],
            features["SepalWidthCm"],
            features["PetalLengthCm"],
            features["PetalWidthCm"]
        ]])
        
        print(f"Input shape: {data.shape}")  # Debug
        
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data).max()

        return JSONResponse({
            "prediction": str(prediction),
            "confidence": round(float(proba), 3)
        })
    except Exception as e:
        print(f"Erreur: {e}")  # Debug
        return JSONResponse({"error": str(e)}, status_code=400)