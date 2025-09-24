from fastapi import APIRouter
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import os

router = APIRouter()

# Charger le modèle en chemin absolu robuste
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_model.pkl")
model = joblib.load(MODEL_PATH)

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
        proba = model.predict_proba(data).max()

        return JSONResponse({
            "prediction": str(prediction),
            "confidence": round(float(proba), 3)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
