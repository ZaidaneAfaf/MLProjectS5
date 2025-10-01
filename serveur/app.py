# serveur/app.py
from fastapi import FastAPI
from .routes import router 

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API Iris Prediction Running 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy"}

app.include_router(router)
