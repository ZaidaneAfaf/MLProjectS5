# serveur/app.py
from fastapi import FastAPI
from routes import router

app = FastAPI()
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "API Iris Prediction Running 🚀"}
