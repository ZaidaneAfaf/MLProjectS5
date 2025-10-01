# ========= Iris ML API - Image d'exécution =========
FROM python:3.11-slim

WORKDIR /app

# Copie du code et du modèle attendu par routes.py (../models/iris_model.pkl)
COPY serveur ./serveur
COPY models  ./models   
# (Tu peux aussi copier artifacts/ si tu en as besoin pour autre chose)
# COPY artifacts ./artifacts

# Dépendances + curl pour le HEALTHCHECK
RUN pip install --no-cache-dir \
      fastapi \
      uvicorn[standard] \
      scikit-learn \
      pandas \
      numpy \
      pydantic \
      joblib \
 && apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# (optionnel, utile pour les imports)
ENV PYTHONUNBUFFERED=1
# ENV PYTHONPATH=/app

EXPOSE 80

# Si ton app fournit bien /health (voir snippet plus bas), garde ceci :
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fs http://localhost:80/health || exit 1

CMD ["uvicorn", "serveur.app:app", "--host", "0.0.0.0", "--port", "80"]
