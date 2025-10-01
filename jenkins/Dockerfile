FROM python:3.11-slim

# Métadonnées
LABEL maintainer="ZaidaneAfaf"
LABEL description="Iris ML Training Container"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Crée le dossier de travail
WORKDIR /app

# Crée le dossier artifacts
RUN mkdir -p artifacts

# Copie d'abord requirements.txt depuis le bon répertoire
COPY jenkins/requirements.txt .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie le dataset depuis le bon répertoire
COPY data/Iris.csv ./Iris.csv

# Copie le script d'entraînement depuis le bon répertoire
COPY jenkins/train.py .

# Permissions pour le script Python
RUN chmod +x train.py

# Commande par défaut pour lancer le training
CMD ["python", "train.py"]