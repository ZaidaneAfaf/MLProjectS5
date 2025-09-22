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

# Copie et installe les dépendances en premier (cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie les fichiers nécessaires
COPY train.py .
COPY Iris.csv .

# Permissions
RUN chmod +x train.py

# Commande par défaut pour lancer le training
CMD ["python", "train.py"]