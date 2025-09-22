# Dockerfile
FROM python:3.11-slim

# Crée le dossier de travail
WORKDIR /app

# Copie les fichiers nécessaires
COPY requirements.txt .
COPY train.py .
COPY Iris.csv .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Commande par défaut pour lancer le training
CMD ["python", "train.py"]
