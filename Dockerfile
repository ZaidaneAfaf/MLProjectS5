# Utilise Python 3.13 léger
FROM python:3.13-slim

# Crée un dossier de travail dans le conteneur
WORKDIR /app

# Copie et installe les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le projet dans le conteneur
COPY . .

# Commande qui sera exécutée quand le conteneur tourne
CMD ["python", "train.py"]
