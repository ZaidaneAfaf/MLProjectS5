# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Chargement des données
iris = pd.read_csv("data/Iris.csv")

# 2. Supprimer la colonne Id si elle existe
if "Id" in iris.columns:
    iris.drop("Id", axis=1, inplace=True)

# 3. Préparation X et y
X = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = iris['Species']

# 4. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Entraînement du modèle
model = SVC(probability=True)  # ⚡ j'active les probabilités si tu veux les scores de confiance
model.fit(X_train, y_train)

# 6. Prédiction et évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy of the model: {accuracy:.2f}")

# 7. Sauvegarde du modèle dans models/
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/iris_model.pkl")
print("💾 Model saved as models/iris_model.pkl")
