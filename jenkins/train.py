# train_fixed.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import json

# 🔹 Répertoire de base (où est le script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 Chemins absolus pour les données et modèles
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Iris.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Chargement des données
iris = pd.read_csv(DATA_PATH)

print("📊 Dataset original:")
print(f"Shape: {iris.shape}")
print(f"Colonnes: {list(iris.columns)}")

# 2. Sélection des colonnes
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target_column = 'Species'

missing_cols = [col for col in feature_columns + [target_column] if col not in iris.columns]
if missing_cols:
    print(f"❌ Colonnes manquantes: {missing_cols}")
    print("Colonnes disponibles:", list(iris.columns))
    exit(1)

# 3. Features et target
X = iris[feature_columns].copy()
y = iris[target_column].copy()

print(f"\n✅ Features sélectionnées: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target: {y.name}")
print(f"Classes uniques: {y.unique()}")

# 4. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
#5. Sauvegarde du X_test et y_test
TEST_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "iris_test.json")

test_data = {
    "X": X_test.values.tolist(),
    "y": y_test.tolist()
}

with open(TEST_DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)

print(f"💾 Jeu de test sauvegardé dans : {TEST_DATA_PATH}")
# 6. Définition des modèles
models = {
    "svm": SVC(probability=True, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=200, random_state=42, multi_class="multinomial")
}

results = {}

# 6. Entraînement et évaluation
for name, model in models.items():
    print(f"\n🚀 Entraînement du modèle: {name}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy
    print(f"✅ {name} Accuracy: {accuracy:.3f}")

    # Test rapide
    test_data = X_test.iloc[0:1]
    pred = model.predict(test_data)[0]
    proba = model.predict_proba(test_data).max()

    print(f"🧪 Test {name}:")
    print(f"Input: {test_data.values}")
    print(f"Prédiction: {pred}, Confiance: {proba:.3f}")

    # Sauvegarde du modèle
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}_iris_model.pkl"))
    print(f"💾 Modèle sauvegardé: {os.path.join(MODELS_DIR, f'{name}_iris_model.pkl')}")

# 7. Sauvegarder infos features
feature_info = {
    'feature_names': feature_columns,
    'n_features': len(feature_columns),
    'models': list(models.keys())
}
joblib.dump(feature_info, os.path.join(MODELS_DIR, "feature_info.pkl"))
print("\n💾 Info features sauvegardées: ", os.path.join(MODELS_DIR, "feature_info.pkl"))

print("\n📊 Résumé des performances:")
for name, acc in results.items():
    print(f"{name}: {acc:.3f}")