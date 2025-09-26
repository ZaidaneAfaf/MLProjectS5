# train_fixed.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
import time

# ---> AJOUT TENSORBOARD
from torch.utils.tensorboard import SummaryWriter

# 1. Chargement des données
iris = pd.read_csv("data/Iris.csv")

print("📊 Dataset original:")
print(f"Shape: {iris.shape}")
print(f"Colonnes: {list(iris.columns)}")

# 2. Supprimer TOUTES les colonnes non-features (Id, index, etc.)
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target_column = 'Species'

missing_cols = []
for col in feature_columns + [target_column]:
    if col not in iris.columns:
        missing_cols.append(col)

if missing_cols:
    print(f"❌ Colonnes manquantes: {missing_cols}")
    print("Colonnes disponibles:", list(iris.columns))
    exit(1)

# 3. Sélection exacte des features
X = iris[feature_columns].copy()
y = iris[target_column].copy()

print(f"\n✅ Features sélectionnées: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target: {y.name}")

# ---> AJOUT TENSORBOARD : création du writer
writer = SummaryWriter("artifacts/tensorboard/iris_svm")

# ---> AJOUT TENSORBOARD : log du dataset
writer.add_scalar("Dataset/nb_samples", len(X), 0)
writer.add_scalar("Dataset/nb_features", len(feature_columns), 0)
writer.add_scalar("Dataset/nb_classes", y.nunique(), 0)

# 4. Vérification des données
print(f"\nDonnées X shape: {X.shape}")
print(f"Données y shape: {y.shape}")
print(f"Classes uniques: {y.unique()}")

# 5. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Entraînement du modèle
start_time = time.time()
model = SVC(probability=True, random_state=42)
model.fit(X_train, y_train)
train_time = time.time() - start_time

# ---> AJOUT TENSORBOARD : log du temps d'entraînement
writer.add_scalar("Training/time_seconds", train_time, 0)

# 7. Prédiction et évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.3f}")

# ---> AJOUT TENSORBOARD : log accuracy
writer.add_scalar("Performance/accuracy", accuracy, 0)

# 8. Vérification finale du modèle
print(f"\n🔍 Modèle entraîné:")
print(f"Features attendues: {model.n_features_in_}")
print(f"Feature names: {getattr(model, 'feature_names_in_', 'Non disponible')}")

# 9. Test rapide
test_data = X_test.iloc[0:1]
print(f"\n🧪 Test avec une prédiction:")
print(f"Input shape: {test_data.shape}")
print(f"Input: {test_data.values}")
pred = model.predict(test_data)
proba = model.predict_proba(test_data).max()
print(f"Prédiction: {pred[0]}")
print(f"Confiance: {proba:.3f}")

# ---> AJOUT TENSORBOARD : log test rapide
writer.add_text("Example/prediction", f"{pred[0]} (confiance={proba:.3f})", 0)

# 10. Sauvegarde
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/iris_model.pkl")
print(f"\n💾 Modèle sauvegardé: models/iris_model.pkl")

# 11. Sauvegarder aussi les noms des features
feature_info = {
    'feature_names': feature_columns,
    'n_features': len(feature_columns)
}
joblib.dump(feature_info, "models/feature_info.pkl")
print("💾 Info features sauvegardées: models/feature_info.pkl")

# ---> FIN TENSORBOARD
writer.close()

# ---> NOUVEAU : INSTRUCTIONS POUR LANCER TENSORBOARD APRÈS L'ENTRAÎNEMENT
print("\n" + "="*60)
print("🎯 POUR LANCER TENSORBOARD APRÈS L'ENTRAÎNEMENT :")
print("="*60)
print("1. Les logs TensorBoard sont sauvegardés dans: artifacts/tensorboard/")
print("2. Pour lancer TensorBoard, exécutez dans un nouveau terminal:")
print("   tensorboard --logdir artifacts/tensorboard --port 6006")
print("3. Ouvrez: http://localhost:6006")
print("4. Les logs sont conservés entre les builds Jenkins")
print("="*60)"# Ajout instructions TensorBoard" 
