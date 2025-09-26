# jenkins/train.py - Version avec TensorBoard
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
import time

# TensorBoard - avec gestion d'erreur au cas où PyTorch n'est pas installé
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
    print("✅ TensorBoard disponible")
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️ TensorBoard non disponible (PyTorch non installé)")

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

# 4. Configuration TensorBoard
writer = None
if TENSORBOARD_AVAILABLE:
    # Créer le dossier dans artifacts (accessible depuis la racine)
    tensorboard_path = "../artifacts/tensorboard/iris_svm"
    os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(tensorboard_path)
    
    # Log du dataset
    writer.add_scalar("Dataset/nb_samples", len(X), 0)
    writer.add_scalar("Dataset/nb_features", len(feature_columns), 0)
    writer.add_scalar("Dataset/nb_classes", y.nunique(), 0)
    print(f"📊 Logs TensorBoard dans: {tensorboard_path}")

# 5. Vérification des données
print(f"\nDonnées X shape: {X.shape}")
print(f"Données y shape: {y.shape}")
print(f"Classes uniques: {y.unique()}")

# 6. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n📊 Split:")
print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# 7. Entraînement du modèle
print("\n🚀 Entraînement du modèle SVM...")
start_time = time.time()
model = SVC(probability=True, random_state=42)
model.fit(X_train, y_train)
train_time = time.time() - start_time

print(f"⏱️ Temps d'entraînement: {train_time:.2f}s")

# Log TensorBoard du temps d'entraînement
if writer:
    writer.add_scalar("Training/time_seconds", train_time, 0)

# 8. Prédiction et évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.3f}")

# Log TensorBoard de l'accuracy
if writer:
    writer.add_scalar("Performance/accuracy", accuracy, 0)

# 9. Métriques détaillées par classe
from sklearn.metrics import classification_report, confusion_matrix
print(f"\n📊 Rapport de classification:")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Log TensorBoard des métriques par classe
if writer:
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            writer.add_scalar(f"Class_{class_name}/precision", metrics['precision'], 0)
            writer.add_scalar(f"Class_{class_name}/recall", metrics['recall'], 0)
            writer.add_scalar(f"Class_{class_name}/f1-score", metrics['f1-score'], 0)

# 10. Matrice de confusion
print(f"\n📊 Matrice de confusion:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 11. Vérification finale du modèle
print(f"\n🔍 Modèle entraîné:")
print(f"Features attendues: {model.n_features_in_}")
print(f"Feature names: {getattr(model, 'feature_names_in_', 'Non disponible')}")
print(f"Classes: {model.classes_}")

# 12. Test rapide
test_data = X_test.iloc[0:1]
print(f"\n🧪 Test avec une prédiction:")
print(f"Input shape: {test_data.shape}")
print(f"Input: {test_data.values}")
pred = model.predict(test_data)
proba = model.predict_proba(test_data).max()
print(f"Prédiction: {pred[0]}")
print(f"Confiance: {proba:.3f}")

# Log TensorBoard du test
if writer:
    writer.add_text("Example/prediction", f"{pred[0]} (confiance={proba:.3f})", 0)

# 13. Sauvegarde du modèle (dans le dossier courant jenkins/)
joblib.dump(model, "iris_model.pkl")
print(f"\n💾 Modèle sauvegardé: iris_model.pkl")

# 14. Sauvegarder les informations des features
feature_info = {
    'feature_names': feature_columns,
    'n_features': len(feature_columns),
    'classes': list(model.classes_)
}
joblib.dump(feature_info, "feature_info.pkl")
print("💾 Info features sauvegardées: feature_info.pkl")

# 15. Fermer TensorBoard
if writer:
    writer.close()
    print("📊 TensorBoard fermé")

# 16. Instructions finales
print("\n" + "="*60)
print("🎯 RÉSULTATS DE L'ENTRAÎNEMENT")
print("="*60)
print(f"✅ Modèle: SVM avec accuracy {accuracy:.3f}")
print(f"✅ Temps d'entraînement: {train_time:.2f}s")
print(f"✅ Features: {len(feature_columns)}")
print(f"✅ Classes: {len(model.classes_)}")

if TENSORBOARD_AVAILABLE:
    print(f"✅ Logs TensorBoard sauvegardés")
    print("\n📊 POUR LANCER TENSORBOARD:")
    print("1. Dans un nouveau terminal, exécutez:")
    print("   tensorboard --logdir artifacts/tensorboard --port 6006")
    print("2. Ouvrez: http://localhost:6006")
    print("3. Les métriques disponibles:")
    print("   - Dataset: samples, features, classes")
    print("   - Training: temps d'entraînement")
    print("   - Performance: accuracy")
    print("   - Par classe: precision, recall, f1-score")
else:
    print("⚠️ TensorBoard non disponible")
    print("   Pour l'activer: pip install torch tensorboard")

print("="*60)