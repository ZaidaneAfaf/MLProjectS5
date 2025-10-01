# jenkins/train.py - Version avec TensorBoard corrigée
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

# 4. Configuration TensorBoard - CHEMIN CORRIGÉ
writer = None
if TENSORBOARD_AVAILABLE:
    # Déterminer le répertoire racine du projet
    current_dir = os.getcwd()
    if current_dir.endswith('jenkins'):
        # Si on est dans jenkins/, remonter d'un niveau
        root_dir = os.path.dirname(current_dir)
    else:
        # Sinon on est déjà à la racine
        root_dir = current_dir
    
    # Chemin absolu vers TensorBoard
    tensorboard_path = os.path.join(root_dir, "artifacts", "tensorboard")
    os.makedirs(tensorboard_path, exist_ok=True)
    
    print(f"📁 Répertoire de travail: {current_dir}")
    print(f"📁 Répertoire racine: {root_dir}")
    print(f"📁 Chemin TensorBoard: {tensorboard_path}")
    
    # Vérifier que le dossier existe
    if os.path.exists(tensorboard_path):
        writer = SummaryWriter(tensorboard_path)
        print(f"📊 TensorBoard writer créé avec succès")
        
        # Log du dataset
        writer.add_scalar("Dataset/nb_samples", len(X), 0)
        writer.add_scalar("Dataset/nb_features", len(feature_columns), 0)
        writer.add_scalar("Dataset/nb_classes", y.nunique(), 0)
        print(f"📊 Logs TensorBoard dans: {tensorboard_path}")
    else:
        print(f"❌ Impossible de créer le dossier TensorBoard: {tensorboard_path}")
        TENSORBOARD_AVAILABLE = False

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
    print("📊 Temps d'entraînement loggé")

# 8. Prédiction et évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.3f}")

# Log TensorBoard de l'accuracy
if writer:
    writer.add_scalar("Performance/accuracy", accuracy, 0)
    print("📊 Accuracy loggée")

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
    print("📊 Métriques par classe loggées")

# 10. Matrice de confusion
print(f"\n📊 Matrice de confusion:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Ajouter la matrice de confusion à TensorBoard comme image
if writer:
    import matplotlib.pyplot as plt
    import numpy as np
    
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Étiquettes
        classes = model.classes_
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes,
               yticklabels=classes,
               title='Matrice de Confusion',
               ylabel='Vraie classe',
               xlabel='Classe prédite')
        
        # Ajouter les valeurs dans les cellules
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        # Sauvegarder et ajouter à TensorBoard
        writer.add_figure("Confusion_Matrix/test_set", fig, 0)
        plt.close(fig)
        print("📊 Matrice de confusion ajoutée à TensorBoard")
    except Exception as e:
        print(f"⚠️ Impossible d'ajouter la matrice de confusion: {e}")

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
    print("📊 Exemple de prédiction loggé")

# 13. Histogrammes des features
if writer:
    try:
        # Ajouter des histogrammes des features
        for i, feature in enumerate(feature_columns):
            writer.add_histogram(f"Features/{feature}", X[feature].values, 0)
        print("📊 Histogrammes des features ajoutés")
    except Exception as e:
        print(f"⚠️ Impossible d'ajouter les histogrammes: {e}")

# 14. Sauvegarde du modèle (dans le dossier courant jenkins/)
joblib.dump(model, "iris_model.pkl")
print(f"\n💾 Modèle sauvegardé: iris_model.pkl")

# 15. Sauvegarder les informations des features
feature_info = {
    'feature_names': feature_columns,
    'n_features': len(feature_columns),
    'classes': list(model.classes_)
}
joblib.dump(feature_info, "feature_info.pkl")
print("💾 Info features sauvegardées: feature_info.pkl")

# 16. Fermer TensorBoard et forcer l'écriture
if writer:
    # Forcer l'écriture des données
    writer.flush()
    writer.close()
    print("📊 TensorBoard fermé et données écrites")
    
    # Vérifier que les fichiers ont été créés
    tensorboard_files = []
    if os.path.exists(tensorboard_path):
        for root, dirs, files in os.walk(tensorboard_path):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    tensorboard_files.append(os.path.join(root, file))
    
    if tensorboard_files:
        print(f"✅ Fichiers TensorBoard créés: {len(tensorboard_files)}")
        for f in tensorboard_files:
            size = os.path.getsize(f)
            print(f"   - {f} ({size} bytes)")
    else:
        print("❌ Aucun fichier TensorBoard trouvé!")

# 17. Instructions finales
print("\n" + "="*60)
print("🎯 RÉSULTATS DE L'ENTRAÎNEMENT")
print("="*60)
print(f"✅ Modèle: SVM avec accuracy {accuracy:.3f}")
print(f"✅ Temps d'entraînement: {train_time:.2f}s")
print(f"✅ Features: {len(feature_columns)}")
print(f"✅ Classes: {len(model.classes_)}")

if TENSORBOARD_AVAILABLE and writer:
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
    print("   - Features: histogrammes des distributions")
    print("   - Confusion_Matrix: matrice de confusion")
    
    # Chemin absolu pour TensorBoard
    abs_tensorboard_path = os.path.abspath(tensorboard_path)
    print(f"\n📁 Chemin absolu TensorBoard: {abs_tensorboard_path}")
else:
    print("⚠️ TensorBoard non disponible")
    print("   Pour l'activer: pip install torch tensorboard matplotlib")

print("="*60)