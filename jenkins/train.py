import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import json
import psutil
import threading
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import shutil

# 🔹 Répertoire de base (où- est le script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 Chemins absolus pour les données et modèles
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Iris.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
LOGS_DIR = os.path.join(BASE_DIR, "..", "artifacts", "tensorboard")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# 🔹 NETTOYAGE DES LOGS EXISTANTS
if os.path.exists(LOGS_DIR):
    shutil.rmtree(LOGS_DIR)
os.makedirs(LOGS_DIR, exist_ok=True)

# 📊 Classe pour monitorer CPU/RAM
class ResourceMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.cpu_usage = []
        self.memory_usage = []
        self.memory_mb = []
        self.monitoring = False
        self.thread = None
        self.start_time = None
        self.process = psutil.Process()
        
    def _monitor(self):
        while self.monitoring:
            try:
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory().percent
                mem_mb = self.process.memory_info().rss / (1024 * 1024 * 1024)  # GB
                
                self.cpu_usage.append(cpu)
                self.memory_usage.append(mem)
                self.memory_mb.append(mem_mb)
                
                time.sleep(0.5)
            except:
                pass
    
    def start(self):
        self.monitoring = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        print(f"🔍 {self.model_name.upper()}: Monitoring démarré...")
    
    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)
        duration = time.time() - self.start_time
        
        # Affichage console
        print(f"\n📊 RÉSUMÉ RESSOURCES - {self.model_name.upper()}")
        print(f"⏱️  Durée monitoring: {duration:.0f} secondes")
        print(f"💻 CPU moyen: {sum(self.cpu_usage)/len(self.cpu_usage):.1f}% (max: {max(self.cpu_usage):.1f}%)")
        print(f"🧠 Mémoire moyenne: {sum(self.memory_usage)/len(self.memory_usage):.1f}% (max: {max(self.memory_usage):.1f}%)")
        print(f"💾 Mémoire utilisée: {sum(self.memory_mb)/len(self.memory_mb):.1f} GB")
        
        return {
            'duration': duration,
            'cpu_mean': sum(self.cpu_usage)/len(self.cpu_usage) if self.cpu_usage else 0,
            'cpu_max': max(self.cpu_usage) if self.cpu_usage else 0,
            'cpu_values': self.cpu_usage,
            'mem_mean': sum(self.memory_usage)/len(self.memory_usage) if self.memory_usage else 0,
            'mem_max': max(self.memory_usage) if self.memory_usage else 0,
            'mem_values': self.memory_usage,
            'mem_gb': sum(self.memory_mb)/len(self.memory_mb) if self.memory_mb else 0
        }

# Fonction pour entraîner progressivement et logger les métriques
def train_with_logging(model, X_train, y_train, X_test, y_test, model_name, writer, n_steps=20):
    """
    Entraîne le modèle progressivement en augmentant la taille du dataset
    et log les métriques à chaque étape pour créer des courbes
    """
    print(f"🔄 Entraînement progressif de {model_name}...")
    
    # Créer des subsets progressifs du dataset d'entraînement
    train_sizes = np.linspace(0.1, 1.0, n_steps)
    
    for step, size in enumerate(train_sizes):
        # Sélectionner un subset du training set
        n_samples = max(10, int(len(X_train) * size))
        X_subset = X_train[:n_samples]
        y_subset = y_train[:n_samples]
        
        # Entraîner le modèle sur ce subset
        model.fit(X_subset, y_subset)
        
        # Prédire sur le test set
        y_pred = model.predict(X_test)
        
        # Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Logger dans TensorBoard
        writer.add_scalar(f'metrics/accuracy/{model_name}', accuracy, step)
        writer.add_scalar(f'metrics/precision/{model_name}', precision, step)
        writer.add_scalar(f'metrics/recall/{model_name}', recall, step)
        writer.add_scalar(f'metrics/f1_score/{model_name}', f1, step)
        
        # Logger aussi pour la comparaison
        writer.add_scalars('comparison/accuracy', {model_name: accuracy}, step)
        writer.add_scalars('comparison/precision', {model_name: precision}, step)
        writer.add_scalars('comparison/recall', {model_name: recall}, step)
        writer.add_scalars('comparison/f1_score', {model_name: f1}, step)
        
        # Afficher la progression
        if (step + 1) % 5 == 0 or step == n_steps - 1:
            print(f"  Step {step+1}/{n_steps} ({int(size*100)}% données) - "
                  f"Acc: {accuracy:.3f}, Prec: {precision:.3f}, "
                  f"Rec: {recall:.3f}, F1: {f1:.3f}")
    
    # Retourner les métriques finales
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

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

# Convertir en numpy arrays pour faciliter le slicing
X_train = X_train.values
y_train = y_train.values

# 5. Sauvegarde du X_test et y_test
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
resources_stats = {}
metrics_stats = {}

# TensorBoard Writer
writer = SummaryWriter(log_dir=LOGS_DIR)

print("\n" + "="*70)
print("💻 ENTRAÎNEMENT PROGRESSIF AVEC COURBES")
print("="*70)

# 7. Entraînement et évaluation avec monitoring ET courbes
for idx, (name, model) in enumerate(models.items()):
    print(f"\n🚀 Entraînement du modèle: {name}")
    
    # Démarrer le monitoring
    monitor = ResourceMonitor(name)
    monitor.start()
    
    # Entraînement progressif avec logging des métriques
    final_metrics = train_with_logging(model, X_train, y_train, X_test, y_test, name, writer, n_steps=20)
    
    # Arrêter le monitoring
    stats = monitor.stop()
    resources_stats[name] = stats
    
    # Sauvegarder les métriques finales
    results[name] = final_metrics['accuracy']
    metrics_stats[name] = final_metrics
    
    print(f"✅ {name} - Métriques finales:")
    print(f"   Accuracy:  {final_metrics['accuracy']:.3f}")
    print(f"   Precision: {final_metrics['precision']:.3f}")
    print(f"   Recall:    {final_metrics['recall']:.3f}")
    print(f"   F1-Score:  {final_metrics['f1_score']:.3f}")
    
    # 📊 Résumé texte pour chaque modèle
    text_summary = f"""
**RÉSUMÉ COMPLET - {name.upper()}**

**MÉTRIQUES DE PERFORMANCE FINALES:**
- **Accuracy:** {final_metrics['accuracy']:.3f}
- **Precision:** {final_metrics['precision']:.3f}
- **Recall:** {final_metrics['recall']:.3f}
- **F1-Score:** {final_metrics['f1_score']:.3f}

**RESSOURCES CONSOMMÉES:**
- **Durée monitoring:** {stats['duration']:.0f} secondes
- **CPU moyen:** {stats['cpu_mean']:.1f}% (max: {stats['cpu_max']:.1f}%)
- **Mémoire moyenne:** {stats['mem_mean']:.1f}% (max: {stats['mem_max']:.1f}%)
- **Mémoire utilisée:** {stats['mem_gb']:.1f} GB
"""
    writer.add_text(f'model_resumes/{name}', text_summary, 0)
    
    # Métriques scalaires ressources
    writer.add_scalar(f'resources/cpu_mean/{name}', stats['cpu_mean'], 0)
    writer.add_scalar(f'resources/cpu_max/{name}', stats['cpu_max'], 0)
    writer.add_scalar(f'resources/memory_mean/{name}', stats['mem_mean'], 0)
    writer.add_scalar(f'resources/memory_max/{name}', stats['mem_max'], 0)
    writer.add_scalar(f'resources/memory_gb/{name}', stats['mem_gb'], 0)
    writer.add_scalar(f'resources/duration/{name}', stats['duration'], 0)
    
    # Test rapide avec le modèle final
    test_data_sample = X_test.iloc[0:1]
    pred = model.predict(test_data_sample)[0]
    proba = model.predict_proba(test_data_sample).max()

    print(f"🧪 Test {name}:")
    print(f"Input: {test_data_sample.values}")
    print(f"Prédiction: {pred}, Confiance: {proba:.3f}")

    # Sauvegarde du modèle final
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}_iris_model.pkl"))
    print(f"💾 Modèle sauvegardé: {os.path.join(MODELS_DIR, f'{name}_iris_model.pkl')}")

# 📊 RÉSUMÉ GLOBAL dans TensorBoard
global_summary = "# 💻 RÉSUMÉ COMPLET - TOUS LES MODÈLES\n\n"

for name in models.keys():
    stats = resources_stats[name]
    metrics = metrics_stats[name]
    global_summary += f"""
## {name.upper()}

**PERFORMANCES:**
- **Accuracy:** {metrics['accuracy']:.3f}
- **Precision:** {metrics['precision']:.3f}
- **Recall:** {metrics['recall']:.3f}
- **F1-Score:** {metrics['f1_score']:.3f}

**RESSOURCES:**
- **Durée:** {stats['duration']:.0f}s
- **CPU moyen:** {stats['cpu_mean']:.1f}% (max: {stats['cpu_max']:.1f}%)
- **Mémoire moyenne:** {stats['mem_mean']:.1f}% (max: {stats['mem_max']:.1f}%)
- **Mémoire utilisée:** {stats['mem_gb']:.1f} GB

---
"""

writer.add_text('0_RESUME_GLOBAL', global_summary, 0)

# Comparaisons scalaires ressources
for name in models.keys():
    writer.add_scalars('comparison/cpu_mean', {name: resources_stats[name]['cpu_mean']}, 0)
    writer.add_scalars('comparison/memory_mean', {name: resources_stats[name]['mem_mean']}, 0)

writer.close()

# 8. Sauvegarder infos features
feature_info = {
    'feature_names': feature_columns,
    'n_features': len(feature_columns),
    'models': list(models.keys())
}
joblib.dump(feature_info, os.path.join(MODELS_DIR, "feature_info.pkl"))
print("\n💾 Info features sauvegardées: ", os.path.join(MODELS_DIR, "feature_info.pkl"))

print("\n" + "="*70)
print("📊 RÉSUMÉ FINAL DES PERFORMANCES")
print("="*70)
for name in models.keys():
    stats = resources_stats[name]
    metrics = metrics_stats[name]
    print(f"\n🔹 {name.upper()}")
    print(f"  📈 MÉTRIQUES:")
    print(f"     Accuracy:  {metrics['accuracy']:.3f}")
    print(f"     Precision: {metrics['precision']:.3f}")
    print(f"     Recall:    {metrics['recall']:.3f}")
    print(f"     F1-Score:  {metrics['f1_score']:.3f}")
    print(f"  💻 RESSOURCES:")
    print(f"     CPU moyen: {stats['cpu_mean']:.1f}% (max: {stats['cpu_max']:.1f}%)")
    print(f"     RAM moyenne: {stats['mem_mean']:.1f}% (max: {stats['mem_max']:.1f}%)")
    print(f"     Mémoire: {stats['mem_gb']:.1f} GB")
    print(f"     Durée: {stats['duration']:.0f}s")

print("\n" + "="*70)
print(f"📊 TensorBoard logs sauvegardés dans: {LOGS_DIR}")
print("🚀 Pour visualiser: tensorboard --logdir=" + LOGS_DIR)
print("\n📋 GUIDE TENSORBOARD - Où trouver les COURBES:")
print("  1. Onglet TEXT:")
print("     - '0_RESUME_GLOBAL' : Résumé complet de tous les modèles")
print("     - 'model_resumes/' : Résumé individuel par modèle")
print("  2. Onglet SCALARS (COURBES D'ENTRAÎNEMENT):")
print("     - 'metrics/accuracy/' : Courbes d'accuracy par modèle (20 points)")
print("     - 'metrics/precision/' : Courbes de precision par modèle (20 points)")
print("     - 'metrics/recall/' : Courbes de recall par modèle (20 points)")
print("     - 'metrics/f1_score/' : Courbes de F1-score par modèle (20 points)")
print("     - 'comparison/' : Comparaisons directes entre modèles")
print("     - 'resources/' : Consommation CPU/RAM")
print("  3. Vous verrez maintenant de VRAIES COURBES au lieu de points isolés!")
print("="*70)