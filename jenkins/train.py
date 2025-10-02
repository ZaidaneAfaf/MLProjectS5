import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import json
import psutil
import threading
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 🔹 Répertoire de base (où est le script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 Chemins absolus pour les données et modèles
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Iris.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
LOGS_DIR = os.path.join(BASE_DIR, "..", "logs", "tensorboard")
os.makedirs(MODELS_DIR, exist_ok=True)
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

# TensorBoard Writer
writer = SummaryWriter(log_dir=LOGS_DIR)

print("\n" + "="*70)
print("💻 CONSOMMATION CPU/RAM PAR MODÈLE")
print("="*70)

# 7. Entraînement et évaluation avec monitoring
for idx, (name, model) in enumerate(models.items()):
    print(f"\n🚀 Entraînement du modèle: {name}")
    
    # Démarrer le monitoring
    monitor = ResourceMonitor(name)
    monitor.start()
    
    # Entraînement
    model.fit(X_train, y_train)
    
    # Arrêter le monitoring
    stats = monitor.stop()
    resources_stats[name] = stats
    
    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"✅ {name} Accuracy: {accuracy:.3f}")
    
    # 📊 Écriture dans TensorBoard - RÉSUMÉ CPU/RAM
    resume_text = f"""## 📊 RÉSUMÉ RESSOURCES - {name.upper()}

⏱️  **Durée monitoring:** {stats['duration']:.0f} secondes  
💻 **CPU moyen:** {stats['cpu_mean']:.1f}% (max: {stats['cpu_max']:.1f}%)  
🧠 **Mémoire moyenne:** {stats['mem_mean']:.1f}% (max: {stats['mem_max']:.1f}%)  
💾 **Mémoire utilisée:** {stats['mem_gb']:.3f} GB  
✅ **Accuracy:** {accuracy:.3f}
"""
    writer.add_text(f'Resume_Ressources/{name}', resume_text, 0)
    
    # Métriques scalaires pour graphiques
    writer.add_scalar(f'Metrics/accuracy', accuracy, idx)
    writer.add_scalar(f'Resources_CPU/mean', stats['cpu_mean'], idx)
    writer.add_scalar(f'Resources_CPU/max', stats['cpu_max'], idx)
    writer.add_scalar(f'Resources_Memory/mean_percent', stats['mem_mean'], idx)
    writer.add_scalar(f'Resources_Memory/max_percent', stats['mem_max'], idx)
    writer.add_scalar(f'Resources_Memory/gb', stats['mem_gb'], idx)
    writer.add_scalar(f'Training/duration_seconds', stats['duration'], idx)
    
    # Test rapide - SÉPARÉ dans un autre tag
    test_data_sample = X_test.iloc[0:1]
    pred = model.predict(test_data_sample)[0]
    proba = model.predict_proba(test_data_sample).max()
    
    test_text = f"""## 🧪 Test Prédiction - {name.upper()}

**Input:** {test_data_sample.values.tolist()}  
**Prédiction:** {pred}  
**Confiance:** {proba:.3f}
"""
    writer.add_text(f'Tests_Predictions/{name}', test_text, 0)

    print(f"🧪 Test {name}:")
    print(f"Input: {test_data_sample.values}")
    print(f"Prédiction: {pred}, Confiance: {proba:.3f}")

    # Sauvegarde du modèle
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}_iris_model.pkl"))
    print(f"💾 Modèle sauvegardé: {os.path.join(MODELS_DIR, f'{name}_iris_model.pkl')}")

# 📊 RÉSUMÉ GLOBAL - Onglet séparé
global_summary = """# 📊 RÉSUMÉ GLOBAL - TOUS LES MODÈLES

## Comparaison des Performances

"""

for name in models.keys():
    stats = resources_stats[name]
    acc = results[name]
    global_summary += f"""
### 🔹 {name.upper()}
- **Accuracy:** {acc:.3f}  
- **CPU moyen:** {stats['cpu_mean']:.1f}% (max: {stats['cpu_max']:.1f}%)  
- **Mémoire moyenne:** {stats['mem_mean']:.1f}% (max: {stats['mem_max']:.1f}%)  
- **Mémoire utilisée:** {stats['mem_gb']:.3f} GB  
- **Durée:** {stats['duration']:.0f}s

---
"""

writer.add_text('RESUME_GLOBAL/comparaison', global_summary, 0)

# Scalaires de comparaison groupés
for idx, name in enumerate(models.keys()):
    writer.add_scalar('Comparison/CPU_mean', resources_stats[name]['cpu_mean'], idx)
    writer.add_scalar('Comparison/Memory_percent', resources_stats[name]['mem_mean'], idx)
    writer.add_scalar('Comparison/Accuracy', results[name], idx)

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
for name, acc in results.items():
    stats = resources_stats[name]
    print(f"\n🔹 {name.upper()}")
    print(f"  ✅ Accuracy: {acc:.3f}")
    print(f"  💻 CPU moyen: {stats['cpu_mean']:.1f}% (max: {stats['cpu_max']:.1f}%)")
    print(f"  🧠 RAM moyenne: {stats['mem_mean']:.1f}% (max: {stats['mem_max']:.1f}%)")
    print(f"  💾 Mémoire: {stats['mem_gb']:.3f} GB")
    print(f"  ⏱️  Durée: {stats['duration']:.0f}s")

print("\n" + "="*70)
print(f"📊 TensorBoard logs sauvegardés dans: {LOGS_DIR}")
print("🚀 Pour visualiser: tensorboard --logdir=" + LOGS_DIR)
print("\n📋 Dans TensorBoard, consultez:")
print("  - Onglet TEXT > Resume_Ressources/* : CPU/RAM par modèle")
print("  - Onglet TEXT > RESUME_GLOBAL/comparaison : Vue d'ensemble")
print("  - Onglet TEXT > Tests_Predictions/* : Prédictions tests")
print("  - Onglet SCALARS : Graphiques et comparaisons")
print("="*70)