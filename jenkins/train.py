# train_with_monitoring.py
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import psutil
import threading
from datetime import datetime

# TensorBoard avec monitoring des ressources
try:
    from torch.utils.tensorboard import SummaryWriter
    import matplotlib.pyplot as plt
    TENSORBOARD_AVAILABLE = True
    print("✅ TensorBoard disponible")
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️ TensorBoard non disponible")

# 🔹 Répertoire de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Iris.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
TENSORBOARD_DIR = os.path.join(BASE_DIR, "..", "artifacts", "tensorboard")
TEST_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "iris_test.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# 🔹 Classe pour monitorer les ressources système
class ResourceMonitor:
    def __init__(self, writer, model_name):
        self.writer = writer
        self.model_name = model_name
        self.monitoring = False
        self.data = []
        
    def start_monitoring(self):
        """Démarre le monitoring des ressources"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Boucle de monitoring"""
        step = 0
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.5)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                memory_available_gb = memory.available / (1024**3)
                
                # Disk usage (optionnel)
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                
                # Enregistrement dans TensorBoard
                if self.writer:
                    self.writer.add_scalar("Resources/CPU_Usage", cpu_percent, step)
                    self.writer.add_scalar("Resources/Memory_Percent", memory_percent, step)
                    self.writer.add_scalar("Resources/Memory_Used_GB", memory_used_gb, step)
                    self.writer.add_scalar("Resources/Memory_Available_GB", memory_available_gb, step)
                    self.writer.add_scalar("Resources/Disk_Usage", disk_percent, step)
                
                # Stockage des données pour analyse
                self.data.append({
                    'timestamp': datetime.now(),
                    'step': step,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_used_gb': memory_used_gb,
                    'memory_available_gb': memory_available_gb
                })
                
                step += 1
                time.sleep(2)  # Mesure toutes les 2 secondes
                
            except Exception as e:
                print(f"⚠️ Erreur monitoring: {e}")
                break
    
    def get_summary(self):
        """Retourne un résumé des ressources utilisées"""
        if not self.data:
            return "Aucune donnée de monitoring"
            
        df = pd.DataFrame(self.data)
        summary = f"""
📊 RÉSUMÉ RESSOURCES - {self.model_name.upper()}
⏱️  Durée monitoring: {len(self.data) * 2} secondes
💻 CPU moyen: {df['cpu_percent'].mean():.1f}% (max: {df['cpu_percent'].max():.1f}%)
🧠 Mémoire moyenne: {df['memory_percent'].mean():.1f}% (max: {df['memory_percent'].max():.1f}%)
💾 Mémoire utilisée: {df['memory_used_gb'].mean():.2f} GB
        """
        return summary

# 🔹 Chargement des données
iris = pd.read_csv(DATA_PATH)
print(f"📊 Dataset: {iris.shape}, colonnes: {list(iris.columns)}")

# 🔹 Sélection des features
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
target_column = 'Species'

X = iris[feature_columns].copy()
y = iris[target_column].copy()

print(f"✅ Features: {X.shape}, Target: {y.name}")
print(f"Classes: {y.unique()}")

# 🔹 Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 🔹 Sauvegarde du jeu de test
test_data = {
    "X": X_test.values.tolist(),
    "y": y_test.tolist()
}

with open(TEST_DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)

print(f"💾 Jeu de test sauvegardé: {TEST_DATA_PATH}")

# 🔹 Définition des modèles
models = {
    "svm": SVC(probability=True, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=200, random_state=42, multi_class="multinomial")
}

results = {}
resource_summaries = {}

# 🔹 Entraînement avec monitoring des ressources
for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"🚀 Entraînement du modèle: {name}")
    print(f"{'='*60}")
    
    # Initialisation TensorBoard et monitoring
    writer = None
    monitor = None
    
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, name))
        monitor = ResourceMonitor(writer, name)
        print("🔍 Démarrage du monitoring des ressources...")
        monitor.start_monitoring()
    
    start_time = time.time()
    
    try:
        # Entraînement progressif avec monitoring
        train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        for i, train_size in enumerate(train_sizes):
            n_samples = int(len(X_train) * train_size)
            X_train_sub = X_train.iloc[:n_samples]
            y_train_sub = y_train.iloc[:n_samples]
            
            # Entraînement intermédiaire
            model_temp = type(model)(**model.get_params())
            model_temp.fit(X_train_sub, y_train_sub)
            
            # Métriques
            y_train_pred = model_temp.predict(X_train_sub)
            y_test_pred = model_temp.predict(X_test)
            
            train_acc = accuracy_score(y_train_sub, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            if writer:
                writer.add_scalar("Training/train_accuracy", train_acc, i)
                writer.add_scalar("Training/test_accuracy", test_acc, i)
                writer.add_scalar("Training/train_size", train_size, i)
                writer.add_scalars("Accuracy/Comparison", 
                                 {'train': train_acc, 'test': test_acc}, i)
                
                # Métriques temporelles
                elapsed_time = time.time() - start_time
                writer.add_scalar("Performance/elapsed_time_seconds", elapsed_time, i)
            
            print(f"📈 Étape {i+1}/{len(train_sizes)}: size={train_size:.1%}, Train={train_acc:.3f}, Test={test_acc:.3f}")
        
        # Entraînement final
        model.fit(X_train, y_train)
        total_time = time.time() - start_time
        
        # Prédictions finales
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"✅ Accuracy finale: {accuracy:.3f}")
        print(f"⏱️ Temps total: {total_time:.2f}s")
        
        if writer:
            writer.add_scalar("Performance/final_accuracy", accuracy, len(train_sizes))
            writer.add_scalar("Performance/total_training_time", total_time, len(train_sizes))
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            for cls, metrics in report.items():
                if isinstance(metrics, dict) and cls not in ['accuracy', 'macro avg', 'weighted avg']:
                    writer.add_scalar(f"Classes/{cls}/precision", metrics['precision'], len(train_sizes))
                    writer.add_scalar(f"Classes/{cls}/recall", metrics['recall'], len(train_sizes))
                    writer.add_scalar(f"Classes/{cls}/f1-score", metrics['f1-score'], len(train_sizes))
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Matrice de confusion:\n{cm}")
        
        if writer and TENSORBOARD_AVAILABLE:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                
                classes = model.classes_
                ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
                       xticklabels=classes, yticklabels=classes,
                       xlabel="Classe prédite", ylabel="Vraie classe",
                       title=f"Matrice de Confusion - {name}")
                
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
                
                fig.tight_layout()
                writer.add_figure("Confusion_Matrix/final", fig, len(train_sizes))
                plt.close(fig)
            except Exception as e:
                print(f"⚠️ Erreur matrice de confusion: {e}")
        
    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
    
    finally:
        # Arrêt du monitoring et sauvegarde du résumé
        if monitor:
            monitor.stop_monitoring()
            summary = monitor.get_summary()
            resource_summaries[name] = summary
            print(summary)
            
            if writer:
                writer.add_text("Resources/summary", summary, len(train_sizes))
        
        # Sauvegarde du modèle
        model_path = os.path.join(MODELS_DIR, f"{name}_iris_model.pkl")
        joblib.dump(model, model_path)
        print(f"💾 Modèle sauvegardé: {model_path}")
        
        if writer:
            writer.flush()
            writer.close()

# 🔹 Sauvegarde des infos features et résumés des ressources
feature_info = {
    'feature_names': feature_columns,
    'n_features': len(feature_columns),
    'models': list(models.keys()),
    'results': results,
    'resource_summaries': resource_summaries
}
feature_info_path = os.path.join(MODELS_DIR, "feature_info.pkl")
joblib.dump(feature_info, feature_info_path)
print(f"\n💾 Info features et ressources sauvegardées: {feature_info_path}")

# 🔹 Résumé final
print(f"\n{'='*60}")
print("📊 RÉSUMÉ DES PERFORMANCES")
print(f"{'='*60}")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:20s}: {accuracy:.3f}")

print(f"\n{'='*60}")
print("📊 CONSOMMATION DES RESSOURCES")
print(f"{'='*60}")
for name, summary in resource_summaries.items():
    print(f"\n🔍 {name.upper()}:")
    print(summary)

if TENSORBOARD_AVAILABLE:
    print(f"\n{'='*60}")
    print("🌐 TENSORBOARD - MONITORING COMPLET")
    print(f"{'='*60}")
    print(f"Commande: tensorboard --logdir {TENSORBOARD_DIR} --port 6006")
    print(f"URL: http://localhost:6006")
    print("\n📊 Données disponibles par modèle:")
    print("  - Training: courbes train/test accuracy")
    print("  - Performance: accuracy finale, temps d'entraînement")
    print("  - Resources: CPU, mémoire, disque en temps réel")
    print("  - Classes: precision, recall, f1-score par classe")
    print("  - Confusion_Matrix: matrice de confusion")

print("\n✅ Entraînement et monitoring terminés avec succès!")