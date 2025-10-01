from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import Query
import joblib
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

router = APIRouter()

# 🔹 Correction : __file__ au lieu de _file_
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 🔹 S'assurer que le dossier models existe
os.makedirs(MODELS_DIR, exist_ok=True)

# 🔹 Charger tous les modèles disponibles au démarrage
MODELS_CACHE = {}
for name in ["svm", "random_forest", "logistic_regression"]:
    path = os.path.join(MODELS_DIR, f"{name}_iris_model.pkl")
    if os.path.exists(path):
        MODELS_CACHE[name] = joblib.load(path)

# 🔹 Fonction pour charger dynamiquement les modèles
def load_all_models():
    loaded_models = {}
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            if file.endswith(".pkl") and not file.startswith("feature_info"):
                name = file.replace("_iris_model.pkl", "").replace(".pkl", "")
                loaded_models[name] = joblib.load(os.path.join(MODELS_DIR, file))
    return loaded_models

# 🔹 Charger info features (avec gestion d'erreur)
FEATURE_NAMES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
feature_info_path = os.path.join(MODELS_DIR, "feature_info.pkl")
if os.path.exists(feature_info_path):
    try:
        feature_info = joblib.load(feature_info_path)
        FEATURE_NAMES = feature_info.get("feature_names", FEATURE_NAMES)
    except:
        pass

# 🔹 Générer une analyse automatique
def generate_analysis(results, best_acc_model, best_f1_model, class_names=None):
    lines = []
    # Meilleurs modèles
    lines.append(f"🔹 Meilleur modèle par accuracy: {best_acc_model.upper()} ({results[best_acc_model]['accuracy']})")
    lines.append(f"🔹 Meilleur modèle par F1-macro: {best_f1_model.upper()} ({results[best_f1_model]['f1_macro']})")
    if best_acc_model == best_f1_model:
        lines.append("✅ Ce modèle est aussi le plus robuste globalement.")
    else:
        lines.append("⚖️ Les deux mesures divergent, à considérer selon l'objectif (accuracy vs équilibre précision/rappel).")
    
    # Analyse des erreurs par classe
    for model, metrics in results.items():
        cm = np.array(metrics["confusion_matrix"])
        n_classes = cm.shape[0]
        total_errors = cm.sum() - cm.trace()
        if total_errors > 0:
            lines.append(f"\n⚠️ {model.upper()} a {total_errors} erreurs au total:")
            if class_names:
                for i in range(n_classes):
                    for j in range(n_classes):
                        if i != j and cm[i, j] > 0:
                            lines.append(f"   - {class_names[i]} confondu avec {class_names[j]}: {cm[i,j]} fois")
    return "\n".join(lines)

# 🔹 Prédiction simple
@router.post("/predict")
async def predict(request: dict = None):
    try:
        if not request:
            raise ValueError("Request body manquant")
            
        mode = request.get("mode", "single")

        if mode == "single":
            # Utiliser les noms de features définis
            data = np.array([[request.get(f, 0.0) for f in FEATURE_NAMES]])
            
            # Utiliser le premier modèle disponible
            if not MODELS_CACHE:
                raise ValueError("Aucun modèle disponible")
                
            model_name = list(MODELS_CACHE.keys())[0]
            model = MODELS_CACHE[model_name]
            
            prediction = model.predict(data)[0]
            proba = model.predict_proba(data).max() if hasattr(model, 'predict_proba') else 1.0
            
            return JSONResponse({
                "mode": "single",
                "model_used": model_name,
                "prediction": str(prediction),
                "confidence": round(float(proba), 3)
            })

        elif mode == "compare":
            # Charger tous les modèles disponibles
            all_models = load_all_models()  
            
            if not all_models:
                raise ValueError("Aucun modèle disponible pour la comparaison")
                
            if "X" not in request or "y" not in request:
                raise ValueError("Les données 'X' et 'y' sont requises pour la comparaison")
                
            X = np.array(request["X"])
            y_true = np.array(request["y"])
            
            if X.shape[1] != len(FEATURE_NAMES):
                raise ValueError(f"Chaque observation doit avoir {len(FEATURE_NAMES)} features.")

            results = {}
            best_acc_model, best_f1_model = None, None
            best_acc, best_f1 = -1, -1

            for name, model in all_models.items():
                y_pred = model.predict(X)
                acc = accuracy_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred).tolist()
                precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
                recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
                f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

                results[name] = {
                    "accuracy": round(acc, 3),
                    "precision_macro": round(precision_macro, 3),
                    "recall_macro": round(recall_macro, 3),
                    "f1_macro": round(f1_macro, 3),
                    "confusion_matrix": cm
                }

                if acc > best_acc: 
                    best_acc, best_acc_model = acc, name
                if f1_macro > best_f1: 
                    best_f1, best_f1_model = f1_macro, name

            # Générer analyse
            class_names = list(map(str, np.unique(y_true)))
            analysis = generate_analysis(results, best_acc_model, best_f1_model, class_names)

            return JSONResponse({
                "mode": "compare",
                "metrics": results,
                "best_models": {
                    "by_accuracy": best_acc_model,
                    "by_f1_macro": best_f1_model
                },
                "analysis": analysis
            })

        else:
            raise ValueError("Mode invalide. Choisir 'single' ou 'compare'.")

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# 🔹 Comparaison via upload fichier JSON
@router.post("/compare_file")
async def compare_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        data = json.loads(contents)
        if "X" not in data or "y" not in data:
            raise ValueError("Le fichier JSON doit contenir 'X' et 'y'.")
        data["mode"] = "compare"
        return await predict(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# Liste des modèles
METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

@router.get("/list_models")
async def list_models():
    # Charger le fichier metrics
    metrics = {}
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                metrics = json.load(f)
        except:
            pass

    # Lister les fichiers .pkl dans le dossier
    models_info = {}
    if os.path.exists(MODELS_DIR):
        models_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl") and not f.startswith("feature_info")]
        for model_file in models_files:
            model_name = model_file.replace("_iris_model.pkl", "").replace(".pkl", "")
            models_info[model_name] = metrics.get(model_name, {})

    return JSONResponse({"models": models_info})

# Ajouter un modèle préentrainé
@router.post("/add_model")
async def add_model(file: UploadFile = File(...), accuracy: float = None, f1_macro: float = None):
    if not file.filename.endswith(".pkl"):
        return JSONResponse({"error": "Le fichier doit être un .pkl"}, status_code=400)

    # Sauvegarder le fichier dans models/
    model_path = os.path.join(MODELS_DIR, file.filename)
    with open(model_path, "wb") as f:
        f.write(await file.read())

    # Mettre à jour le fichier metrics
    metrics = {}
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                metrics = json.load(f)
        except:
            pass

    model_name = file.filename.replace("_iris_model.pkl", "").replace(".pkl", "")
    metrics[model_name] = {"accuracy": accuracy, "f1_macro": f1_macro}

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    return JSONResponse({"message": f"Modèle {model_name} ajouté avec succès !"})

# Supprimer un modèle
@router.delete("/delete_model")
async def delete_model(model_name: str = Query(...)):
    # Essayer différents formats de noms de fichiers
    possible_files = [
        f"{model_name}.pkl",
        f"{model_name}_iris_model.pkl"
    ]
    
    model_file = None
    for filename in possible_files:
        file_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(file_path):
            model_file = file_path
            break

    if not model_file:
        return JSONResponse({"error": "Modèle non trouvé"}, status_code=404)

    os.remove(model_file)

    # Mettre à jour le fichier metrics
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                metrics = json.load(f)
            if model_name in metrics:
                metrics.pop(model_name)
                with open(METRICS_FILE, "w") as f:
                    json.dump(metrics, f, indent=4)
        except:
            pass

    return JSONResponse({"message": f"Modèle {model_name} supprimé avec succès !"})