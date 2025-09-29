from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import Query
import joblib
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# 🔹 Charger tous les modèles disponibles
models = {}
for name in ["svm", "random_forest", "logistic_regression"]:
    path = os.path.join(MODELS_DIR, f"{name}_iris_model.pkl")
    if os.path.exists(path):
        models[name] = joblib.load(path)

if not models:
    raise RuntimeError("Aucun modèle trouvé dans le dossier models/")
def load_models():
    models = {}
    for file in os.listdir(MODELS_DIR):
        if file.endswith(".pkl"):
            name = file.replace(".pkl", "")
            models[name] = joblib.load(os.path.join(MODELS_DIR, file))
    return models

# 🔹 Charger info features
feature_info = joblib.load(os.path.join(MODELS_DIR, "feature_info.pkl"))
FEATURE_NAMES = feature_info["feature_names"]

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
        mode = request.get("mode", "single") if request else None

        if mode == "single":
            data = np.array([[request[f] for f in FEATURE_NAMES]])
            model = models["svm"]
            prediction = model.predict(data)[0]
            proba = model.predict_proba(data).max()
            return JSONResponse({
                "mode": "single",
                "model_used": "svm",
                "prediction": str(prediction),
                "confidence": round(float(proba), 3)
            })

        elif mode == "compare":
            models = load_models()  
            X = np.array(request["X"])
            y_true = np.array(request["y"])
            if X.shape[1] != len(FEATURE_NAMES):
                raise ValueError(f"Chaque observation doit avoir {len(FEATURE_NAMES)} features.")

            results = {}
            best_acc_model, best_f1_model = None, None
            best_acc, best_f1 = -1, -1

            for name, model in models.items():
                y_pred = model.predict(X)
                acc = accuracy_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred).tolist()
                precision_macro = precision_score(y_true, y_pred, average="macro")
                recall_macro = recall_score(y_true, y_pred, average="macro")
                f1_macro = f1_score(y_true, y_pred, average="macro")

                results[name] = {
                    "accuracy": round(acc,3),
                    "precision_macro": round(precision_macro,3),
                    "recall_macro": round(recall_macro,3),
                    "f1_macro": round(f1_macro,3),
                    "confusion_matrix": cm
                }

                if acc > best_acc: 
                    best_acc, best_acc_model = acc, name
                if f1_macro > best_f1: 
                    best_f1, best_f1_model = f1_macro, name

            # Générer analyse
            class_names = list(np.unique(y_true))
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
#liste des modéle 

METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

@router.get("/list_models")
async def list_models():
    # Charger le fichier metrics
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    # Lister les fichiers .pkl dans le dossier
    models_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    models_info = {}
    for model_file in models_files:
        model_name = model_file.replace(".pkl", "")
        models_info[model_name] = metrics.get(model_name, {})

    return JSONResponse({"models": models_info})
#Ajouter un model préentrainer 


@router.post("/add_model")
async def add_model(file: UploadFile = File(...), accuracy: float = None, f1_macro: float = None):
    if not file.filename.endswith(".pkl"):
        return JSONResponse({"error": "Le fichier doit être un .pkl"}, status_code=400)

    # Sauvegarder le fichier dans models/
    model_path = os.path.join(MODELS_DIR, file.filename)
    with open(model_path, "wb") as f:
        f.write(await file.read())

    # Mettre à jour le fichier metrics
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    model_name = file.filename.replace(".pkl", "")
    metrics[model_name] = {"accuracy": accuracy, "f1_macro": f1_macro}

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    return JSONResponse({"message": f"Modèle {model_name} ajouté avec succès !"})
#Supprimer un modele


@router.delete("/delete_model")
async def delete_model(model_name: str = Query(...)):
    model_file = os.path.join(MODELS_DIR, f"{model_name}.pkl")

    if not os.path.exists(model_file):
        return JSONResponse({"error": "Modèle non trouvé"}, status_code=404)

    os.remove(model_file)

    # Mettre à jour le fichier metrics
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)
        if model_name in metrics:
            metrics.pop(model_name)
            with open(METRICS_FILE, "w") as f:
                json.dump(metrics, f, indent=4)

    return JSONResponse({"message": f"Modèle {model_name} supprimé avec succès !"})
