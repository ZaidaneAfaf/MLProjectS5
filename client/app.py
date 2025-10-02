# client/app.py
import os
import json
import time
from typing import Dict, Any, List, Optional

import requests
import streamlit as st

# ===============================
# Config générale
# ===============================
st.set_page_config(
    page_title="Prédiction Iris & Gestion Modèles",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Backend (Azure) : BACKEND_URL
#   - En local (docker-compose) : http://backend:8000
#   - En Azure (Web App)       : https://app-jenkis-ml.azurewebsites.net
# -------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
HEALTH_URL = f"{BACKEND_URL}/health"
PREDICT_URL = f"{BACKEND_URL}/predict"
MODELS_URL = f"{BACKEND_URL}/models"
FEATURE_INFO_URL = f"{BACKEND_URL}/feature-info"

REQUEST_TIMEOUT = 15  # secondes
DEFAULT_MODELS = ["auto", "svm", "random_forest", "logistic_regression"]
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]


# ===============================
# CSS global
# ===============================
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}

    section[data-testid="stSidebar"] {
        visibility: visible !important;
        transform: none !important;
        position: fixed !important;
        height: 100vh !important;
    }
    .main .block-container { padding-left: 300px; }

    button[title="Hide sidebar"] { display: none !important; }

    .stApp {
        background: linear-gradient(to right, #f0f4f8, #d9e2ec);
        font-family: 'Arial', sans-serif;
    }

    .stButton>button {
        background-color: #FF6F61;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        margin-bottom: 10px;
        cursor: pointer;
    }
    .stButton>button:hover { background-color: #FF3B2E; }

    .centered-content {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; text-align: center; min-height: 50vh;
    }
    .flower-emoji {
        font-size: 120px; margin-bottom: 20px; animation: float 3s ease-in-out infinite;
    }
    @keyframes float { 0%,100% {transform:translateY(0)} 50% {transform:translateY(-10px)} }

    .main-title { text-align:center; color:#333; font-size:30px; margin-bottom:20px; }
    .small { color:#666; font-size: 0.9rem; }
    .ok { color: #0a7f2e; font-weight: 600; }
    .ko { color: #b00020; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===============================
# Helpers HTTP
# ===============================
def _get_json(url: str) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except Exception:
                return {"status": resp.status_code, "text": resp.text}
        return {"error": f"Status {resp.status_code}", "text": resp.text}
    except requests.RequestException as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unknown error: {e}"}

def probe_health() -> Dict[str, Any]:
    """Essaye /health, puis /, puis /docs (retourne le premier résultat reçu)."""
    for path in ("/health", "/", "/docs"):
        data = _get_json(f"{BACKEND_URL}{path}")
        if data and not data.get("error"):
            return {"ok": True, "path": path, "data": data}
    # Dernière tentative brute pour / et /docs (peuvent renvoyer HTML)
    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return {"ok": True, "path": "/", "data": {"status": r.status_code, "text": "OK"}}
    except Exception:
        pass
    return {"ok": False}

def get_models() -> List[str]:
    """Essaie de récupérer la liste des modèles depuis l’API, sinon valeurs par défaut."""
    data = _get_json(MODELS_URL)
    if isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
        return ["auto"] + data["models"]
    # fallback: peut-être dans /feature-info
    data2 = _get_json(FEATURE_INFO_URL)
    if isinstance(data2, dict) and "models" in data2 and isinstance(data2["models"], list):
        return ["auto"] + data2["models"]
    return DEFAULT_MODELS

def try_predict(payload_variants: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Teste plusieurs formes de payload pour s’adapter au schéma du backend."""
    errors = []
    for payload in payload_variants:
        try:
            r = requests.post(PREDICT_URL, json=payload, timeout=REQUEST_TIMEOUT)
            if 200 <= r.status_code < 300:
                return {"ok": True, "payload_used": payload, "response": r.json()}
            else:
                errors.append(f"{r.status_code}: {r.text[:300]}")
        except requests.RequestException as e:
            errors.append(str(e))
    return {"ok": False, "errors": errors}

def build_payloads(values: List[float], chosen_model: Optional[str]) -> List[Dict[str, Any]]:
    """Construit plusieurs variantes de payload pour maximiser la compatibilité."""
    variant_list = []

    # 1) Noms de champs explicites (très fréquent)
    named = {
        "SepalLengthCm": values[0],
        "SepalWidthCm":  values[1],
        "PetalLengthCm": values[2],
        "PetalWidthCm":  values[3],
    }
    if chosen_model and chosen_model != "auto":
        # plusieurs serveurs utilisent model ou model_name ; on essaie les deux
        variant_list.append({**named, "model": chosen_model})
        variant_list.append({**named, "model_name": chosen_model})
    variant_list.append(named)

    # 2) Tableau features (autre style fréquent)
    base_list = {"features": values}
    if chosen_model and chosen_model != "auto":
        variant_list.append({**base_list, "model": chosen_model})
        variant_list.append({**base_list, "model_name": chosen_model})
    variant_list.append(base_list)

    # 3) Matrice X (certains backends attendent X: [[...]])
    base_X = {"X": [values]}
    if chosen_model and chosen_model != "auto":
        variant_list.append({**base_X, "model": chosen_model})
        variant_list.append({**base_X, "model_name": chosen_model})
    variant_list.append(base_X)

    return variant_list


# ===============================
# Sidebar (navigation + diagnostics)
# ===============================
st.sidebar.title("🌸 Iris App")
st.sidebar.caption(f"Backend: `{BACKEND_URL}`")

# Test santé
health = probe_health()
if health["ok"]:
    st.sidebar.markdown(f"**Santé API:** <span class='ok'>OK</span> (`{health['path']}`)", unsafe_allow_html=True)
else:
    st.sidebar.markdown("**Santé API:** <span class='ko'>KO</span>", unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Prédire", "Gestion", "Comparaison", "Diagnostics"],
    index=0,
)

# ===============================
# Pages
# ===============================

# ---- Accueil
if page == "Accueil":
    st.markdown("<h1 class='main-title'>🌸 Prédiction des Fleurs d'Iris & Gestion des Modèles</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="centered-content">
            <div class="flower-emoji">🌺</div>
            <h2 style='color: #333; margin-bottom: 20px;'>Bienvenue sur votre application</h2>
            <p style='font-size: 18px; color: #666; max-width: 700px;'>
              Cette plateforme vous permet de :
            </p>
            <ul style='font-size: 18px; color: #444; text-align:left; max-width:700px; margin:0 auto;'>
                <li>🌼 <b>Prédire</b> l'espèce d'une fleur d'iris à partir de ses mesures.</li>
                <li>⚙️ <b>Gérer</b> vos modèles de Machine Learning (si exposés par l'API).</li>
                <li>📊 <b>Comparer</b> rapidement différents réglages.</li>
            </ul>
            <p class='small' style='margin-top:8px;'>Utilisez le menu latéral pour naviguer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Prédire
elif page == "Prédire":
    st.header("🔮 Prédire l'espèce d'iris")

    colL, colR = st.columns([1, 1], gap="large")
    with colL:
        st.subheader("Mesures (cm)")
        sepal_length = st.number_input("SepalLengthCm", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_width  = st.number_input("SepalWidthCm",  min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        petal_length = st.number_input("PetalLengthCm", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width  = st.number_input("PetalWidthCm",  min_value=0.0, max_value=10.0, value=0.2, step=0.1)

        models = get_models()
        model_choice = st.selectbox("Modèle (si supporté par l'API)", models, index=0)

        if st.button("Prédire"):
            values = [sepal_length, sepal_width, petal_length, petal_width]

            with st.spinner("Appel API /predict…"):
                payloads = build_payloads(values, model_choice)
                result = try_predict(payloads)

            if result["ok"]:
                st.success("Prédiction réussie ✅")
                st.json(result["response"])
                with st.expander("Payload effectivement utilisé"):
                    st.code(json.dumps(result["payload_used"], indent=2, ensure_ascii=False), language="json")
            else:
                st.error("Échec d'appel API /predict ❌")
                st.write("Erreurs renvoyées par les différentes tentatives :")
                st.code("\n".join(result.get("errors", [])) or "Aucune trace", language="text")

    with colR:
        st.subheader("Aide")
        st.markdown(
            """
            - Le backend est cherché à l’URL:  
              `BACKEND_URL = """ + BACKEND_URL + """`
            - En **Azure**, assurez-vous que la Web App *front* a bien cette variable :
              `BACKEND_URL=https://app-jenkis-ml.azurewebsites.net`
            - Le port Streamlit doit être **8501** (WEBSITES_PORT = 8501).
            """
        )

# ---- Gestion (basée sur endpoints facultatifs)
elif page == "Gestion":
    st.header("⚙️ Gestion des modèles")
    st.caption("Cette page s’appuie sur des endpoints facultatifs (/models, /feature-info).")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Modèles disponibles (API)")
        ms = get_models()
        st.write(ms)

    with c2:
        st.subheader("Infos features (API)")
        finfo = _get_json(FEATURE_INFO_URL)
        if finfo and not finfo.get("error"):
            st.json(finfo)
        else:
            st.info("Endpoint /feature-info non disponible sur l’API.")

# ---- Comparaison simple (juste plusieurs appels /predict)
elif page == "Comparaison":
    st.header("📊 Comparaison rapide")
    st.caption("Envoie plusieurs prédictions avec différentes combinaisons (si l’API supporte la sélection de modèle).")

    values = st.text_input(
        "Entrez 4 valeurs (SepalLength, SepalWidth, PetalLength, PetalWidth) séparées par des virgules",
        "7.3,2.9,6.3,1.8",
    )
    try:
        arr = [float(x.strip()) for x in values.split(",")]
    except Exception:
        st.error("Format invalide. Exemple: 7.3,2.9,6.3,1.8")
        arr = None

    chosen = st.multiselect("Modèles à tester", get_models(), default=["svm", "random_forest", "logistic_regression"])

    if st.button("Lancer comparaison") and arr and len(arr) == 4:
        table = []
        with st.spinner("Appels en cours…"):
            for m in chosen:
                payloads = build_payloads(arr, m)
                r = try_predict(payloads)
                table.append({
                    "model": m,
                    "ok": r["ok"],
                    "prediction": None if not r["ok"] else r["response"],
                    "error": None if r["ok"] else "\n".join(r.get("errors", [])[:1]),
                })
        st.write(table)

# ---- Diagnostics
elif page == "Diagnostics":
    st.header("🛠️ Diagnostics")
    st.write(f"**BACKEND_URL**: `{BACKEND_URL}`")
    st.write(f"**HEALTH_URL**: `{HEALTH_URL}`")
    st.write(f"**PREDICT_URL**: `{PREDICT_URL}`")
    st.write(f"**MODELS_URL**: `{MODELS_URL}`")
    st.write(f"**FEATURE_INFO_URL**: `{FEATURE_INFO_URL}`")

    st.subheader("Santé API")
    st.json(health)

    st.subheader("Essai GET /health brut")
    st.json(_get_json(HEALTH_URL))
