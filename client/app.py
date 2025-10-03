# app.py (front Streamlit) — version robuste Azure + Docker Compose
import os
import json
import requests
import streamlit as st

# =========================================
# Configuration BACKEND_URL (Azure vs Local)
# =========================================
# Sur Azure App Service, la variable d'env WEBSITE_SITE_NAME est présente.
IS_AZURE = bool(os.getenv("WEBSITE_SITE_NAME"))

# ⚠️ IMPORTANT :
# - En PRODUCTION (Azure), on N'UTILISE PAS de fallback "backend:...".
# - En LOCAL (Docker Compose), le service s'appelle "backend" et écoute 80 dans le conteneur
#   (ton compose mappe 8000:80 côté host, mais ici le front parle via le réseau docker à backend:80).
BACKEND_URL = (
    os.getenv("BACKEND_URL")
    or st.secrets.get("BACKEND_URL", None)
    or (None if IS_AZURE else "http://backend:80")  # fallback uniquement hors Azure
)

if not BACKEND_URL:
    st.set_page_config(page_title="Prédiction Iris & Gestion Modèles", page_icon="🌸", layout="wide")
    st.error(
        "❌ BACKEND_URL n'est pas configuré. "
        "Sur Azure, ajoute-la dans Configuration → Variables d’application de la Web App **front**.\n\n"
        "Exemple : `https://app-jenkis-ml.azurewebsites.net`"
    )
    st.stop()

# Normalise l’URL et construit l’endpoint /predict
BACKEND_URL = BACKEND_URL.strip()
API_URL = BACKEND_URL.rstrip("/") + "/predict"

# (Optionnel) Debug visible si DEBUG=1 en App Settings Azure
DEBUG = os.getenv("DEBUG", "0") == "1"


# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Prédiction Iris & Gestion Modèles",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CSS global (inchangé)
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

    .main .block-container {
        padding-left: 300px;
    }

    button[title="Hide sidebar"] {
        display: none !important;
    }

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
    }

    .stButton>button:hover {
        background-color: #FF3B2E;
    }

    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-height: 60vh;
    }

    .flower-emoji {
        font-size: 120px;
        margin-bottom: 20px;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .main-title {
        text-align:center;
        color:#333;
        font-size: 30px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Barre latérale (Debug doux)
# ===============================
st.sidebar.header("🔧 Connexion API")
st.sidebar.caption("Ces infos n'apparaissent qu'ici. Le design principal reste identique.")
st.sidebar.write("BACKEND_URL =", BACKEND_URL)
st.sidebar.write("API_URL =", API_URL)

if DEBUG:
    # Petit test à la demande (ne modifie pas la page principale)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🩺 Test rapide /predict")
    if st.sidebar.button("Tester /predict avec un échantillon Iris"):
        # ⚠️ Ton backend attend ces clés EXACTES (côté routes.py)
        sample = {
            "mode": "single",
            "SepalLengthCm": 5.1,
            "SepalWidthCm": 3.5,
            "PetalLengthCm": 1.4,
            "PetalWidthCm": 0.2
        }
        try:
            r = requests.post(API_URL, json=sample, timeout=10)
            st.sidebar.write("Status:", r.status_code)
            # Essaie de parser en JSON, sinon affiche le texte brut
            try:
                st.sidebar.json(r.json())
            except Exception:
                st.sidebar.code(r.text)
        except Exception as e:
            st.sidebar.error(f"Erreur appel API: {e}")

# ===============================
# Contenu principal (inchangé)
# ===============================
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
            <li>⚙️ <b>Gérer</b> vos modèles de Machine Learning (ajout, suppression, liste).</li>
            <li>📊 <b>Comparer</b> différents modèles et leurs performances.</li>
        </ul>
        <p style='font-size: 16px; color: #777; margin-top:20px;'>
        Utilisez le menu latéral pour naviguer entre les sections : Observation, Gestion et Comparaison.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# (Option léger) Aide contextuelle
# ===============================
with st.expander("ℹ️ Aide connexion API", expanded=False):
    st.write("""
- En **Azure**, configure `BACKEND_URL` dans *Configuration → Variables d’application* de la Web App **front**.
  - Exemple : `https://app-jenkis-ml.azurewebsites.net`
- En **local (Docker Compose)**, le fallback utilise `http://backend:80`.
- Si tu vois encore `backend:...` en Azure, c’est que `BACKEND_URL` n’est pas défini ou que l’app n’a pas redémarré.
- Côté **backend (FastAPI)**, n’oublie pas d’activer **CORS** avec
  `allow_origins=["https://app-jenkis-ml-front.azurewebsites.net", "http://localhost:8502"]`.
    """)
