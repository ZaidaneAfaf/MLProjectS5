import streamlit as st

API_URL = "http://backend:8000/predict"

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

    /* Force le texte en noir */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, 
    .stMarkdown h2, .stMarkdown h3, .stMarkdown div, 
    p, span, div, li, ul {
        color: #333 !important;
    }

    .centered-content * {
        color: #333 !important;
    }

    .centered-content b {
        color: #FF6F61 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Contenu principal
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
        Utilisez le menu latéral pour naviguer entre les sections :Observation , Gestion et Comparaison.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)