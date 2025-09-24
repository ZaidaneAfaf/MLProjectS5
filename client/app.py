import streamlit as st
import requests

# ===============================
# Config API URL (Docker: "serveur")
# ===============================
API_URL = "http://serveur:8000/predict"

# ===============================
# Custom CSS
# ===============================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #f0f4f8, #d9e2ec);
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #FF6F61;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF3B2E;
    }
    h1 {
        color: #333;
    }
    .stSlider>div>div>div>div {
        color: #FF6F61;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Titre et description
# ===============================
st.markdown("<h1 style='text-align:center;'>🌸 Prédiction Iris 🌸</h1>", unsafe_allow_html=True)
st.markdown(
    """
    Entrez les mesures des sépales et des pétales pour prédire l'espèce d'iris.  
    Cliquez sur **Prédiction** pour voir le résultat.
    """,
    unsafe_allow_html=True
)

# ===============================
# Sliders pour les mesures
# ===============================
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Longueur des sépales (cm)", 0.0, 10.0, 5.0)
    sepal_width = st.slider("Largeur des sépales (cm)", 0.0, 10.0, 3.5)

with col2:
    petal_length = st.slider("Longueur des pétales (cm)", 0.0, 10.0, 4.0)
    petal_width = st.slider("Largeur des pétales (cm)", 0.0, 10.0, 1.3)

# ===============================
# Bouton de prédiction
# ===============================
if st.button("Prédiction"):
    with st.spinner("Calcul en cours..."):
        data = {
            "SepalLengthCm": sepal_length,
            "SepalWidthCm": sepal_width,
            "PetalLengthCm": petal_length,
            "PetalWidthCm": petal_width
        }
        try:
            response = requests.post(API_URL, json=data)
            res_json = response.json()

            if "error" in res_json:
                st.error(f"Erreur API: {res_json['error']}")
            else:
                species_name = res_json.get("prediction")
                confidence = res_json.get("confidence")

                st.markdown(
                    f"""
                    <div style='text-align: center; margin-top: 20px;'>
                        <h2 style='color:#FF6F61;'>🌼 Fleur prédite: {species_name}</h2>
                        <p>Confiance: {confidence}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Erreur lors de l'appel API : {e}")
