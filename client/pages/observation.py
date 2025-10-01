import streamlit as st
import requests
import os

# ===============================
# Config API URL
# ===============================
API_URL = "http://backend:8000/predict"

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
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #FF3B2E;
    }
    h1 {
        color: #333;
        text-align: center;
        font-size: 24px;
    }
    .stSlider>div>div>div>div {
        color: #FF6F61;
    }
    .result-container {
        text-align: center;
        margin: 20px 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .species-name {
        color: #FF6F61;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .confidence {
        color: #666;
        font-size: 1.1rem;
        margin-top: 15px;
    }
    .flower-image {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 15px auto;
    }
    .image-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 15px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Dictionnaire des images d'iris
# ===============================
IRIS_IMAGES = {
    "0": "Image/Iris_Setosa.jpg",
    "1": "Image/Iris_versicolor.jpg", 
    "2": "Image/Iris_virginica.jpg",
    "Iris-setosa": "Image/Iris_Setosa.jpg",
    "Iris-versicolor": "Image/Iris_versicolor.jpg",
    "Iris-virginica": "Image/Iris_virginica.jpg"
}

# Noms complets des espèces
IRIS_NAMES = {
    "0": "Iris Setosa",
    "1": "Iris Versicolor",
    "2": "Iris Virginica",
    "Iris-setosa": "Iris Setosa",
    "Iris-versicolor": "Iris Versicolor", 
    "Iris-virginica": "Iris Virginica"
}

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
                species_code = res_json.get("prediction")
                confidence = res_json.get("confidence")
                
                # Obtenir le nom complet et l'image
                species_name = IRIS_NAMES.get(species_code, "Espèce inconnue")
                image_path = IRIS_IMAGES.get(species_code)
                
                # Afficher le résultat
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                
                # Nom au dessus
                st.markdown(f"<div class='species-name'>{species_name}</div>", unsafe_allow_html=True)
                
                # Image au milieu avec wrapper pour centrage
                st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
                if image_path and os.path.exists(image_path):
                    st.image(image_path, width=200)  # Taille moyenne
                else:
                    st.warning("Image non trouvée")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Accuracy en bas
                st.markdown(f"<div class='confidence'>Accuracy: {confidence}</div>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur lors de l'appel API : {e}")