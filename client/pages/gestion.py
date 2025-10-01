import streamlit as st
import requests
import json

# ===============================
# CSS global simplifié
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
        margin: 5px;
    }
    .stButton>button:hover {
        background-color: #FF3B2E;
    }
    h1 {
        color: #333;
        text-align: center;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# ===============================
# Configuration API - CORRIGÉ
# ===============================
api_base = "http://backend:8000"  # 🟢 CHANGEMENT ICI

# ===============================
# Titre et description
# ===============================
st.markdown("<h1>⚙️ Gestion des modèles</h1>", unsafe_allow_html=True)

# ===============================
# Boutons d'action en haut
# ===============================
col1, col2, col3 = st.columns(3)

with col1:
    lister_btn = st.button("📋 Lister les modèles")

with col2:
    ajouter_btn = st.button("➕ Ajouter un modèle")

with col3:
    supprimer_btn = st.button("🗑️ Supprimer un modèle")

# ===============================
# Section LISTER les modèles
# ===============================
if lister_btn:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 📋 Liste des modèles")
    try:
        with st.spinner("Chargement des modèles..."):
            res = requests.get(f"{api_base}/list_models")
            if res.status_code == 200:
                models = res.json()
                st.success(f"✅ {len(models.get('models', {}))} modèle(s) trouvé(s)")
                st.json(models)
            else:
                st.error(f"❌ Erreur API: {res.status_code}")
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Section AJOUTER un modèle
# ===============================
if ajouter_btn:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### ➕ Ajouter un modèle pré-entraîné")
    
    uploaded_model = st.file_uploader("Choisir un fichier .pkl", type=["pkl"], key="uploader")
    col1, col2 = st.columns(2)
    with col1:
        accuracy = st.number_input("Accuracy", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    with col2:
        f1_macro = st.number_input("F1-macro", min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    
    if st.button("✅ Confirmer l'ajout", key="add_confirm"):
        if uploaded_model is not None:
            try:
                with st.spinner("Ajout du modèle en cours..."):
                    files = {"file": uploaded_model}
                    data = {"accuracy": accuracy, "f1_macro": f1_macro}
                    res = requests.post(f"{api_base}/add_model", files=files, data=data)
                    
                    if res.status_code == 200:
                        st.success("✅ Modèle ajouté avec succès!")
                        st.json(res.json())
                    else:
                        st.error(f"❌ Erreur lors de l'ajout: {res.status_code}")
            except Exception as e:
                st.error(f"❌ Erreur API: {e}")
        else:
            st.warning("⚠️ Veuillez choisir un fichier .pkl")
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Section SUPPRIMER un modèle
# ===============================
if supprimer_btn:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### 🗑️ Supprimer un modèle")
    
    model_name = st.text_input("Nom du modèle à supprimer", placeholder="Entrez le nom exact du modèle")
    
    if st.button("✅ Confirmer la suppression", key="delete_confirm"):
        if model_name.strip():
            try:
                with st.spinner("Suppression en cours..."):
                    res = requests.delete(f"{api_base}/delete_model?model_name={model_name}")
                    
                    if res.status_code == 200:
                        st.success("✅ Modèle supprimé avec succès!")
                        st.json(res.json())
                    else:
                        st.error(f"❌ Erreur lors de la suppression: {res.status_code}")
            except Exception as e:
                st.error(f"❌ Erreur API: {e}")
        else:
            st.warning("⚠️ Veuillez entrer un nom de modèle")
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Message d'information initial
# ===============================
if not any([lister_btn, ajouter_btn, supprimer_btn]):
    st.markdown(
        '<div class="info-box">👆 Sélectionnez une action ci-dessus pour gérer vos modèles</div>', 
        unsafe_allow_html=True
    )