import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Comparaison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

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
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #FF6F61;
    }
    .best-model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 6px 10px rgba(0,0,0,0.2);
        margin: 15px 0;
        text-align: center;
    }
    .analysis-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 15px 0;
        white-space: pre-line;
        font-family: 'Arial', sans-serif;
    }
    .model-header {
        color: #2c3e50;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 2px solid #eee;
        padding-bottom: 8px;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: bold;
        color: #FF6F61;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 5px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# ===============================
# Configuration API
# ===============================
api_base = "http://backend:8000"

# ===============================
# Titre et description
# ===============================
st.markdown("<h1>📊 Comparaison de modèles via fichier JSON</h1>", unsafe_allow_html=True)
st.markdown(
    """
    Uploader un fichier JSON contenant les données pour comparer les performances des modèles.
    Cliquez sur **Uploader et comparer** pour lancer l'analyse.
    """,
    unsafe_allow_html=True
)

# ===============================
# Upload et comparaison
# ===============================
uploaded_file = st.file_uploader("Choisir un fichier JSON", type=["json"])

if uploaded_file is not None:
    if st.button("Uploader et comparer"):
        with st.spinner("Analyse en cours..."):
            try:
                files = {"file": uploaded_file}
                response = requests.post(f"{api_base}/compare_file", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Analyse terminée avec succès!")
                    
                    # Affichage des résultats sous forme de cartes
                    st.markdown("### 📈 Résultats de la comparaison")
                    
                    # Afficher les meilleurs modèles
                    if "best_models" in result:
                        best_models = result["best_models"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(
                                f"""
                                <div class="best-model-card">
                                    <h3>🥇 Meilleur Modèle (Accuracy)</h3>
                                    <h2>{best_models.get('by_accuracy', 'N/A').upper()}</h2>
                                    <p>Plus haute précision globale</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            st.markdown(
                                f"""
                                <div class="best-model-card">
                                    <h3>🎯 Meilleur Modèle (F1-Score)</h3>
                                    <h2>{best_models.get('by_f1_macro', 'N/A').upper()}</h2>
                                    <p>Meilleur équilibre précision/rappel</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    # Afficher les métriques par modèle
                    if "metrics" in result:
                        metrics_data = result["metrics"]
                        
                        for model_name, metrics in metrics_data.items():
                            st.markdown(f'<div class="model-header">📊 Modèle {model_name.upper()}</div>', unsafe_allow_html=True)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Accuracy</div>
                                        <div class="metric-value">{metrics.get('accuracy', 0):.3f}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            with col2:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Precision</div>
                                        <div class="metric-value">{metrics.get('precision_macro', 0):.3f}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            with col3:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-label">Recall</div>
                                        <div class="metric-value">{metrics.get('recall_macro', 0):.3f}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            with col4:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div class="metric-label">F1-Score</div>
                                        <div class="metric-value">{metrics.get('f1_macro', 0):.3f}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            
                            # Afficher la matrice de confusion si disponible
                            if "confusion_matrix" in metrics:
                                st.markdown("**Matrice de Confusion:**")
                                confusion_matrix = metrics["confusion_matrix"]
                                # Afficher sous forme de tableau simple
                                for i, row in enumerate(confusion_matrix):
                                    st.write(f"Classe {i}: {row}")
                            
                            st.markdown("---")
                    
                    # Afficher l'analyse
                    if "analysis" in result and result["analysis"]:
                        st.markdown("### 🔍 Analyse Détaillée")
                        st.markdown(
                            f'<div class="analysis-box">{result["analysis"]}</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Option pour voir les données brutes
                    with st.expander("📄 Voir les données brutes (JSON)"):
                        st.json(result)
                        
                else:
                    st.error(f"❌ Erreur API: {response.status_code} - {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de l'appel API: {e}")
else:
    st.markdown(
        '<div class="info-box">📝 Sélectionnez un fichier JSON pour lancer la comparaison.</div>', 
        unsafe_allow_html=True
    )