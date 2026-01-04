import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# CONFIGURATION STREAMLIT
# =========================
st.set_page_config(
    page_title="Détection d'Attaques Réseau",
    layout="wide"
)

st.title("🛡️ Système de Détection d'Attaques IoT")
st.markdown("""
Cette application utilise un modèle **Gradient Boosting**  
incluant **tout le preprocessing dans un seul pipeline**.
""")

# =========================
# CHARGEMENT DU MODÈLE
# =========================
@st.cache_resource
def load_model():
    return joblib.load("GradientBoosting.joblib")

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    st.stop()

# =========================
# RÉCUPÉRATION DES FEATURES
# =========================
if hasattr(pipeline, "feature_names_in_"):
    FEATURES = pipeline.feature_names_in_
else:
    st.error("❌ Impossible de récupérer les features du modèle.")
    st.stop()

# =========================
# INTERFACE UTILISATEUR
# =========================
st.sidebar.header("Paramètres du flux réseau")

def user_input_features():
    data = {}

    # Champs principaux visibles
    data["id.orig_p"] = st.sidebar.number_input("Port origine", value=38667)
    data["id.resp_p"] = st.sidebar.number_input("Port destination", value=1883)
    data["flow_duration"] = st.sidebar.number_input("Durée du flux", value=32.0)
    data["fwd_pkts_tot"] = st.sidebar.number_input("Paquets forward", value=9)
    data["bwd_pkts_tot"] = st.sidebar.number_input("Paquets backward", value=5)

    df = pd.DataFrame([data])

    # Ajouter automatiquement les colonnes manquantes
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df[FEATURES]

input_df = user_input_features()

# =========================
# APERÇU DES DONNÉES
# =========================
st.subheader("Aperçu des données d'entrée")
st.dataframe(input_df.iloc[:, :8])

# =========================
# PRÉDICTION
# =========================
if st.button("🔍 Analyser le flux"):
    prediction = pipeline.predict(input_df)[0]

    st.subheader("Résultat de l'analyse")

    if prediction in ["Normal", "Thing_Speak"]:
        st.success(f"✅ Flux sécurisé : **{prediction}**")
    else:
        st.error(f"⚠️ Attaque détectée : **{prediction}**")

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)
        prob_df = pd.DataFrame(
            probs,
            columns=pipeline.classes_
        ).T
        st.subheader("Score de confiance")
        st.bar_chart(prob_df)
