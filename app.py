import streamlit as st
import pandas as pd
import pickle

# 1. Chargement du pipeline complet sauvegardé avec Pickle
with open("pipeline_churn.pkl", "rb") as f:
    pipeline = pickle.load(f)

# 2. Titre de l'application
st.title("Sénégal Connect — Détection du Churn")
st.write("Saisissez les informations du client pour évaluer le risque de résiliation.")

# 3. Formulaire de saisie des données brutes
st.header("Profil du Client")

anciennete = st.slider("Ancienneté (en mois)", min_value=1, max_value=72, value=12)
type_contrat = st.selectbox("Type de Contrat", ["Mensuel", "1 an", "2 ans"])
montant = st.number_input("Montant Mensuel (FCFA)", min_value=0, value=15000, step=500)
methode_paiement = st.selectbox("Méthode de Paiement", ["Wave", "Orange Money", "Carte Bdf", "Prélèvement"])
reclamations = st.number_input("Nombre de réclamations", min_value=0, max_value=10, value=1)

# 4. Bouton de prédiction
if st.button("Évaluer le Risque"):
    # Organisation des données saisies dans un DataFrame avec les mêmes noms de colonnes qu'à l'entraînement
    donnees_client = pd.DataFrame([{
        "anciennete_mois": anciennete,
        "type_contrat": type_contrat,
        "montant_mensuel": montant,
        "methode_paiement": methode_paiement,
        "reclamations": reclamations
    }])

    # Calcul de la prédiction et de la probabilité via le pipeline
    prediction = pipeline.predict(donnees_client)[0]
    probabilite = pipeline.predict_proba(donnees_client)[0][1]

    st.divider()
    st.subheader("Résultat du Diagnostic")
    st.write(f"Probabilité de résiliation : **{probabilite:.1%}**")

    if prediction == 1:
        st.error(" **Risque Élevé de Churn (Client à Rétentir)**")
        st.info("**Action conseillée :** Proposer une remise sur le réchargement ou un passage en contrat annuel.")
    else:
        st.success(" **Client Stable (Fidèle)**")
        st.info(" **Action conseillée :** Pas d'intervention nécessaire.")