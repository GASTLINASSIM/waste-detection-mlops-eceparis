"""
Interface Streamlit — Drone Waste Detection.
"""
import os
from datetime import datetime, timedelta, timezone

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Waste Detection", page_icon="🗑️", layout="wide")
st.title("🗑️ Drone Waste Detection — Live Map")


# ---------- Fonctions de récupération de données ----------
@st.cache_data(ttl=30)
def fetch_models():
    try:
        return requests.get(f"{API}/models", timeout=5).json()
    except Exception as e:
        st.error(f"Impossible de contacter l'API : {e}")
        return []


@st.cache_data(ttl=10)
def fetch_history():
    try:
        return requests.get(f"{API}/history", timeout=10).json()
    except Exception as e:
        st.error(f"Erreur /history : {e}")
        return []


# ---------- Récupération des modèles ----------
models = fetch_models()
model_names = [m["name"] for m in models]

col_left, col_right = st.columns([1, 2])

# ---------- Colonne gauche : détection manuelle ----------
with col_left:
    st.subheader("🛰️ Détection manuelle")

    if not model_names:
        st.warning("Aucun modèle disponible — vérifie que l'API et MLflow tournent.")
        st.stop()

    model_choice = st.selectbox("Modèle", model_names)
    lat = st.number_input("Latitude", value=48.8566, min_value=-90.0, max_value=90.0, format="%.6f")
    lon = st.number_input("Longitude", value=2.3522, min_value=-180.0, max_value=180.0, format="%.6f")
    upload = st.file_uploader("Image (JPEG / PNG)", type=["jpg", "jpeg", "png"])

    if st.button("🚀 Lancer la détection", disabled=upload is None, type="primary"):
        with st.spinner("Inférence en cours..."):
            r = requests.post(
                f"{API}/predict",
                files={"file": (upload.name, upload.getvalue(), upload.type)},
                data={
                    "latitude": lat,
                    "longitude": lon,
                    "model_name": model_choice,
                    "source": "manual",
                },
                timeout=60,
            )
        if r.status_code == 200:
            result = r.json()
            st.success(
                f"✅ Confiance : **{result['confiance']:.2%}** · "
                f"Modèle : `{result['model_used']}`"
            )
            if result["rubbish"]:
                st.warning("🗑️ Déchet détecté !")
            else:
                st.info("Aucun déchet détecté au-dessus du seuil")
            # Rafraîchir l'historique
            fetch_history.clear()
        else:
            st.error(f"Erreur {r.status_code} : {r.text}")

# ---------- Colonne droite : carte et filtres ----------
with col_right:
    st.subheader("🗺️ Carte des détections")

    history = fetch_history()

    if not history:
        st.info("Aucune détection enregistrée pour le moment.")
        m = folium.Map(location=[46.6, 2.5], zoom_start=6)
        st_folium(m, height=500, use_container_width=True)
    else:
        df = pd.DataFrame(history)
        df["ts"] = pd.to_datetime(df["timestamp"])

        # --- Filtres ---
        f1, f2, f3 = st.columns(3)
        with f1:
            src_options = sorted(df["source"].unique())
            src_filter = st.multiselect("Source", src_options, default=src_options)
        with f2:
            mdl_options = sorted(df["model_name"].unique())
            mdl_filter = st.multiselect("Modèle", mdl_options, default=mdl_options)
        with f3:
            hours = st.slider("Dernières N heures", 1, 168, 24)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        # Enlève le timezone pour la comparaison
        df["ts_naive"] = df["ts"].dt.tz_localize(None)
        cutoff_naive = cutoff.replace(tzinfo=None)

        mask = (
            df["source"].isin(src_filter)
            & df["model_name"].isin(mdl_filter)
            & (df["ts_naive"] >= cutoff_naive)
        )
        filtered = df[mask]

        st.caption(f"📊 {len(filtered)} détection(s) affichée(s)")

        # --- Carte ---
        if not filtered.empty:
            center = [filtered["latitude"].mean(), filtered["longitude"].mean()]
        else:
            center = [46.6, 2.5]

        m = folium.Map(location=center, zoom_start=6)

        for _, row in filtered.iterrows():
            # 🔴 rouge = manual / 🟠 orange = drone_patrol
            color = "red" if row["source"] == "manual" else "orange"
            icon = "user" if row["source"] == "manual" else "plane"
            popup_html = (
                f"<b>{row['model_name']}</b><br>"
                f"Confiance : {row['confiance']:.2%}<br>"
                f"Source : {row['source']}<br>"
                f"<small>{row['timestamp']}</small>"
            )
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color=color, icon=icon, prefix="fa"),
            ).add_to(m)

        st_folium(m, height=500, use_container_width=True)