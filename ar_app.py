import streamlit as st
import numpy as np
from app_features import extract_features

MODEL_FILE - "models/prototypes.npz"
# This gets cached so that it only loads once
@st.cache_resource
def load_prototypes():
    return np.load(MODEL_FILE, allow_pickle=True)

# Streamlit User Interface
st.title("Accent Recognition App Demo")

# Upload audio file function
uploaded = st.file_uploader("Upload an WAV audio file", type=["wav"])
if uploaded:
    feats  = extract_features(uploaded) # Extracting featured from the uploaded audio
    # Loading the protypes (getting the average features from each accent)
    prototypes = load_prototypes()

    results = {}
    # Comparing the uploaded clips to the accent prototypes by calculating the distance
    for accent, centroid in prototypes.items():
        dist = np.linalg.norm(feats - centroid)
        results[accent] = dist
        # Converting the distances to similarity percentages (closer = higher percentage)
        max_d = max(results.values())
        sims = {a: 100 * (1 - d / max_d) for a, d in results.items()}
        # Shows the results in a bar chart
        st.write ("### Accent Similarities")
        st.bar_chart(sims)