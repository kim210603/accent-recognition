from pathlib import Path
from uuid import uuid4
from datetime import datetime
import io
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder
from app_features import extract_features  # helps to upload and process files


# This is the streamlit page setup and styling
st.set_page_config(page_title="Accent Recognition App", layout="wide")
st.markdown(
    """
    <style>
      #MainMenu {visibility:hidden;} footer {visibility:hidden;}
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# This function shows the pie chart of the accent breakdown
def show_chart(sims: dict):
    st.write("### Accent breakdown")
    fig = px.pie(
        values=list(sims.values()),
        names=[k.capitalize() for k in sims.keys()],
        hole=0.45,
    )
    # This shows only the label when you hover over the section, and the percentage is shown on the chart
    fig.update_traces(hovertemplate="%{label}", textinfo="percent")
    st.plotly_chart(fig, use_container_width=True)


def show_top_match(sims: dict):
    best_acc, best_pct = max(sims.items(), key=lambda kv: kv[1])
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.subheader("Best match")
        st.markdown(f"### **{best_acc.capitalize()}**  ·  {best_pct:.1f}%")
    with col2:
        st.metric(label="Confidence", value=f"{best_pct:.1f}%")

# This is the function that extracts features from the audio files
def extract_features_from_bytes(file_bytes: bytes, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    try:
        y, _ = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True)
    except Exception:
        decoded = None
        for fmt in ("wav", "webm", "ogg", "mp3", "m4a"):
            try:
                decoded = AudioSegment.from_file(io.BytesIO(file_bytes), format=fmt)
                break
            except Exception:
                continue
        if decoded is None:
            raise RuntimeError("Could not decode mic audio.")

        samples = np.array(decoded.get_array_of_samples(), dtype=np.float32)
        if decoded.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        max_abs = float(1 << (8 * decoded.sample_width - 1))
        y = samples / max_abs
        if decoded.frame_rate != sr:
            y = librosa.resample(y, orig_sr=decoded.frame_rate, target_sr=sr)

    # This turns the audio into numbers that captures the sound fingerprints, instead of keeping the raw waveform (Help from AI)
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        return np.zeros(n_mfcc, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1).astype(np.float32)

#This is a csv file that logs the results of the accent recognition recordings
LOG_FILE = Path("results_log.csv")

def log_results(source: str, sims: dict, rec_id: str):
    row = {"id": rec_id, "time": datetime.now().isoformat(timespec="seconds"), "source": source}
    row.update({k: float(v) for k, v in sims.items()})
    df = pd.DataFrame([row])
    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, mode="w", header=True, index=False)

# The history section that shows the last 20 recordings
if "history" not in st.session_state:
    st.session_state.history = []

def add_to_history(source: str, sims: dict, rec_id: str):
    st.session_state.history.insert(0, {
        "id": rec_id,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        **{k: round(float(v), 1) for k, v in sims.items()},
    })
    st.session_state.history = st.session_state.history[:20]

# Loads the accent prototypes from the models folder
MODEL_FILE = Path("models/prototypes.npz")

@st.cache_resource
def load_prototypes():
    return np.load(MODEL_FILE, allow_pickle=True)


# Main UI for the application
st.title("Accent Recognition App")

st.subheader("Speak")
st.caption("Tap Speak, say one sentence (~5 seconds), then Stop. Audio is processed in memory and not stored.")

# Microphone audio recording and processing
audio = mic_recorder(start_prompt="Speak", stop_prompt="Stop", just_once=False, use_container_width=True)

if audio is not None:
    wav_bytes = audio["bytes"]

    feats = extract_features_from_bytes(wav_bytes)
    prototypes = np.load("models/prototypes.npz", allow_pickle=True)

    dists = {accent: np.linalg.norm(feats - centroid) for accent, centroid in prototypes.items()}
    max_d = max(dists.values())
    sims = {a: 100 * (1 - (d / (max_d + 1e-9))) for a, d in dists.items()}

    # Shows the best match and the pie chart breakdown
    show_top_match(sims)
    show_chart(sims)

    # Saves the recording results to the history and the log file
    rec_id = str(uuid4())
    add_to_history("mic", sims, rec_id)
    log_results("mic", sims, rec_id)
    del wav_bytes

# File upload section
st.subheader("Or upload a WAV file")
uploaded = st.file_uploader("Choose file", type=["wav"])
if uploaded:
    feats = extract_features(uploaded)
    prototypes = load_prototypes()

    dists = {accent: np.linalg.norm(feats - centroid) for accent, centroid in prototypes.items()}
    max_d = max(dists.values())
    sims = {a: 100 * (1 - d / max_d) for a, d in dists.items()}

    show_top_match(sims)
    show_chart(sims)

    rec_id = str(uuid4())
    add_to_history("upload", sims, rec_id)
    log_results("upload", sims, rec_id)


# History section
st.markdown("---")
st.subheader("History (latest 20)")
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history).copy()
    hist_df = hist_df.drop(columns=["select"], errors="ignore")
    cols = ["id", "time", "source"] + [c for c in hist_df.columns if c not in ("id", "time", "source")]
    hist_df = hist_df[cols]

    rename_map = {c: f"{c.capitalize()} (Confidence %)" for c in ("scottish", "irish", "yorkshire") if c in hist_df.columns}
    hist_df.rename(columns=rename_map, inplace=True)
    conf_cols = {name: st.column_config.NumberColumn(format="%.1f %%") for name in rename_map.values()}

    st.dataframe(hist_df, use_container_width=True, column_config=conf_cols)
else:
    st.caption("No results yet.")


# Sidebar with the latest result
with st.sidebar:
    st.header("Latest result")
    if st.session_state.history:
        last = st.session_state.history[0]
        accents = [k for k in last.keys() if k not in ("id", "time", "source")]
        sims_latest = {a: last[a] for a in accents}
        show_top_match(sims_latest)
        show_chart(sims_latest)
        st.caption(f"{last['time']} • {last['source']}")
    else:
        st.caption("No results yet.")
