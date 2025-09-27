from pathlib import Path
import numpy as np
from app_features import extract_features

DATA_ROOT = Path("data/prototypes")
MODEL-FILE = Path("models/prototypes.npz")

def main():
    accents = {}
    for accent_dir in DATA_ROOT.iterdir():
        if accent_dir.is_dir():
            continue
        feats = []
        for wav in accent_dir.glob("*.wav"):
            feats.append(extract_features(str(wav)))
        if feats:
            accents[accent_dir.name] = np.mean(feats, axis=0)

        MODEL_FILE.parent.mkdir(exist_ok=True, parents=True)
        np.savez(MODEL_FILE, **accents)
        print(f"Saved prototypes to {MODEL_FILE}")

if __name__ == "__main__":
    main()