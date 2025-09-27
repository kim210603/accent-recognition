from pathlib import Path
import librosa
import soundfile as sf

# The target sample rate for audio files for consistency
SR = 16000

# Map words in the filename to target accents
TARGETS = {
    "scottish": "scottish",
    "irish": "irish",
    "yorkshire": "yorkshire",
}
# Folders for input and output data
SOURCE_DIR = Path("tts_downloads")
OUT_ROOT = Path("data/prototypes")

# Guesses the accent by looking at the filename
def guess_accent(filename: str):
    lower = filename.lower()
    for key, label in TARGETS.items():
        if key in lower:
            return label
    return None

# Saves the processed audio to disk as a WAV file
def save_wav(y, out_path: Path):
    y, _ = librosa.effects.trim(y, top_db=30)

    # Skips empty audio after it has been trimmed
    if y.size == 0:
        print(f"Warning: {out_path.name} is empty")
        return
    # Makes sure the folder exists, then saves the file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, SR)
    print(f"Saved {out_path}")

    def main():
        counts = {}
        # Goes through each file in the tts_downloads document
        for path in SOURCE_DIR.glob("*.*"):
            if path.suffix.lower() not in {".mp3", ".wav", ".m4a"}:
                continue

            # Figuring out which accent this belongs to
            label = guess_accent(path.name)
            if not label:
                print(f"Skip: {path.name}")
                continue

            # Load audio and resample
            y, _ = librosa.load(path, sr=SR, mono=True)

            # Increment counter for the specific accent (so filenames are unique)
            counts[label] = counts.get(label, 0) + 1
            out_path = OUT_ROOT / label / f"sample{counts[label]}.wav"

            # Save processed WAV
            save_wav(y, out_path)

if __name__ == "__main__":
    main()