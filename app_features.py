import librosa
import numpy as np

# Audio Extraction Function
# It represents the audio files as numbers and averages them so they can have a specific length fingerprint for the sound
def extract_features(file_path, sr=16000, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)