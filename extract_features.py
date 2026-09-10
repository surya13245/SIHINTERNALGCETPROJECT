import librosa
import numpy as np


def extract_features(file_path):

    # Load audio
    y, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    features = {}

    # ========================================
    # 1. MFCC
    # ========================================

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    # ========================================
    # 2. Spectral features
    # ========================================

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    zero_crossing = librosa.feature.zero_crossing_rate(y)

    features["spectral_centroid_mean"] = np.mean(spectral_centroid)
    features["spectral_centroid_std"] = np.std(spectral_centroid)

    features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
    features["spectral_bandwidth_std"] = np.std(spectral_bandwidth)

    features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
    features["spectral_rolloff_std"] = np.std(spectral_rolloff)

    features["zero_crossing_mean"] = np.mean(zero_crossing)
    features["zero_crossing_std"] = np.std(zero_crossing)

    # ========================================
    # 3. Energy
    # ========================================

    rms = librosa.feature.rms(y=y)

    features["energy_mean"] = np.mean(rms)
    features["energy_std"] = np.std(rms)

    # ========================================
    # 4. Pitch
    # ========================================

    f0 = librosa.yin(
        y,
        fmin=70,
        fmax=400,
        sr=sr
    )

    f0 = f0[np.isfinite(f0)]

    if len(f0) > 0:
        features["pitch_mean"] = np.mean(f0)
        features["pitch_std"] = np.std(f0)
        features["pitch_min"] = np.min(f0)
        features["pitch_max"] = np.max(f0)
    else:
        features["pitch_mean"] = 0
        features["pitch_std"] = 0
        features["pitch_min"] = 0
        features["pitch_max"] = 0

    # ========================================
    # 5. Silence / pause characteristics
    # ========================================

    intervals = librosa.effects.split(
        y,
        top_db=30
    )

    speech_samples = sum(
        end - start
        for start, end in intervals
    )

    total_samples = len(y)

    speech_ratio = speech_samples / total_samples
    silence_ratio = 1 - speech_ratio

    features["speech_ratio"] = speech_ratio
    features["silence_ratio"] = silence_ratio

    features["speech_segments"] = len(intervals)

    # ========================================

    return features