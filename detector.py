import os
import json
import tempfile

import numpy as np
from xgboost import XGBClassifier

from Velaris.extract_features import extract_features
from preprocess import preprocess_audio


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "voice_deepfake_xgboost.json"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "features.json"
)


# ============================================================
# LOAD FEATURE ORDER
# ============================================================

with open(FEATURES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

def load_model():

    model = XGBClassifier()

    model.load_model(MODEL_PATH)

    return model


# ============================================================
# BUILD FEATURE ROW
# ============================================================

def build_feature_row(model, features):

    # Check for missing features
    missing = [
        name
        for name in FEATURE_NAMES
        if name not in features
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    # IMPORTANT:
    # Use EXACTLY the feature order from training

    values = [
        features[name]
        for name in FEATURE_NAMES
    ]

    X = np.array(
        [values],
        dtype=np.float32
    )

    return X


# ============================================================
# INTERPRET PREDICTION
# ============================================================

def interpret_prediction(
    prediction,
    classes,
    probabilities
):

    prediction = int(prediction)

    # YOUR LABEL MAPPING:
    #
    # 0 = AI / Synthetic
    # 1 = Human

    is_clone = prediction == 0

    # Find confidence of predicted class

    confidence = 0.0

    if probabilities is not None and classes is not None:

        for cls, prob in zip(classes, probabilities):

            if int(cls) == prediction:

                confidence = float(prob) * 100

                break

    # Probability map

    prob_map = {}

    if probabilities is not None and classes is not None:

        for cls, prob in zip(classes, probabilities):

            cls = int(cls)

            if cls == 0:

                prob_map["ai"] = float(prob)

            elif cls == 1:

                prob_map["human"] = float(prob)

    return (
        is_clone,
        confidence,
        prob_map
    )


# ============================================================
# PROBABILITY BREAKDOWN
# ============================================================

def probability_breakdown(prob_map):

    ai_pct = prob_map.get("ai", 0.0) * 100

    human_pct = prob_map.get("human", 0.0) * 100

    return ai_pct, human_pct


# ============================================================
# COMPLETE DETECTION FUNCTION
# ============================================================

def detect_voice(audio_file):

    # Create temporary WAV

    temp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    temp_wav.close()

    try:

        # ----------------------------------------
        # 1. PREPROCESS AUDIO
        # ----------------------------------------

        preprocess_audio(
            audio_file,
            temp_wav.name
        )

        # ----------------------------------------
        # 2. EXTRACT 43 FEATURES
        # ----------------------------------------

        features = extract_features(
            temp_wav.name
        )

        if len(features) != len(FEATURE_NAMES):

            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} features, "
                f"but extractor produced {len(features)}"
            )

        # ----------------------------------------
        # 3. LOAD MODEL
        # ----------------------------------------

        model = load_model()

        # ----------------------------------------
        # 4. BUILD INPUT
        # ----------------------------------------

        X = build_feature_row(
            model,
            features
        )

        # ----------------------------------------
        # 5. PREDICT
        # ----------------------------------------

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        classes = list(model.classes_)

        # ----------------------------------------
        # 6. INTERPRET
        # ----------------------------------------

        is_clone, confidence, probability_map = (
            interpret_prediction(
                prediction,
                classes,
                probabilities
            )
        )

        ai_pct, human_pct = probability_breakdown(
            probability_map
        )

        # ----------------------------------------
        # 7. RETURN RESULT
        # ----------------------------------------

        return {
            "prediction": int(prediction),

            "is_clone": is_clone,

            "confidence": confidence,

            "probabilities": probability_map,

            "ai_probability": ai_pct,

            "human_probability": human_pct,

            "simulated": False
        }

    finally:

        # Remove temporary WAV

        if os.path.exists(temp_wav.name):

            os.remove(temp_wav.name)