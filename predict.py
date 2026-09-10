import os
import json
import numpy as np
from xgboost import XGBClassifier


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
# LOAD MODEL
# ============================================================

def load_model():

    model = XGBClassifier()

    model.load_model(MODEL_PATH)

    return model


# ============================================================
# BUILD FEATURE ROW
# ============================================================

def build_feature_row(model, features):

    missing = [
        name
        for name in FEATURE_NAMES
        if name not in features
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

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

    if prediction == 0:

        is_clone = True

    else:

        is_clone = False


    # Find probability belonging to predicted class

    confidence = 0.0

    if probabilities is not None and classes is not None:

        for cls, prob in zip(classes, probabilities):

            if int(cls) == prediction:

                confidence = float(prob) * 100

                break


    # Build probability dictionary

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