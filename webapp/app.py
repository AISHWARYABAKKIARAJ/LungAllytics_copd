"""
COPD Insight - Flask backend.

Serves two INDEPENDENT prediction tools (deliberately not fused into one
score - see README.md for why):

  1. /clinical  - tabular model (KNN + XGBoost + RandomForest voting
                  ensemble) that estimates COPD severity (mild/moderate vs
                  severe/very severe) from symptoms and vitals.
  2. /xray       - ResNet50 transfer-learning model that screens a chest
                  X-ray image for visible signs of COPD.
"""

import os

import numpy as np
from flask import Flask, render_template, request
import joblib
from PIL import Image

# TensorFlow import is slow - load it once at startup so the first
# request from a user isn't delayed by it.
import tensorflow as tf

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CLINICAL_MODEL_PATH = os.path.join(APP_DIR, "models", "clinical_model.joblib")
IMAGE_MODEL_PATH = os.path.join(APP_DIR, "models", "best_copd_image_model.h5")

app = Flask(__name__)

clinical_model = None
image_model = None


def load_models():
    """Loaded lazily/once so the app still starts (and shows a clear error
    on the relevant page) even if one of the two model files is missing."""
    global clinical_model, image_model
    if clinical_model is None and os.path.exists(CLINICAL_MODEL_PATH):
        clinical_model = joblib.load(CLINICAL_MODEL_PATH)
    if image_model is None and os.path.exists(IMAGE_MODEL_PATH):
        image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)


# Column order the clinical model was trained on. This MUST match the
# order of df_new.drop(columns=['copd_gold', 'severe_copd']) in
# clinical_model_voting_classifier.ipynb - do not reorder.
FEATURE_ORDER = [
    "age", "gender", "bmi_kg/m2", "height/m", "history_of_heart_failure",
    "working_place", "mmrc", "status_of_smoking", "pack_history",
    "vaccination", "depression", "dependent", "temperature",
    "respiratory_rate", "heart_rate", "blood_pressure",
    "oxygen_saturation", "sputum", "fev1",
]

IMAGE_SIZE = (224, 224)


# --------------------------------------------------------------------------
# Encoding helpers.
#
# The source clinical dataset stores several fields as pre-binned integer
# codes with no published data dictionary. Where the meaning of a code
# could be confirmed (against clinical convention, or against patterns in
# the training data itself), we convert a real-world value the patient
# enters. Where it could not be confirmed, we ask for the training-data
# code directly and say so in the form - see README.md for the evidence
# behind each choice.
# --------------------------------------------------------------------------

def bmi_to_code(bmi):
    """WHO adult BMI categories -> the 0-4 bucket used in training."""
    if bmi < 18.5:
        return 0  # underweight
    if bmi < 25:
        return 1  # normal
    if bmi < 30:
        return 2  # overweight
    if bmi < 35:
        return 3  # obese (class I)
    return 4       # obese (class II+)


def fev1_pct_to_code(pct):
    """GOLD airflow-limitation grading by FEV1 % predicted, mapped onto
    this dataset's fev1 column. Verified against the training data: rows
    with fev1 == 4 are almost all GOLD stage 1 (mild); rows with fev1 == 1
    are almost all GOLD stage 3/4 (severe) - consistent with this mapping."""
    if pct >= 80:
        return 4
    if pct >= 50:
        return 3
    if pct >= 30:
        return 2
    return 1


def smoking_to_code(value):
    """value: 'never' | 'current' | 'former'.
    Confirmed from training data: status_of_smoking == 3 has pack_history
    == 0 for every one of the 92 patients with that code, so 3 = never
    smoked. Whether 1 or 2 means current vs former could not be confirmed
    from the data alone (best-effort: 1 = current, 2 = former)."""
    return {"never": 3, "current": 1, "former": 2}[value]


def yes_no_to_code(value):
    return 1 if value == "yes" else 0


def gender_to_code(value):
    # Which code means male/female could not be verified (both genders are
    # distributed similarly across every severity level in the training
    # data). Defaulting to male=1, female=0.
    return 1 if value == "male" else 0


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/clinical", methods=["GET", "POST"])
def clinical():
    load_models()
    result = None

    if request.method == "POST":
        if clinical_model is None:
            result = {"error": "Clinical model file not found on the server. "
                                "See README.md for where to put clinical_model.joblib."}
        else:
            form = request.form
            try:
                features = {
                    "age": float(form["age"]),
                    "gender": gender_to_code(form["gender"]),
                    "bmi_kg/m2": bmi_to_code(float(form["bmi"])),
                    "height/m": 1,  # constant in training data - see README
                    "history_of_heart_failure": yes_no_to_code(form["heart_failure"]),
                    "working_place": int(form["working_place"]),
                    "mmrc": int(form["mmrc"]),
                    "status_of_smoking": smoking_to_code(form["smoking"]),
                    "pack_history": float(form["pack_history"]),
                    "vaccination": yes_no_to_code(form["vaccination"]),
                    "depression": yes_no_to_code(form["depression"]),
                    "dependent": yes_no_to_code(form["dependent"]),
                    "temperature": int(form["temperature"]),
                    "respiratory_rate": float(form["respiratory_rate"]),
                    "heart_rate": int(form["heart_rate"]),
                    "blood_pressure": int(form["blood_pressure"]),
                    "oxygen_saturation": float(form["oxygen_saturation"]) / 100.0,
                    "sputum": yes_no_to_code(form["sputum"]),
                    "fev1": fev1_pct_to_code(float(form["fev1_pct"])),
                }
                row = np.array([[features[c] for c in FEATURE_ORDER]])
                proba_severe = float(clinical_model.predict_proba(row)[0, 1])
                label = "Severe / Very Severe" if proba_severe > 0.5 else "Mild / Moderate"
                result = {
                    "label": label,
                    "is_severe": proba_severe > 0.5,
                    "proba_severe": round(proba_severe * 100, 1),
                    "proba_mild": round((1 - proba_severe) * 100, 1),
                }
            except (KeyError, ValueError) as exc:
                result = {"error": f"Please check your inputs ({exc})."}

    return render_template("clinical.html", result=result)


@app.route("/xray", methods=["GET", "POST"])
def xray():
    load_models()
    result = None

    if request.method == "POST":
        if image_model is None:
            result = {"error": "Image model file not found on the server. "
                                "See README.md for where to put best_copd_image_model.h5."}
        else:
            file = request.files.get("xray_image")
            if not file or file.filename == "":
                result = {"error": "Please choose a chest X-ray image first."}
            else:
                try:
                    img = Image.open(file.stream).convert("RGB").resize(IMAGE_SIZE)
                    arr = np.asarray(img, dtype="float32") / 255.0
                    arr = np.expand_dims(arr, axis=0)

                    # Trained with classes=['aff', 'normal'], so the sigmoid
                    # output is P(normal); affected probability is 1 minus that.
                    p_normal = float(image_model.predict(arr, verbose=0)[0, 0])
                    p_affected = 1 - p_normal

                    is_affected = p_affected > 0.5
                    label = "Signs of COPD-related abnormality detected" if is_affected else "No abnormality detected"
                    result = {
                        "label": label,
                        "is_affected": is_affected,
                        "proba_affected": round(p_affected * 100, 1),
                        "proba_normal": round(p_normal * 100, 1),
                    }
                except Exception as exc:
                    result = {"error": f"Could not read that image ({exc})."}

    return render_template("xray.html", result=result)


if __name__ == "__main__":
    # Locally: python app.py -> http://127.0.0.1:5000, debug on for clearer errors.
    # Deployed (Render/Hugging Face/etc.): PORT is set by the host and the
    # process is normally started by gunicorn instead (see Dockerfile/README),
    # but this fallback keeps `python app.py` working there too.
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
