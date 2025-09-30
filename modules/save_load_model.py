# modules/save_load_model.py
import os
from joblib import load

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(filename="lgbm_model.joblib"):
    return load(os.path.join(BASE_DIR, "modules", filename))

def load_encoders(filename="encoders.joblib"):
    return load(os.path.join(BASE_DIR, "modules", filename))
