# modules/save_load_model.py
from joblib import dump, load

def save_model(model, filename):
    dump(model, filename)
    print(f"Model saved as '{filename}'")

def load_model(filename):
    model = load(filename)
    print(f"Model loaded from '{filename}'")
    return model
