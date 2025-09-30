# modules/save_load_model.py
from joblib import dump, load

def save_model(model, filename):
    """Save a trained model to disk."""
    dump(model, filename)
    print(f"Model saved as '{filename}'")

def load_model(filename):
    """Load a trained model from disk."""
    model = load(filename)
    print(f"Model loaded from '{filename}'")
    return model
