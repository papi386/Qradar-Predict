import joblib

def save_model(obj, filepath: str):
    """Save a model or pipeline"""
    joblib.dump(obj, filepath)
    print(f"Saved: {filepath}")

def load_model(filepath: str):
    """Load a model or pipeline"""
    obj = joblib.load(filepath)
    print(f"Loaded: {filepath}")
    return obj