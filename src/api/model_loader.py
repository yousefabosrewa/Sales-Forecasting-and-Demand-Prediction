import joblib
import os

# Global variables to hold the loaded model and preprocessor
# This ensures they are loaded once when the module is imported
# or a dedicated loading function is called.
MODEL = None
PREPROCESSOR = None

def load_latest_model(model_path, preprocessor_path):
    """
    Loads the trained model and the fitted preprocessor.
    Stores them in global variables.
    """
    global MODEL, PREPROCESSOR
    print(f"API: Loading model from {model_path}")
    MODEL = joblib.load(model_path)
    print("API: Model loaded.")

    print(f"API: Loading preprocessor from {preprocessor_path}")
    PREPROCESSOR = joblib.load(preprocessor_path)
    print("API: Preprocessor loaded.")

def get_model():
    """Returns the loaded model."""
    return MODEL

def get_preprocessor():
    """Returns the loaded preprocessor."""
    return PREPROCESSOR

# Auto-load the model and preprocessor when the module is imported
# Define paths relative to the project root
# Assuming the API is run from the project root or its path is managed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use the model trained without lag/rolling features for real-time prediction
# Let's assume train.py saves this specific model with a different name or we load
# the main RF/XGB model and acknowledge it might be less accurate without lags.
# Given the prompt asks for best_random_forest_revenue_model.pkl, let's load that one.
# This implies the model *can* handle inputs without lags (e.g., they are set to 0).
# The preprocess_for_prediction function will *not* add lags/rolling for real-time,
# so the model needs to be robust to this.
MODEL_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'best_random_forest_revenue_model.pkl') # Using RF as example
PREPROCESSOR_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'preprocessor.pkl')

# Check if files exist before attempting to load on import
if os.path.exists(MODEL_TO_LOAD) and os.path.exists(PREPROCESSOR_TO_LOAD):
    load_latest_model(MODEL_TO_LOAD, PREPROCESSOR_TO_LOAD)
else:
    print(f"API: Warning: Model ({MODEL_TO_LOAD}) or preprocessor ({PREPROCESSOR_TO_LOAD}) not found on import.")
    print("API: The API will not be able to make predictions until these files are available.")
    MODEL = None
    PREPROCESSOR = None