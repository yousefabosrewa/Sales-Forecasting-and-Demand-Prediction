import joblib # Use joblib for loading
import os
import threading

# Global variables to hold the loaded model, preprocessor, SARIMA model, and feature columns list
_model = None
_preprocessor = None
_sarima_model = None
_feature_columns = None # Add global variable for feature columns list

# Threading lock to prevent multiple threads from loading models simultaneously
_lock = threading.Lock()

def load_latest_model(model_path, preprocessor_path, sarima_model_path, feature_columns_path):
    """
    Loads the latest model, preprocessor, SARIMA model, and feature columns list
    from the specified paths.
    Uses a lock to ensure thread-safe loading.
    Uses joblib.load for all .pkl files for consistency with saving.
    """
    global _model, _preprocessor, _sarima_model, _feature_columns # Declare global variables

    with _lock:
        print(f"API: Attempting to load models from: {model_path}, {preprocessor_path}, {sarima_model_path}, {feature_columns_path}")
        try:
            if os.path.exists(model_path):
                # Use joblib.load for consistency
                _model = joblib.load(model_path)
                print(f"API: Loaded transactional model from {model_path}")
            else:
                _model = None
                print(f"API: Transactional model not found at {model_path}")

            if os.path.exists(preprocessor_path):
                 # Use joblib.load for consistency
                 _preprocessor = joblib.load(preprocessor_path)
                 print(f"API: Loaded preprocessor from {preprocessor_path}")
            else:
                 _preprocessor = None
                 print(f"API: Preprocessor not found at {preprocessor_path}")

            # Load the SARIMA model
            if os.path.exists(sarima_model_path):
                 # Use joblib.load for consistency
                 _sarima_model = joblib.load(sarima_model_path)
                 print(f"API: Loaded SARIMA model from {sarima_model_path}")
            else:
                 _sarima_model = None
                 print(f"API: SARIMA model not found at {sarima_model_path}")

            # Load the feature columns list
            if os.path.exists(feature_columns_path):
                 # Use joblib.load for consistency (even though saved with pickle)
                 # joblib can often load pickle files
                 _feature_columns = joblib.load(feature_columns_path)
                 print(f"API: Loaded feature columns from {feature_columns_path}")
            else:
                 _feature_columns = None
                 print(f"API: Feature columns file not found at {feature_columns_path}")
                 # This is a critical file for prediction, might want to raise an error or log a strong warning

        except Exception as e:
            print(f"API: Error loading models or feature columns: {e}")
            # Log the full traceback for better debugging
            import traceback
            traceback.print_exc()
            _model = None
            _preprocessor = None
            _sarima_model = None
            _feature_columns = None # Ensure feature columns are also set to None on error


def get_model():
    """Returns the loaded transactional model."""
    return _model

def get_preprocessor():
    """Returns the loaded preprocessor."""
    return _preprocessor

def get_sarima_model():
    """Returns the loaded SARIMA model."""
    return _sarima_model

def get_feature_columns():
    """Returns the loaded list of feature columns."""
    return _feature_columns


# Initial load when the module is imported
# This will attempt to load models when the Flask app starts
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models_initial', 'best_random_forest_revenue_model.pkl')
DEFAULT_PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models_initial', 'preprocessor.pkl')
DEFAULT_SARIMA_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models_initial', 'sarima_revenue_model.pkl')
DEFAULT_FEATURE_COLUMNS_PATH = os.path.join(PROJECT_ROOT, 'models_initial', 'feature_columns.pkl') # Define path for feature columns

# Attempt initial load, including the feature columns path
load_latest_model(DEFAULT_MODEL_PATH, DEFAULT_PREPROCESSOR_PATH, DEFAULT_SARIMA_MODEL_PATH, DEFAULT_FEATURE_COLUMNS_PATH)
