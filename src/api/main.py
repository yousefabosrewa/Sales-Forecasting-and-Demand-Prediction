from flask import Flask, request, jsonify
from flask_pydantic import validate
import pandas as pd
import io # For CSV processing
import traceback # For error logging
import sys # Import sys for printing to stderr
import os
# Import model loading and prediction utilities
from src.api.model_loader import get_model, get_preprocessor, load_latest_model # Import load_latest_model for /reload endpoint
from src.predict_utils import preprocess_for_prediction, predict_revenue
from src.api.schemas import PredictionInput, PredictionOutput, BatchPredictionInput, BatchPredictionOutput

app = Flask(__name__)

# --- API Endpoints ---

@app.route('/predict', methods=['POST'])
@validate() # Validate input using Pydantic schema
def predict_single(body: PredictionInput):
    """
    Endpoint for single real-time predictions.
    Accepts JSON matching PredictionInput schema.
    """
    model = get_model()
    preprocessor = get_preprocessor()

    if model is None or preprocessor is None:
        return jsonify({"error": "Model or preprocessor not loaded. Server is not ready."}), 503 # Service Unavailable

    try:
        # Convert Pydantic model to pandas DataFrame row
        input_data_raw = body.model_dump() # For pydantic v2+, use model_dump()
        # input_data_raw = body.dict() # For pydantic v1

        input_df = pd.DataFrame([input_data_raw]) # Convert single dict to DataFrame

        # Preprocess the input data
        processed_features = preprocess_for_prediction(input_df, preprocessor)

        if processed_features.empty:
             return jsonify({"error": "Preprocessing resulted in no valid features."}), 400 # Bad Request


        # Make prediction
        predictions = predict_revenue(model, processed_features, preprocessor)

        # Assuming a single prediction is returned for single input
        if len(predictions) != 1:
             # This should ideally not happen for a single input
             print(f"Warning: predict_revenue returned {len(predictions)} predictions for single input.")
             return jsonify({"error": "Unexpected number of predictions returned."}), 500 # Internal Server Error


        # Return prediction as JSON
        output = PredictionOutput(predicted_revenue=predictions[0])
        return jsonify(output.model_dump()) # For pydantic v2+, use model_dump()


    except Exception as e:
        print(f"Error during single prediction: {e}", file=sys.stderr) # Print errors to stderr
        traceback.print_exc(file=sys.stderr) # Print traceback to stderr
        return jsonify({"error": "An internal error occurred during prediction.", "details": str(e)}), 500 # Internal Server Error


@app.route('/predict_batch_json', methods=['POST'])
@validate() # Validate input using Pydantic schema for JSON batch
def predict_batch_json(body: BatchPredictionInput):
     """
     Endpoint for batch predictions using JSON list of inputs.
     """
     model = get_model()
     preprocessor = get_preprocessor()

     if model is None or preprocessor is None:
        return jsonify({"error": "Model or preprocessor not loaded. Server is not ready."}), 503 # Service Unavailable

     try:
         # Convert list of Pydantic models to pandas DataFrame
         input_list_of_dicts = [item.model_dump() for item in body.data] # pydantic v2+
         input_df_raw = pd.DataFrame(input_list_of_dicts)

         if input_df_raw.empty:
              return jsonify({"predictions": []}), 200 # OK, but no data processed

         # Preprocess the batch data
         processed_features = preprocess_for_prediction(input_df_raw, preprocessor)

         if processed_features.empty:
              return jsonify({"error": "Preprocessing resulted in no valid features for any record."}), 400


         # Make predictions for the batch
         predictions = predict_revenue(model, processed_features, preprocessor)

         # Structure the output
         output_predictions = [PredictionOutput(predicted_revenue=float(p)) for p in predictions] # Ensure float serializable

         # Return batch predictions as JSON
         output = BatchPredictionOutput(predictions=output_predictions)
         return jsonify(output.model_dump()) # pydantic v2+

     except Exception as e:
        print(f"Error during batch JSON prediction: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": "An internal error occurred during batch prediction (JSON).", "details": str(e)}), 500


@app.route('/predict_batch_csv', methods=['POST'])
def predict_batch_csv():
    """
    Endpoint for batch predictions using CSV file upload.
    """
    model = get_model()
    preprocessor = get_preprocessor()

    if model is None or preprocessor is None:
        return jsonify({"error": "Model or preprocessor not loaded. Server is not ready."}), 503 # Service Unavailable

    if 'file' not in request.files:
        print("API: No 'file' part in the request.", file=sys.stderr) # Debug print
        return jsonify({"error": "No file part in the request"}), 400 # Bad Request

    file = request.files['file']

    if file.filename == '':
        print("API: No selected file.", file=sys.stderr) # Debug print
        return jsonify({"error": "No selected file"}), 400 # Bad Request

    if file and file.filename.endswith('.csv'):
        try:
            print(f"API: Received file: {file.filename}", file=sys.stderr) # Debug print

            # Read the CSV file into a pandas DataFrame
            # Use BytesIO to read from memory buffer
            csv_data = io.BytesIO(file.read())
            print("API: Attempting to read CSV into DataFrame...", file=sys.stderr) # Debug print
            input_df_raw = pd.read_csv(csv_data)
            print("API: Successfully read CSV into DataFrame.", file=sys.stderr) # Debug print
            print("API: DataFrame head:", file=sys.stderr) # Debug print
            print(input_df_raw.head(), file=sys.stderr) # Debug print
            print("API: DataFrame columns:", file=sys.stderr) # Debug print
            print(input_df_raw.columns.tolist(), file=sys.stderr) # Debug print
            print("API: DataFrame info:", file=sys.stderr) # Debug print
            input_df_raw.info(buf=sys.stderr) # Print info to stderr


            if input_df_raw.empty:
                 print("API: Read an empty DataFrame from CSV.", file=sys.stderr) # Debug print
                 return jsonify({"predictions": []}), 200 # OK, but no data processed


            print("API: Calling preprocess_for_prediction...", file=sys.stderr) # Debug print
            # Preprocess the batch data
            processed_features = preprocess_for_prediction(input_df_raw, preprocessor)
            print("API: preprocess_for_prediction completed.", file=sys.stderr) # Debug print
            print("API: Processed features shape:", processed_features.shape, file=sys.stderr) # Debug print
            print("API: Processed features columns:", processed_features.columns.tolist(), file=sys.stderr) # Debug print


            if processed_features.empty:
                 print("API: Preprocessing resulted in no valid features.", file=sys.stderr) # Debug print
                 return jsonify({"error": "Preprocessing resulted in no valid features for any record from CSV."}), 400


            print("API: Calling predict_revenue...", file=sys.stderr) # Debug print
            # Make predictions for the batch
            predictions = predict_revenue(model, processed_features, preprocessor)
            print("API: predict_revenue completed.", file=sys.stderr) # Debug print


            # Structure the output
            # Return as a list of dicts or a simple list
            output_predictions = [{"predicted_revenue": float(p)} for p in predictions]

            print("API: Returning batch predictions JSON.", file=sys.stderr) # Debug print
            return jsonify({"predictions": output_predictions})

        except Exception as e:
            print(f"API: Error during batch CSV prediction: {e}", file=sys.stderr) # Print errors to stderr
            traceback.print_exc(file=sys.stderr) # Print traceback to stderr
            return jsonify({"error": "An internal error occurred during batch prediction (CSV).", "details": str(e)}), 500

    else:
        print(f"API: Invalid file type uploaded: {file.filename}", file=sys.stderr) # Debug print
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400 # Bad Request

@app.route('/status', methods=['GET'])
def status():
    """
    Health check endpoint.
    """
    model_loaded = get_model() is not None
    preprocessor_loaded = get_preprocessor() is not None
    status_ok = model_loaded and preprocessor_loaded
    return jsonify({
        "status": "ok" if status_ok else "loading_error",
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded
    }), 200 if status_ok else 503 # Service Unavailable


@app.route('/reload_model', methods=['POST'])
def reload_model():
    """
    Endpoint to manually trigger model and preprocessor reload.
    Useful after deploying a new model version.
    """
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'best_random_forest_revenue_model.pkl')
    PREPROCESSOR_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'preprocessor.pkl')
    try:
        load_latest_model(MODEL_TO_LOAD, PREPROCESSOR_TO_LOAD)
        return jsonify({"message": "Model and preprocessor reloaded successfully."}), 200
    except FileNotFoundError as e:
        return jsonify({"error": f"Model or preprocessor file not found: {e}"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to reload model: {e}"}), 500


# --- Main execution ---
if __name__ == '__main__':
    # In production, use a WSGI server like Gunicorn or Waitress.
    # This is for local development testing.
    print("Running Flask app in development mode. Use Gunicorn/Waitress for production.")
    # Ensure model and preprocessor are loaded on startup if not already by import
    if get_model() is None or get_preprocessor() is None:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        MODEL_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'best_random_forest_revenue_model.pkl')
        PREPROCESSOR_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'preprocessor.pkl')
        if os.path.exists(MODEL_TO_LOAD) and os.path.exists(PREPROCESSOR_TO_LOAD):
             load_latest_model(MODEL_TO_LOAD, PREPROCESSOR_TO_LOAD)
        else:
             print("API: Model/preprocessor not found. Cannot start in a ready state.")
             print("Please run training (src/train.py) first.")

    app.run(debug=True, host='0.0.0.0', port=5001) # Use port 5001 for API
