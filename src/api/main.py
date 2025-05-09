from flask import Flask, request, jsonify
from flask_pydantic import validate
import pandas as pd
import io # For CSV processing
import traceback # For error logging
import sys # Import sys for printing to stderr
import os # Import os for path joining
from datetime import datetime # Import datetime for date handling

# Import model loading and prediction utilities
from src.api.model_loader import get_model, get_preprocessor, load_latest_model, get_sarima_model # Import get_sarima_model
from src.predict_utils import preprocess_for_prediction, predict_revenue # Keep for transactional prediction
# Assuming you will add a new predict_sarima function in predict_utils.py
# from src.predict_utils import predict_sarima # Will need this later

# Import Pydantic schemas
from src.api.schemas import PredictionInput, PredictionOutput, BatchPredictionInput, BatchPredictionOutput
# Assuming you will add a new schema for SARIMA forecast input
from pydantic import BaseModel, Field # Import BaseModel and Field for new schema

# Define a new Pydantic schema for SARIMA forecast input
class SarimaForecastInput(BaseModel):
    """Schema for SARIMA forecast input."""
    start_date: str = Field(..., description="Start date for forecasting (YYYY-MM-DD).")
    end_date: str = Field(..., description="End date for forecasting (YYYY-MM-DD).")

    # Optional: Add frequency if needed, but for simplicity, assume daily forecast
    # frequency: str = Field("D", description="Forecast frequency (e.g., D for daily, M for monthly).")

# Define a new Pydantic schema for SARIMA forecast output
class SarimaForecastOutput(BaseModel):
    """Schema for SARIMA forecast output."""
    date: str = Field(..., description="Date of the forecast.")
    predicted_revenue: float = Field(..., description="Predicted revenue for the date.")

# Define a schema for a list of SARIMA forecast outputs
class BatchSarimaForecastOutput(BaseModel):
    """Schema for batch SARIMA forecast output."""
    forecasts: list[SarimaForecastOutput] = Field(..., description="List of SARIMA forecasts.")


app = Flask(__name__)

# --- API Endpoints ---

@app.route('/predict', methods=['POST'])
@validate() # Validate input using Pydantic schema
def predict_single(body: PredictionInput):
    """
    Endpoint for single real-time predictions (transactional model).
    Accepts JSON matching PredictionInput schema.
    """
    model = get_model() # Get the transactional model (RF or XGBoost)
    preprocessor = get_preprocessor()

    if model is None or preprocessor is None:
        return jsonify({"error": "Transactional model or preprocessor not loaded. Server is not ready."}), 503 # Service Unavailable

    try:
        # Convert Pydantic model to pandas DataFrame row
        input_data_raw = body.model_dump() # For pydantic v2+, use model_dump()
        # input_data_raw = body.dict() # For pydantic v1

        input_df = pd.DataFrame([input_data_raw]) # Convert single dict to DataFrame

        # Preprocess the input data
        # Assuming preprocess_for_prediction works for a single row DataFrame
        processed_features = preprocess_for_prediction(input_df, preprocessor)

        if processed_features.empty:
             return jsonify({"error": "Preprocessing resulted in no valid features for single prediction."}), 400 # Bad Request


        # Make prediction
        # Assuming predict_revenue works for a single row DataFrame
        predictions = predict_revenue(model, processed_features, preprocessor)

        # Assuming a single prediction is returned for single input
        if len(predictions) != 1:
             # This should ideally not happen for a single input
             print(f"Warning: predict_revenue returned {len(predictions)} predictions for single input.")
             return jsonify({"error": "Unexpected number of predictions returned for single input."}), 500 # Internal Server Error


        # Return prediction as JSON
        output = PredictionOutput(predicted_revenue=predictions[0])
        return jsonify(output.model_dump()) # For pydantic v2+, use model_dump()


    except Exception as e:
        print(f"Error during single prediction: {e}", file=sys.stderr) # Print errors to stderr
        traceback.print_exc(file=sys.stderr) # Print traceback to stderr
        return jsonify({"error": "An internal error occurred during single prediction.", "details": str(e)}), 500 # Internal Server Error


@app.route('/predict_batch_json', methods=['POST'])
@validate() # Validate input using Pydantic schema for JSON batch
def predict_batch_json(body: BatchPredictionInput):
     """
     Endpoint for batch predictions using JSON list of inputs (transactional model).
     """
     model = get_model() # Get the transactional model
     preprocessor = get_preprocessor()

     if model is None or preprocessor is None:
        return jsonify({"error": "Transactional model or preprocessor not loaded. Server is not ready."}), 503 # Service Unavailable

     try:
         # Convert list of Pydantic models to pandas DataFrame
         input_list_of_dicts = [item.model_dump() for item in body.data] # pydantic v2+
         input_df_raw = pd.DataFrame(input_list_of_dicts)

         if input_df_raw.empty:
              return jsonify({"predictions": []}), 200 # OK, but no data processed

         # Preprocess the batch data
         processed_features = preprocess_for_prediction(input_df_raw, preprocessor)

         if processed_features.empty:
              return jsonify({"error": "Preprocessing resulted in no valid features for batch JSON."}), 400


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
    Endpoint for batch predictions using CSV file upload (transactional model).
    """
    model = get_model() # Get the transactional model
    preprocessor = get_preprocessor()

    if model is None or preprocessor is None:
        return jsonify({"error": "Transactional model or preprocessor not loaded. Server is not ready."}), 503 # Service Unavailable

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
            # print("API: DataFrame head:", file=sys.stderr) # Debug print (can be verbose)
            # print(input_df_raw.head(), file=sys.stderr) # Debug print
            # print("API: DataFrame columns:", file=sys.stderr) # Debug print
            # print(input_df_raw.columns.tolist(), file=sys.stderr) # Debug print
            # print("API: DataFrame info:", file=sys.stderr) # Debug print
            # input_df_raw.info(buf=sys.stderr) # Print info to stderr


            if input_df_raw.empty:
                 print("API: Read an empty DataFrame from CSV.", file=sys.stderr) # Debug print
                 return jsonify({"predictions": []}), 200 # OK, but no data processed


            print("API: Calling preprocess_for_prediction...", file=sys.stderr) # Debug print
            # Preprocess the batch data
            # Assuming preprocess_for_prediction works for a batch DataFrame
            processed_features = preprocess_for_prediction(input_df_raw, preprocessor)
            print("API: preprocess_for_prediction completed.", file=sys.stderr) # Debug print
            # print("API: Processed features shape:", processed_features.shape, file=sys.stderr) # Debug print
            # print("API: Processed features columns:", processed_features.columns.tolist(), file=sys.stderr) # Debug print


            if processed_features.empty:
                 print("API: Preprocessing resulted in no valid features.", file=sys.stderr) # Debug print
                 return jsonify({"error": "Preprocessing resulted in no valid features for any record from CSV."}), 400


            print("API: Calling predict_revenue...", file=sys.stderr) # Debug print
            # Make predictions for the batch
            # Assuming predict_revenue works for a batch DataFrame
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

@app.route('/forecast_sarima', methods=['POST'])
@validate() # Validate input using Pydantic schema
def forecast_sarima(body: SarimaForecastInput):
    """
    Endpoint for time-series forecasting using the SARIMA model.
    Accepts JSON matching SarimaForecastInput schema (start_date, end_date).
    """
    sarima_model = get_sarima_model()
    # The SARIMA model in this project was trained on daily aggregated revenue
    # It does not require the preprocessor from the transactional model

    if sarima_model is None:
        return jsonify({"error": "SARIMA model not loaded. Server is not ready for forecasting."}), 503 # Service Unavailable

    try:
        start_date_str = body.start_date
        end_date_str = body.end_date

        # Validate date formats and range (basic check)
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            if start_date > end_date:
                 return jsonify({"error": "Start date cannot be after end date."}), 400
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

        # Perform the forecast
        # The SARIMA model's forecast method typically takes steps or a date index
        # Assuming the SARIMA model object has a 'forecast' or 'predict' method
        # that can take a start and end date index.
        # Need to ensure the index matches the training data frequency (daily)
        # Let's create a date range for the forecast period
        forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        if forecast_dates.empty:
             return jsonify({"forecasts": []}), 200 # OK, but no dates in range

        print(f"API: Generating SARIMA forecast from {start_date_str} to {end_date_str}", file=sys.stderr)

        # Assuming the loaded SARIMA model object (from statsmodels) has a predict method
        # that accepts start and end dates or indices.
        # The index should align with the original time series the model was trained on.
        # A common way is to use the index of the last training data point as the start for forecasting.
        # However, for simplicity and assuming the model is ready to forecast from the date after its training data ended,
        # we can pass the forecast_dates directly or calculate steps.
        # Let's assume the model's predict method can take the date index directly.
        # Note: This might require the model object to have access to its training data's end date.
        # A more robust implementation might store the last training date with the model or preprocessor.

        # For demonstration, let's assume the model's predict method takes the date index.
        # If your SARIMA model object requires steps, you'd calculate them based on the last training date.
        # Example if steps are needed:
        # last_train_date = ... # Need to get this from saved preprocessor or elsewhere
        # start_step = (start_date - last_train_date).days
        # end_step = (end_date - last_train_date).days
        # predictions_series = sarima_model.predict(start=start_step, end=end_step)

        # Assuming predict can take date index:
        # Need to handle potential errors if forecast_dates are before the end of training data
        try:
            predictions_series = sarima_model.predict(start=forecast_dates.min(), end=forecast_dates.max())
            print("API: SARIMA prediction complete.", file=sys.stderr)
        except Exception as e:
             print(f"API: Error during SARIMA model prediction: {e}", file=sys.stderr)
             # Handle specific statsmodels errors if needed
             return jsonify({"error": f"Error during SARIMA model prediction: {e}"}), 500


        # Structure the output
        forecast_output_list = []
        for date, predicted_revenue in predictions_series.items():
            forecast_output_list.append(SarimaForecastOutput(
                date=date.strftime("%Y-%m-%d"),
                predicted_revenue=float(predicted_revenue) # Ensure float serializable
            ).model_dump()) # pydantic v2+

        output = BatchSarimaForecastOutput(forecasts=forecast_output_list)

        print("API: Returning SARIMA forecast JSON.", file=sys.stderr)
        return jsonify(output.model_dump()) # pydantic v2+


    except Exception as e:
        print(f"API: Error during SARIMA forecasting: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": "An internal error occurred during SARIMA forecasting.", "details": str(e)}), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """
    Endpoint to provide model information including feature importance.
    This is used for model explainability visualizations.
    """
    model = get_model()  # Get the transactional model (RF or XGBoost)
    preprocessor = get_preprocessor()
    sarima_model = get_sarima_model()  # Get the SARIMA model

    if model is None or preprocessor is None:
        return jsonify({
            "status": "error",
            "details": "Transactional model or preprocessor not loaded"
        }), 503  # Service Unavailable

    try:
        model_info = {
            "model_type": str(type(model).__name__),
            "features": []
        }

        # Get feature names if available
        if hasattr(preprocessor, 'get_feature_names_out'):
            try:
                feature_names = preprocessor.get_feature_names_out()
                model_info["features"] = feature_names.tolist()
            except Exception as e:
                print(f"Error getting feature names: {e}", file=sys.stderr)
                # Continue without feature names
                feature_names = [f"feature_{i}" for i in range(100)]  # Fallback
        else:
            feature_names = [f"feature_{i}" for i in range(100)]  # Fallback

        # Extract feature importance from model if available
        feature_importance = {}
        
        # For Random Forest or similar models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Limit to actual feature count
            for i, importance in enumerate(importances[:len(feature_names)]):
                feature_importance[feature_names[i]] = float(importance)
            
            model_info["feature_importance"] = feature_importance
        
        # For XGBoost model specifically
        if hasattr(model, 'get_booster'):
            try:
                # This is XGBoost-specific
                xgb_importance = model.get_booster().get_score(importance_type='weight')
                model_info["xgb_feature_importance"] = xgb_importance
            except Exception as e:
                print(f"Error getting XGBoost feature importance: {e}", file=sys.stderr)
        
        # SARIMA model info if available
        if sarima_model:
            model_info["sarima_model_loaded"] = True
            
            # Add SARIMA parameters if available
            try:
                model_info["sarima_params"] = {
                    "order": sarima_model.order,
                    "seasonal_order": sarima_model.seasonal_order,
                }
            except Exception as e:
                print(f"Error getting SARIMA parameters: {e}", file=sys.stderr)
        else:
            model_info["sarima_model_loaded"] = False

        return jsonify({
            "status": "ok",
            "model_info": model_info
        })

    except Exception as e:
        print(f"Error getting model info: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify({
            "status": "error",
            "details": f"An error occurred while retrieving model information: {str(e)}"
        }), 500


@app.route('/status', methods=['GET'])
def status():
    """
    Health check endpoint.
    """
    model_loaded = get_model() is not None
    preprocessor_loaded = get_preprocessor() is not None
    sarima_model_loaded = get_sarima_model() is not None # Check SARIMA model status
    status_ok = model_loaded and preprocessor_loaded and sarima_model_loaded
    return jsonify({
        "status": "ok" if status_ok else "loading_error",
        "transactional_model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "sarima_model_loaded": sarima_model_loaded
    }), 200 if status_ok else 503 # Service Unavailable


@app.route('/reload_model', methods=['POST'])
def reload_model():
    """
    Endpoint to manually trigger model and preprocessor reload.
    Useful after deploying a new model version.
    """
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'best_random_forest_revenue_model.pkl') # Transactional model
    PREPROCESSOR_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'preprocessor.pkl') # Preprocessor for transactional model
    SARIMA_MODEL_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'sarima_revenue_model.pkl') # SARIMA model

    try:
        load_latest_model(MODEL_TO_LOAD, PREPROCESSOR_TO_LOAD, SARIMA_MODEL_TO_LOAD) # Update load_latest_model to handle SARIMA
        return jsonify({"message": "Models and preprocessor reloaded successfully."}), 200
    except FileNotFoundError as e:
        return jsonify({"error": f"Model or preprocessor file not found: {e}"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to reload models: {e}"}), 500


@app.route('/shap_values', methods=['POST'])
def get_shap_values():
    """
    Endpoint to calculate and return SHAP values for model explainability.
    """
    model = get_model()
    preprocessor = get_preprocessor()
    
    if model is None or preprocessor is None:
        return jsonify({
            "status": "error",
            "details": "Model or preprocessor not loaded"
        }), 503
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and file.filename.endswith('.csv'):
        try:
            # Read the CSV file
            csv_data = io.BytesIO(file.read())
            input_df = pd.read_csv(csv_data)
            
            # Limit to a reasonable number of rows for SHAP calculation
            if len(input_df) > 100:
                input_df = input_df.sample(100, random_state=42)
            
            # Preprocess the data
            processed_features = preprocess_for_prediction(input_df, preprocessor)
            
            if processed_features.empty:
                return jsonify({"error": "Preprocessing resulted in no valid features"}), 400
            
            # Try to import SHAP
            try:
                import shap
            except ImportError:
                return jsonify({
                    "status": "error",
                    "details": "SHAP library not installed on the server"
                }), 503
            
            # Calculate SHAP values
            if hasattr(model, 'predict_proba'):
                # For classifiers
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(processed_features)
                
                # For multi-class, use the first class or expected value
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                # For regressors
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(processed_features)
            
            # Get feature names
            feature_names = processed_features.columns.tolist()
            
            # Convert SHAP values to a list of dictionaries for JSON serialization
            shap_data = []
            for i in range(len(processed_features)):
                row_data = {
                    "index": i,
                    "base_value": float(explainer.expected_value) if not isinstance(explainer.expected_value, list) else float(explainer.expected_value[0]),
                    "feature_values": {}
                }
                
                for j, feature in enumerate(feature_names):
                    row_data["feature_values"][feature] = {
                        "value": float(processed_features.iloc[i, j]),
                        "shap_value": float(shap_values[i, j])
                    }
                
                shap_data.append(row_data)
            
            # Return the SHAP values
            return jsonify({
                "status": "ok",
                "feature_names": feature_names,
                "shap_data": shap_data
            })
            
        except Exception as e:
            print(f"Error calculating SHAP values: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return jsonify({
                "status": "error",
                "details": f"An error occurred while calculating SHAP values: {str(e)}"
            }), 500
    else:
        return jsonify({"error": "File must be a CSV"}), 400


# --- Main execution ---
if __name__ == '__main__':
    # In production, use a WSGI server like Gunicorn or Waitress.
    # This is for local development testing.
    print("Running Flask app in development mode. Use Gunicorn/Waitress for production.")
    # Ensure models and preprocessor are loaded on startup if not already by import
    if get_model() is None or get_preprocessor() is None or get_sarima_model() is None:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        MODEL_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'best_random_forest_revenue_model.pkl')
        PREPROCESSOR_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'preprocessor.pkl')
        SARIMA_MODEL_TO_LOAD = os.path.join(PROJECT_ROOT, 'models_initial', 'sarima_revenue_model.pkl')
        if os.path.exists(MODEL_TO_LOAD) and os.path.exists(PREPROCESSOR_TO_LOAD) and os.path.exists(SARIMA_MODEL_TO_LOAD):
             load_latest_model(MODEL_TO_LOAD, PREPROCESSOR_TO_LOAD, SARIMA_MODEL_TO_LOAD)
        else:
             print("API: One or more models/preprocessor not found. Cannot start in a ready state.")
             print("Please run training (src/train.py) first.")

    app.run(debug=True, host='0.0.0.0', port=5001) # Use port 5001 for API
