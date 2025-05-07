import pandas as pd
import numpy as np
import os
import json
import mlflow
from datetime import datetime
import time # For sleep

# Import necessary modules from src
from src.predict_utils import load_model_and_preprocessor, preprocess_for_prediction, predict_revenue # Need preprocess_for_prediction
from src.evaluate import perform_monitoring # Import the monitoring logic


def run_monitoring_check(
    new_data_path,
    model_path, # Path to the model pkl file
    preprocessor_path, # Path to the preprocessor pkl file
    baseline_metrics_path, # Path to saved baseline metrics (e.g., json artifact)
    alert_thresholds, # Dictionary of thresholds
    report_output_dir, # Directory to save reports
    alert_log_path # Path for the alert log file
):
    """
    Loads new data, runs the monitoring evaluation, logs results, and triggers alerts.
    """
    print(f"Running scheduled monitoring check at {datetime.now().isoformat()}")

    # Set MLflow experiment name for monitoring runs
    mlflow.set_experiment("Sales Demand Forecasting - Monitoring")

    # Start a new MLflow run for this monitoring check
    with mlflow.start_run(run_name=f"Monitoring Check {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):

        # Log monitoring run parameters (optional, but good practice)
        mlflow.log_param("monitoring_data_source", new_data_path)
        mlflow.log_param("model_used", os.path.basename(model_path))
        mlflow.log_param("preprocessor_used", os.path.basename(preprocessor_path))


        # --- Load Model and Preprocessor ---
        try:
            model, preprocessor = load_model_and_preprocessor(model_path, preprocessor_path)
        except FileNotFoundError as e:
            print(f"Error loading model or preprocessor: {e}")
            mlflow.log_text(f"Error loading model/preprocessor: {e}", "error_log.txt")
            return # Cannot proceed without model/preprocessor
        except Exception as e:
             print(f"An unexpected error occurred loading model/preprocessor: {e}")
             mlflow.log_text(f"Unexpected error loading model/preprocessor: {e}", "error_log.txt")
             return


        # --- Load New Data ---
        try:
            # In a real scenario, this would load data from a database, stream, etc.
            # For this example, load a portion of the test set or a separate 'new' file.
            # Let's load a segment of the test data saved from the training run.
            # Need to know where the test data is saved or have a dedicated new data source.
            # For this example, let's load the processed data, split again and use the last part
            # or load from a pre-saved 'new_data_sample.csv' if available.
            # Assuming a 'new_data_sample.csv' exists for simplicity.
            # OR load processed data and take a recent slice.
            # Let's assume we load the PreparedSalesData and take the very last segment
            # that wasn't used in the original test set, simulating "new" data.
            # This requires saving the full processed data with date index preserved.
            # Let's modify data_preprocessing to save with date index.

            # Assuming for this example, `new_data_path` points to the original processed CSV,
            # and we'll take a recent slice from it that wasn't in the train/test split.
            # A better approach is a dedicated new data file. Let's simulate new data.

            # Simulation: Load the original processed data and take the *very last* few entries
            # that weren't in the test set (which was the last 6 months). This is imperfect.
            # A more realistic simulation is to generate new data or load a separate file.
            # Let's instruct the user to place a `new_data_sample.csv` in `data/processed/`.
            # This file should have the same column structure as the *raw* data (including Transaction_Date, Category, etc.).

            if not os.path.exists(new_data_path):
                 print(f"Error: New data sample file not found at {new_data_path}")
                 mlflow.log_text(f"New data sample not found: {new_data_path}", "error_log.txt")
                 print("Please create a 'new_data_sample.csv' in data/processed/ with recent data.")
                 return

            new_data_df_raw = pd.read_csv(new_data_path)

        except FileNotFoundError as e:
            print(f"Error loading new data: {e}")
            mlflow.log_text(f"Error loading new data: {e}", "error_log.txt")
            return
        except Exception as e:
             print(f"An unexpected error occurred loading new data: {e}")
             mlflow.log_text(f"Unexpected error loading new data: {e}", "error_log.txt")
             return


        # --- Load Baseline Metrics ---
        try:
            with open(baseline_metrics_path, 'r') as f:
                baseline_metrics = json.load(f)
            print(f"Loaded baseline metrics: {baseline_metrics}")
            mlflow.log_dict(baseline_metrics, "loaded_baseline_metrics.json")
        except FileNotFoundError:
            print(f"Warning: Baseline metrics file not found at {baseline_metrics_path}. Using default thresholds.")
            baseline_metrics = {} # Use empty dict if file not found
        except json.JSONDecodeError:
             print(f"Warning: Error decoding JSON from {baseline_metrics_path}. Using empty baseline metrics.")
             baseline_metrics = {}
        except Exception as e:
             print(f"An unexpected error occurred loading baseline metrics: {e}")
             baseline_metrics = {}


        # --- Perform Monitoring ---
        # The perform_monitoring function expects the data *including actuals*
        # and handles preprocessing internally using the provided preprocessor.
        try:
            perform_monitoring(
                new_data_df_raw, # Pass the raw-like new data
                model,
                preprocessor,
                baseline_metrics,
                alert_thresholds,
                report_output_dir,
                alert_log_path,
                run_name=f"Check {datetime.now().strftime('%Y-%m-%d')}"
            )
            mlflow.log_text("Monitoring check completed successfully.", "status.txt")

        except Exception as e:
            print(f"An error occurred during the monitoring process: {e}")
            mlflow.log_text(f"Error during monitoring: {e}", "error_log.txt")
            # Optionally, trigger a critical alert if monitoring itself fails


    print("Monitoring check finished.")


if __name__ == "__main__":
    # Define paths relative to the project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Assuming the model and preprocessor saved by train.py are used for monitoring
    MODEL_PATH_RF = os.path.join(PROJECT_ROOT, 'models_initial', 'best_random_forest_revenue_model.pkl') # Use the full model trained with lags/rolling? Or the base model?
    # For monitoring on "new data", we should ideally use the model that predicts
    # on data *including* lags/rolling, as the evaluation data can have these computed.
    # So, use the full model trained in train.py.
    PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models_initial', 'preprocessor.pkl')
    # Baseline metrics are saved as an artifact in the training run.
    # We need to know the run ID or locate the latest training run artifact.
    # For simplicity in this script, let's assume the baseline metrics were copied
    # from the MLflow run artifact to a known path or hardcoded.
    # Let's manually specify the path where the baseline metrics artifact *would* be saved
    # if downloaded from MLflow, or hardcode values. Hardcoding is simplest for the example.

    # --- Hardcoded Baseline Metrics (Obtained from a training run's test evaluation) ---
    # REPLACE THESE WITH ACTUAL VALUES FROM YOUR TRAINING RUN ARTIFACT
    # Example values - YOU MUST UPDATE THESE AFTER RUNNING train.py
    baseline_metrics_example = {
        'r2': 0.85, # Example R2 from test set
        'mae': 50.0 # Example MAE from test set
    }
    # Optionally, save these to a local file for monitor.py to load
    BASELINE_METRICS_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'baseline_metrics.json')
    os.makedirs(os.path.dirname(BASELINE_METRICS_FILE), exist_ok=True)
    with open(BASELINE_METRICS_FILE, 'w') as f:
         json.dump(baseline_metrics_example, f)
    print(f"Example baseline metrics saved to {BASELINE_METRICS_FILE}")
    # monitor.py will load from BASELINE_METRICS_FILE

    # --- Alert Thresholds ---
    alert_thresholds_config = {
        'r2_min': 0.7, # Minimum acceptable R2
        'mae_max': baseline_metrics_example['mae'] * 1.1 # Allow 10% increase over baseline MAE
    }

    # --- New Data Source ---
    # For demonstration, use a sample file that represents "new" data.
    # This file should ideally have the same structure as the raw input data.
    # Create a dummy file if it doesn't exist or instruct user.
    NEW_DATA_SAMPLE_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'new_data_sample.csv')
    if not os.path.exists(NEW_DATA_SAMPLE_CSV):
         print(f"Creating a dummy new data sample at {NEW_DATA_SAMPLE_CSV}")
         # Load some data from the original, filter by date to make it "new"
         try:
             original_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'synthetic_ecommerce_data.csv'))
             original_data['Transaction_Date'] = pd.to_datetime(original_data['Transaction_Date'])
             # Take the last few rows to simulate new data
             new_sample = original_data.tail(50).copy()
             new_sample.to_csv(NEW_DATA_SAMPLE_CSV, index=False)
             print("Dummy new data sample created.")
         except FileNotFoundError:
             print("Error: Original synthetic data not found. Cannot create dummy new data sample.")
             NEW_DATA_SAMPLE_CSV = None # Indicate no new data file
         except Exception as e:
             print(f"Error creating dummy new data sample: {e}")
             NEW_DATA_SAMPLE_CSV = None

    REPORT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'performance_reports')
    ALERT_LOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'monitoring_alerts.log')


    # Ensure required model/preprocessor files exist before attempting monitoring
    if not os.path.exists(MODEL_PATH_RF):
        print(f"Error: Model file not found at {MODEL_PATH_RF}")
    if not os.path.exists(PREPROCESSOR_PATH):
        print(f"Error: Preprocessor file not found at {PREPROCESSOR_PATH}")
    if not os.path.exists(BASELINE_METRICS_FILE):
         print(f"Error: Baseline metrics file not found at {BASELINE_METRICS_FILE}")
         print("Please ensure train.py runs successfully and saves baseline_metrics.json.")
    if NEW_DATA_SAMPLE_CSV is None or not os.path.exists(NEW_DATA_SAMPLE_CSV):
         print("Error: New data sample file not available. Monitoring cannot run.")
    else:
        # Run the monitoring check
        # This could be run in a loop for continuous monitoring, but for demonstration, run once.
        run_monitoring_check(
            NEW_DATA_SAMPLE_CSV,
            MODEL_PATH_RF,
            PREPROCESSOR_PATH,
            BASELINE_METRICS_FILE,
            alert_thresholds_config,
            REPORT_OUTPUT_DIR,
            ALERT_LOG_PATH
        )
        # Example of continuous monitoring (uncomment and adjust interval)
        # while True:
        #     run_monitoring_check(...)
        #     time.sleep(86400) # Check daily (86400 seconds)