import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.statsmodels # For SARIMA
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # <-- ADD THIS
from datetime import datetime

# Import modularized functions
from src.data_preprocessing import preprocess_data # Need to import the function itself
from src.feature_engineering import add_time_features, add_lag_rolling_features, aggregate_daily_revenue
from src.evaluate import evaluate_model # Import evaluate_model function
# from src.predict_utils import preprocess_for_prediction # Only needed for prediction, not training data prep


def train_models(
    raw_data_path,
    processed_data_path,
    models_output_dir,
    preprocessor_output_path,
    daily_aggregated_path
):
    """
    Runs the full training pipeline: preprocesses data, engineers features,
    trains multiple models, tracks with MLflow, and saves models/preprocessor.
    """
    # Ensure output directories exist
    os.makedirs(models_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(daily_aggregated_path), exist_ok=True)

    # --- Data Preprocessing ---
    # This step also saves PreparedSalesData.csv and preprocessor.pkl
    preprocess_data(raw_data_path, processed_data_path, preprocessor_output_path)

    # --- Feature Engineering ---
    print("Loading processed data for feature engineering...")
    df_processed = pd.read_csv(processed_data_path)

    # Add time features (sets date as index) - this is for RF/XGB data
    df_features = add_time_features(df_processed)

    # --- Prepare data for SARIMA (daily aggregated) ---
    # Need the original Revenue column before scaling for SARIMA aggregation
    # Load raw data again specifically for SARIMA aggregation
    print("Loading raw data for SARIMA aggregation...")
    raw_df_for_sarima = pd.read_csv(raw_data_path)

    # --- FIX: Convert to datetime and set index for SARIMA data ---
    print("Converting 'Transaction_Date' to datetime and setting as index for SARIMA data...")
    raw_df_for_sarima['Transaction_Date'] = pd.to_datetime(raw_df_for_sarima['Transaction_Date'])
    raw_df_for_sarima.set_index('Transaction_Date', inplace=True)
    raw_df_for_sarima = raw_df_for_sarima.sort_index() # Ensure sorted by date


    # Aggregate data to daily frequency for SARIMA
    daily_revenue_series = aggregate_daily_revenue(raw_df_for_sarima) # Pass the DataFrame with DatetimeIndex
    daily_revenue_series.to_csv(daily_aggregated_path, header=['Revenue'])
    print(f"Daily aggregated revenue saved to {daily_aggregated_path}")


    # --- Add Lag/Rolling features for RF/XGB ---
    # Use the dataframe that already has time features and is sorted by date index
    df_all_features = add_lag_rolling_features(df_features)

    # Handle NAs created by lag/rolling features - drop them *after* adding features
    # This needs to be done *before* train/test split
    print("Dropping rows with NAs created by lag/rolling features...")
    initial_na_rows = df_all_features.isnull().any(axis=1).sum()
    df_all_features.dropna(inplace=True)
    print(f"Dropped {initial_na_rows} rows.")


    # --- Train/Test Split ---
    print("Performing train/test split (last 6 months as test set)...")
    split_date = df_all_features.index.max() - pd.DateOffset(months=6)
    train_df = df_all_features[df_all_features.index <= split_date].copy() # Use .copy() to avoid SettingWithCopyWarning
    test_df = df_all_features[df_all_features.index > split_date].copy()

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")


    # --- Feature Selection ---
    # Identify feature columns - all columns except the target 'Revenue'
    feature_columns = [col for col in train_df.columns if col != 'Revenue']
    target_column = 'Revenue'

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    # --- Save the list of feature columns used for training ---
    # This is needed by the preprocessor for column alignment during prediction/monitoring
    preprocessor = joblib.load(preprocessor_output_path)
    preprocessor['feature_columns'] = feature_columns
    joblib.dump(preprocessor, preprocessor_output_path)
    print(f"Feature columns list saved in {preprocessor_output_path}")


    # --- Train Models with MLflow ---
    print("Starting MLflow experiment...")
    # Set MLflow experiment name
    mlflow.set_experiment("Sales Demand Forecasting")

    with mlflow.start_run(run_name="Model Training"):
        # Log parameters (example)
        mlflow.log_param("train_start_date", train_df.index.min().strftime('%Y-%m-%d'))
        mlflow.log_param("train_end_date", train_df.index.max().strftime('%Y-%m-%d'))
        mlflow.log_param("test_start_date", test_df.index.min().strftime('%Y-%m-%d'))
        mlflow.log_param("test_end_date", test_df.index.max().strftime('%Y-%m-%d'))
        mlflow.log_param("features_count", len(feature_columns))
        mlflow.log_param("feature_list", feature_columns) # Log the full list

        # --- Random Forest ---
        print("Training Random Forest model...")
        rf_params = {'n_estimators': 100, 'random_state': 42}
        mlflow.log_params({f"rf_{k}": v for k, v in rf_params.items()})
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train)

        # Evaluate RF
        print("Evaluating Random Forest model...")
        rf_metrics, rf_predictions = evaluate_model(rf_model, X_test, y_test)
        mlflow.log_metrics({f"rf_test_{k}": v for k, v in rf_metrics.items()})

        # Log RF model artifact
        mlflow.sklearn.log_model(rf_model, "random_forest_model")

        # Save RF model as pkl
        rf_model_path = os.path.join(models_output_dir, 'best_random_forest_revenue_model.pkl')
        joblib.dump(rf_model, rf_model_path)
        print(f"Random Forest model saved to {rf_model_path}")

        # --- XGBoost ---
        print("Training XGBoost model...")
        xgb_params = {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train)

        # Evaluate XGBoost
        print("Evaluating XGBoost model...")
        xgb_metrics, xgb_predictions = evaluate_model(xgb_model, X_test, y_test)
        mlflow.log_metrics({f"xgb_test_{k}": v for k, v in xgb_metrics.items()})

        # Log XGBoost model artifact
        mlflow.xgboost.log_model(xgb_model, "xgboost_model")

        # Save XGBoost model as pkl
        xgb_model_path = os.path.join(models_output_dir, 'xgboost_revenue_model.pkl')
        joblib.dump(xgb_model, xgb_model_path)
        print(f"XGBoost model saved to {xgb_model_path}")

        # --- SARIMA ---
        print("Training SARIMA model...")
        # SARIMA uses the daily aggregated data
        # Need to split the daily aggregated data based on the same time split
        # Use the index from the test_df (which has the correct split date) to find the split point
        sarima_train = daily_revenue_series[daily_revenue_series.index <= split_date]
        sarima_test = daily_revenue_series[daily_revenue_series.index > split_date]


        sarima_order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12) # Assuming monthly seasonality

        mlflow.log_params({
            'sarima_order': str(sarima_order),
            'sarima_seasonal_order': str(seasonal_order)
        })

        try:
            sarima_model = SARIMAX(sarima_train,
                                   order=sarima_order,
                                   seasonal_order=seasonal_order,
                                   enforce_stationarity=False, # Set to False as in notebook
                                   enforce_invertibility=False)
            sarima_results = sarima_model.fit(disp=False) # disp=False to reduce output

            # Make SARIMA forecasts on the test period
            # Use predict method which respects index frequency
            sarima_forecast = sarima_results.predict(start=sarima_test.index.min(), end=sarima_test.index.max())

            # Evaluate SARIMA
            # SARIMA predicts the series value, evaluation is direct comparison
            sarima_mae = mean_absolute_error(sarima_test, sarima_forecast)
            sarima_rmse = np.sqrt(mean_squared_error(sarima_test, sarima_forecast))
            sarima_r2 = r2_score(sarima_test, sarima_forecast)
            sarima_metrics = {
                'mae': sarima_mae,
                'rmse': sarima_rmse,
                'r2': sarima_r2
            }
            print(f"SARIMA Metrics: MAE={sarima_mae:.4f}, RMSE={sarima_rmse:.4f}, R2={sarima_r2:.4f}")
            mlflow.log_metrics({f"sarima_test_{k}": v for k, v in sarima_metrics.items()})

            # Log SARIMA model artifact
            mlflow.statsmodels.log_model(sarima_results, "sarima_model")

            # Save SARIMA model as pkl (joblib might work for statsmodels results)
            sarima_model_path = os.path.join(models_output_dir, 'sarima_revenue_model.pkl')
            joblib.dump(sarima_results, sarima_model_path)
            print(f"SARIMA model saved to {sarima_model_path}")

        except Exception as e:
            print(f"Error training or evaluating SARIMA model: {e}")
            mlflow.log_text(str(e), "sarima_error.txt")
            sarima_metrics = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan} # Log NaNs on error
            mlflow.log_metrics({f"sarima_test_{k}": v for k, v in sarima_metrics.items()})
            sarima_forecast = pd.Series() # Empty series if failed


        # --- Log Baseline Metrics for Monitoring ---
        # We will use the test set metrics from the best model (e.g., RF or XGB) as baseline
        # Let's pick RF as the default deployed model unless specified otherwise.
        baseline_metrics = {
            'r2': rf_metrics.get('r2', 0.0), # Use get with default in case of error
            'mae': rf_metrics.get('mae', np.inf)
        }
        # Ensure the directory exists before saving
        baseline_metrics_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
        os.makedirs(baseline_metrics_dir, exist_ok=True)
        baseline_metrics_path = os.path.join(baseline_metrics_dir, "baseline_metrics.json")
        with open(baseline_metrics_path, 'w') as f:
            import json
            json.dump(baseline_metrics, f)

        mlflow.log_dict(baseline_metrics, "baseline_metrics.json")
        print(f"Baseline metrics for monitoring logged: {baseline_metrics}")


        print("MLflow run completed.")
        print(f"MLflow tracking UI available at http://localhost:5000 (if running mlflow ui)")


if __name__ == "__main__":
    # Define paths relative to the project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'synthetic_ecommerce_data.csv')
    PROCESSED_DATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'PreparedSalesData.csv')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models_initial')
    PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models_initial', 'preprocessor.pkl')
    DAILY_AGGREGATED_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'daily_aggregated_revenue.csv')

    # Ensure raw data exists
    if not os.path.exists(RAW_DATA_CSV):
        print(f"Error: Raw data file not found at {RAW_DATA_CSV}")
        print("Please place synthetic_ecommerce_data.csv in data/synthetic/")
    else:
        train_models(RAW_DATA_CSV, PROCESSED_DATA_CSV, MODELS_DIR, PREPROCESSOR_PATH, DAILY_AGGREGATED_CSV)
