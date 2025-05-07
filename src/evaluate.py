import pandas as pd
import numpy as np
import os
import json
import mlflow
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X, y):
    """
    Evaluates a given model on features X and target y.
    Returns a dictionary of evaluation metrics.
    """
    print("Evaluating model...")
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    print(f"Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    return metrics, y_pred

def perform_monitoring(
    new_data_df, # DataFrame of new data including actuals
    model,
    preprocessor,
    baseline_metrics,
    alert_thresholds,
    report_output_dir,
    alert_log_path,
    run_name="Monitoring Check"
):
    """
    Performs a monitoring check on new data using the deployed model.
    Evaluates performance, detects drift, triggers alerts, and generates reports.
    """
    print(f"Running monitoring check '{run_name}'...")

    if new_data_df.empty:
        print("No new data provided for monitoring.")
        return

    # --- Preprocess New Data ---
    # Need to separate features (X_new) and actual target (y_new_actual)
    # Assume new_data_df contains all original columns + Revenue
    if 'Revenue' not in new_data_df.columns:
         print("Warning: 'Revenue' column not found in new data. Cannot calculate performance metrics.")
         # Can still make predictions, but cannot evaluate performance
         y_new_actual = None
         X_new = new_data_df.copy()
    else:
        y_new_actual = new_data_df['Revenue']
        X_new = new_data_df.drop('Revenue', axis=1) # Drop target for prediction input

    # Apply the same preprocessing and feature engineering as training
    # This assumes new_data_df structure is similar to the training data input
    # Reapply time features (requires 'Transaction_Date')
    if 'Transaction_Date' in X_new.columns:
        X_new = add_time_features(X_new) # This sets index to Date
    elif isinstance(X_new.index, pd.DatetimeIndex):
         pass # Date is already index
    else:
         print("Warning: Cannot add time features. 'Transaction_Date' not found and index is not datetime.")

    # Apply lag/rolling features. This is tricky for monitoring on a single batch.
    # For simplicity, we assume the input new_data_df for monitoring *already includes*
    # pre-calculated lag/rolling features based on its historical context.
    # A real-world scenario needs access to recent historical data.
    # Let's apply feature_engineering.add_lag_rolling_features assuming it works on the input structure
    # and then drop NaNs if any are created (common at the start of the batch).
    if isinstance(X_new.index, pd.DatetimeIndex): # Only add lags if date index exists
        X_new = add_lag_rolling_features(X_new)
        initial_rows_with_na = X_new.isnull().any(axis=1).sum()
        if initial_rows_with_na > 0:
            print(f"Warning: Dropping {initial_rows_with_na} rows with NAs created by lag/rolling features.")
            # Align y_new_actual if dropping rows from X_new
            if y_new_actual is not None:
                 y_new_actual = y_new_actual.loc[X_new.dropna().index]
            X_new = X_new.dropna()


    # Apply preprocessing using the loaded preprocessor (categorical encoding, scaling)
    # This requires features in X_new to match columns expected by preprocessor
    # Need a way to ensure columns match - a simple approach is to reindex X_new
    # based on the columns the preprocessor was fitted on during training.
    # This requires saving the training feature columns. Let's add this to preprocessor.pkl.
    # Assuming the preprocessor object now has a 'feature_columns' attribute.
    if not hasattr(preprocessor, 'feature_columns'):
        print("Error: Preprocessor object does not have 'feature_columns' attribute. Cannot align features.")
        # Cannot proceed with prediction/evaluation reliably
        return

    # Reindex X_new to match the training feature columns, filling missing with 0 (for one-hot encoded)
    # or appropriate defaults for other features. This is a simplification.
    # A robust system needs careful feature store or data alignment.
    X_new_aligned = pd.DataFrame(index=X_new.index)
    # Ensure the date index is preserved before adding columns back
    temp_index = X_new.index

    # Apply preprocessor transformations
    # Need a dedicated function in predict_utils or here to apply the saved preprocessor
    # Let's create a simple application logic here for now, assuming preprocessor has
    # encoder and scaler and column lists.
    # This needs to handle categorical and numerical columns based on the preprocessor state.

    # Simplified application of preprocessor:
    # 1. Handle Categorical: If original categorical cols are in input, one-hot encode
    # 2. Handle Numerical: Apply scaling (log1p then minmax)
    # 3. Ensure final columns match training features

    # Re-applying preprocessing transformations requires the original columns *before* one-hot/scaling.
    # This means the `new_data_df` should ideally contain the raw-like features.
    # Let's revise: `predict_utils.preprocess_for_prediction` should take raw-ish data
    # and apply all steps from scratch but using the *fitted* preprocessor.

    # Load preprocessor details
    encoder = preprocessor.get('encoder')
    scaler = preprocessor.get('scaler')
    cat_cols_trained = preprocessor.get('categorical_columns', [])
    log_cols_trained = preprocessor.get('log_cols', [])
    minmax_cols_trained = preprocessor.get('minmax_cols', [])
    dropped_corr_cols_trained = preprocessor.get('dropped_corr_cols', [])
    train_feature_columns = preprocessor.get('feature_columns') # Assuming this is saved now

    if train_feature_columns is None:
         print("Error: Training feature columns not saved in preprocessor.pkl. Cannot monitor.")
         return

    # Apply preprocessing logic (simplified - a robust system needs dedicated function)
    # Drop columns that were dropped during training
    data_for_pred = new_data_df.drop(columns=dropped_corr_cols_trained, errors='ignore')
    if 'Revenue' in data_for_pred.columns:
        data_for_pred = data_for_pred.drop('Revenue', axis=1) # Drop target

    # Re-apply time features (needs 'Transaction_Date' in input)
    if 'Transaction_Date' in data_for_pred.columns:
        data_for_pred = add_time_features(data_for_pred)
    elif isinstance(data_for_pred.index, pd.DatetimeIndex):
        pass
    else:
        print("Warning: Cannot add time features for prediction preprocessing.")
        # If date index doesn't exist, lag/rolling features cannot be computed dynamically.
        # Assuming they are pre-calculated in the input for simplicity here.

    # Re-apply lag/rolling features (assuming date index is set)
    if isinstance(data_for_pred.index, pd.DatetimeIndex):
        data_for_pred = add_lag_rolling_features(data_for_pred)
        # Drop NaNs from lags/rolling - this means predictions are only possible after enough history
        initial_rows_with_na = data_for_pred.isnull().any(axis=1).sum()
        if initial_rows_with_na > 0:
            print(f"Warning: Dropping {initial_rows_with_na} rows for prediction due to lag/rolling NAs.")
            if y_new_actual is not None:
                 y_new_actual = y_new_actual.loc[data_for_pred.dropna().index]
            data_for_pred = data_for_pred.dropna()

    # Apply categorical encoding
    if encoder and cat_cols_trained:
         # Need to handle unseen categories if handle_unknown='ignore'
         try:
             one_hot_encoded = encoder.transform(data_for_pred[cat_cols_trained])
             one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(cat_cols_trained), index=data_for_pred.index)
             data_for_pred = data_for_pred.drop(cat_cols_trained, axis=1)
             data_for_pred = pd.concat([data_for_pred, one_hot_df], axis=1)
         except ValueError as e:
             print(f"Error during categorical encoding: {e}. This might be due to new categories.")
             # Handle or skip predictions for this batch if encoding fails critically
             return


    # Apply scaling (log1p then minmax)
    # Apply log1p first (ensure cols exist)
    log_cols_apply = [col for col in log_cols_trained if col in data_for_pred.columns]
    if log_cols_apply:
        data_for_pred[log_cols_apply] = data_for_pred[log_cols_apply].apply(np.log1p)

    # Apply minmax scaling (ensure cols exist)
    minmax_cols_apply = [col for col in minmax_cols_trained if col in data_for_pred.columns]
    if scaler and minmax_cols_apply:
        try:
            # Scaler expects input with the same columns as fitted on, in the same order.
            # This is a common issue. A robust preprocessor handles column alignment.
            # Simple fix: temporary DataFrame with only cols to scale, then put back.
            temp_df_for_scaling = data_for_pred[minmax_cols_apply].copy()
            data_for_pred[minmax_cols_apply] = scaler.transform(temp_df_for_scaling)
        except ValueError as e:
            print(f"Error during scaling: {e}. This might be due to missing columns or unexpected values.")
            # Cannot proceed reliably
            return


    # Reindex features to match the exact training feature columns order
    # This is critical for models like Random Forest/XGBoost
    missing_cols = set(train_feature_columns) - set(data_for_pred.columns)
    for c in missing_cols:
        data_for_pred[c] = 0 # Assume missing one-hot encoded columns are 0

    X_new_processed = data_for_pred[train_feature_columns] # Ensure order matches training


    # --- Make Predictions ---
    print("Making predictions on new data...")
    y_new_pred_scaled = model.predict(X_new_processed)

    # Inverse transform the predictions if Revenue was log-transformed
    if 'Revenue' in log_cols_trained:
        y_new_pred = np.expm1(y_new_pred_scaled)
    else:
        y_new_pred = y_new_pred_scaled

    # --- Evaluate Performance (if actuals are available) ---
    current_metrics = {}
    if y_new_actual is not None and not y_new_actual.empty:
        print("Evaluating performance on new data...")
        # Align actuals with predictions after any row dropping
        y_new_actual_aligned = y_new_actual.loc[X_new_processed.index]
        current_metrics, _ = evaluate_model(model, X_new_processed, y_new_actual_aligned)

        # --- Log Monitoring Metrics to MLflow ---
        with mlflow.start_run(run_name=run_name, nested=True): # Nested run under a parent monitoring run if desired
            mlflow.log_params({
                'monitoring_data_points': len(X_new_processed),
                'monitoring_start_date': X_new_processed.index.min().strftime('%Y-%m-%d') if not X_new_processed.empty else 'N/A',
                'monitoring_end_date': X_new_processed.index.max().strftime('%Y-%m-%d') if not X_new_processed.empty else 'N/A',
            })
            mlflow.log_metrics(current_metrics)
            print(f"Logged monitoring metrics to MLflow run: {mlflow.active_run().info.run_id}")

    else:
        print("Actual revenue data not available. Skipping performance evaluation.")
        # Can still log prediction count etc.

    # --- Generate Performance Report ---
    print("Generating performance report...")
    report_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    report_filename = f"performance_report_{report_timestamp}.csv"
    report_path = os.path.join(report_output_dir, report_filename)
    os.makedirs(report_output_dir, exist_ok=True)

    report_data = {
        'Metric': list(current_metrics.keys()),
        'Value': list(current_metrics.values())
    }
    # Add metrics per category/region if implemented (requires joining predictions back to original new_data_df)
    # For simplicity, let's just report overall metrics for now.
    pd.DataFrame(report_data).to_csv(report_path, index=False)
    print(f"Performance report saved to {report_path}")


    # --- Simple Alert System ---
    print("Checking alert thresholds...")
    alert_triggered = False
    alert_messages = []

    if y_new_actual is not None and not y_new_actual.empty and current_metrics:
        baseline_r2 = baseline_metrics.get('r2', -np.inf) # Use -inf if baseline not available
        baseline_mae = baseline_metrics.get('mae', np.inf)  # Use inf if baseline not available

        if current_metrics.get('r2', -np.inf) < alert_thresholds.get('r2_min', 0.7) and current_metrics.get('r2', -np.inf) < baseline_r2 * 0.9: # Alert if R2 drops below threshold AND significantly below baseline
            msg = f"ALERT: R2 ({current_metrics['r2']:.4f}) dropped below threshold ({alert_thresholds.get('r2_min', 0.7):.4f}) and baseline ({baseline_r2:.4f})."
            alert_messages.append(msg)
            alert_triggered = True

        if current_metrics.get('mae', np.inf) > alert_thresholds.get('mae_max', baseline_mae * 1.1 if baseline_mae != np.inf else np.inf) and current_metrics.get('mae', np.inf) > baseline_mae * 1.1: # Alert if MAE exceeds threshold AND significantly above baseline
             msg = f"ALERT: MAE ({current_metrics['mae']:.4f}) exceeded threshold ({alert_thresholds.get('mae_max', baseline_mae * 1.1 if baseline_mae != np.inf else 'N/A'):.4f}) and baseline ({baseline_mae:.4f})."
             alert_messages.append(msg)
             alert_triggered = True

        # Flag high residuals
        residuals = y_new_actual_aligned - y_new_pred
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        high_residual_threshold = 2 * residual_std # 2 standard deviations

        high_residuals_indices = residuals[abs(residuals - residual_mean) > high_residual_threshold].index.tolist()

        if high_residuals_indices:
             msg = f"WARNING: {len(high_residuals_indices)} predictions have high residuals (> 2 std dev)."
             alert_messages.append(msg)
             # Log indices or examples to a file or MLflow artifact
             high_res_log = f"High residual predictions indices on {datetime.now().isoformat()}: {high_residuals_indices}\n"
             with open(alert_log_path.replace('.log', '_high_res.log'), 'a') as f:
                 f.write(high_res_log)


    if alert_messages:
        alert_log_entry = f"[{datetime.now().isoformat()}] Monitoring Alerts:\n" + "\n".join(alert_messages) + "\n---\n"
        print(alert_log_entry)
        # Append to a simple log file
        os.makedirs(os.path.dirname(alert_log_path), exist_ok=True)
        with open(alert_log_path, 'a') as f:
            f.write(alert_log_entry)
    else:
        print("No alerts triggered.")


# Helper function for feature engineering needed within evaluation monitoring
# (Copying the necessary parts from feature_engineering.py to avoid circular imports if evaluate needed feature_engineering)
# A cleaner approach is to refactor feature engineering application into predict_utils.
# Let's add necessary functions from feature_engineering here directly for now for simplicity
# but note that predict_utils should ideally handle this application using the preprocessor.

def add_time_features(df):
    """Helper: Adds time-based features. Assumes 'Transaction_Date' or DatetimeIndex."""
    df_copy = df.copy() # Work on a copy
    if 'Transaction_Date' in df_copy.columns:
        df_copy['Transaction_Date'] = pd.to_datetime(df_copy['Transaction_Date'])
        df_copy.set_index('Transaction_Date', inplace=True)
    elif not isinstance(df_copy.index, pd.DatetimeIndex):
         # Cannot add time features without date
         return df_copy # Return original if no date info

    df_copy['month'] = df_copy.index.month
    df_copy['dayofweek'] = df_copy.index.dayofweek
    try: df_copy['weekofyear'] = df_copy.index.isocalendar().week.astype(int)
    except AttributeError: df_copy['weekofyear'] = df_copy.index.weekofyear
    df_copy['quarter'] = df_copy.index.quarter
    df_copy['is_weekend'] = df_copy['dayofweek'].isin([5, 6]).astype(int)

    holidays = pd.to_datetime(['2024-04-10', '2024-06-16']) # Example holidays
    df_copy['is_holiday'] = df_copy.index.isin(holidays).astype(int)

    def get_season(month):
        if month in [12, 1, 2]: return 'winter'
        if month in [3, 4, 5]: return 'spring'
        if month in [6, 7, 8]: return 'summer'
        if month in [9, 10, 11]: return 'autumn'
    df_copy['season'] = df_copy['month'].apply(get_season)
    df_copy = pd.get_dummies(df_copy, columns=['season'], drop_first=True)

    return df_copy

def add_lag_rolling_features(df):
    """Helper: Adds lag and rolling window features. Assumes DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
         # Cannot add lag/rolling without date index
         return df

    df_copy = df.copy().sort_index()

    for lag in [1, 7, 14, 30]:
        df_copy[f'revenue_lag_{lag}'] = df_copy['Revenue'].shift(lag)
        df_copy[f'units_lag_{lag}'] = df_copy['Units_Sold'].shift(lag)

    for window in [7, 14, 30]:
        df_copy[f'revenue_rollmean_{window}'] = df_copy['Revenue'].rolling(window).mean()
        df_copy[f'units_rollmean_{window}'] = df_copy['Units_Sold'].rolling(window).mean()

    return df_copy

if __name__ == "__main__":
    # Example Usage for evaluate_model (e.g., from train script)
    # This block is mainly for testing the evaluate_model function in isolation
    # Monitoring is tested via monitor.py

    print("Testing evaluate.py - evaluate_model function...")
    # Create dummy data for testing
    from sklearn.linear_model import LinearRegression
    X_dummy = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [5, 4, 3, 2, 1]})
    y_dummy = pd.Series([10, 12, 15, 17, 20]) # Simple linear relationship + noise

    # Train a dummy model
    dummy_model = LinearRegression()
    dummy_model.fit(X_dummy, y_dummy)

    # Evaluate the dummy model
    metrics, predictions = evaluate_model(dummy_model, X_dummy, y_dummy)
    print("Dummy Model Metrics:", metrics)
    print("Dummy Predictions:", predictions)

    # Example of how monitoring *would* be called (details in monitor.py)
    # print("\nTesting evaluate.py - perform_monitoring function (partial test)...")
    # # Need dummy data, model, preprocessor, baselines, thresholds, paths
    # # This is best tested end-to-end via monitor.py
    # pass