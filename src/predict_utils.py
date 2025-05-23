import pandas as pd
import numpy as np
import joblib
import os
import warnings # To manage warnings if needed
from datetime import datetime # Import datetime for date handling in SARIMA prediction

# Import necessary preprocessing and feature engineering functions/classes
# Note: We need to *apply* the fitted preprocessor, not refit it.
# We also need to apply the *same* feature engineering steps (time/season for simplified real-time).
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # Needed for type hinting/structure

# Note: Model and preprocessor loading is handled by model_loader.py
# We will import necessary getters from there.
from src.api.model_loader import get_preprocessor, get_feature_columns # Import necessary getters


# Define columns for different scaling methods (should match data_preprocessing)
# These are the names *before* any transformation or dropping.
# Assuming these are consistent with how the preprocessor was fitted.
# These are used here for context and potentially for identifying columns to scale,
# but the actual lists used should come from the loaded preprocessor dictionary.
LOG_COLS_PREPROC = ['Ad_Spend', 'Revenue', 'Conversion_Rate']
MINMAX_COLS_PREPROC = ['Discount_Applied', 'Clicks', 'Impressions', 'Ad_CTR', 'Ad_CPC', 'Units_Sold'] # Included Units_Sold for scaling consistency


# Note: load_model_and_preprocessor is not used by the API via model_loader.py
# It's kept here for potential standalone script usage or testing.
def load_model_and_preprocessor(model_path, preprocessor_path):
    """
    Loads the trained transactional model and the fitted preprocessor objects.
    Note: This function is primarily for scripts like train.py or monitor.py.
    The API loads models via model_loader.py.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(preprocessor_path):
         raise FileNotFoundError(f"Preprocessor file not found at: {preprocessor_path}")

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded.")

    print(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)
    print("Preprocessor loaded.")

    # Basic check for essential preprocessor components
    required_preproc_keys = ['encoder', 'scaler', 'categorical_columns', 'log_cols', 'minmax_cols', 'dropped_corr_cols', 'feature_columns']
    if not all(key in preprocessor for key in required_preproc_keys):
         missing_keys = [key for key in required_preproc_keys if key not in preprocessor]
         raise ValueError(f"Preprocessor file is missing required components: {missing_keys}. Please retrain the model.")


    return model, preprocessor


def preprocess_for_prediction(data_point_or_df, preprocessor):
    """
    Applies preprocessing and feature engineering steps to raw input data
    using a fitted preprocessor.
    Assumes data_point_or_df is a DataFrame (or can be converted to one)
    with columns similar to the raw input CSV, INCLUDING 'Transaction_Date'.
    Handles datetime conversion, adds time/season features dynamically,
    applies fitted preprocessor (encoding, scaling), and aligns columns.
    Lag/Rolling features are NOT added dynamically here for real-time simplification.
    """
    df_temp = data_point_or_df.copy() # Work on a copy
    print(f"DEBUG: preprocess_for_prediction received data with columns: {df_temp.columns.tolist()}")
    print(f"DEBUG: preprocess_for_prediction received data shape: {df_temp.shape}")


    # Ensure data is DataFrame and has 'Transaction_Date'
    if not isinstance(df_temp, pd.DataFrame):
         try:
            df_temp = pd.DataFrame([df_temp]) # Convert single data point (e.g., dict) to DataFrame
         except Exception as e:
              print(f"DEBUG: Error converting input to DataFrame: {e}")
              raise ValueError(f"Input data cannot be converted to DataFrame: {e}")


    if 'Transaction_Date' not in df_temp.columns and not isinstance(df_temp.index, pd.DatetimeIndex):
         print("DEBUG: Input data missing 'Transaction_Date' or DatetimeIndex.")
         raise ValueError("Input data must contain 'Transaction_Date' column or have a DatetimeIndex for feature engineering.")

    # --- Initial Cleaning/Preparation ---
    # Load preprocessor details
    encoder = preprocessor.get('encoder')
    scaler = preprocessor.get('scaler')
    cat_cols_trained = preprocessor.get('categorical_columns', [])
    log_cols_trained = preprocessor.get('log_cols', [])
    minmax_cols_trained = preprocessor.get('minmax_cols', [])
    dropped_corr_cols_trained = preprocessor.get('dropped_corr_cols', [])

    # Get the expected final feature columns from model_loader
    train_feature_columns = get_feature_columns() # Use the getter from model_loader

    if train_feature_columns is None:
         print("DEBUG: Feature columns list not loaded by model_loader.")
         raise ValueError("Feature columns list is not available. Cannot align data. Ensure model_loader loaded feature_columns.pkl")

    print(f"DEBUG: Expected training feature columns count: {len(train_feature_columns)}")
    # print(f"DEBUG: Expected training feature columns: {train_feature_columns}") # Optional: print full list


    # Drop columns that were dropped during initial training preprocessing
    print(f"DEBUG: Dropping columns based on dropped_corr_cols_trained: {dropped_corr_cols_trained}")
    df_temp = df_temp.drop(columns=dropped_corr_cols_trained, errors='ignore')
    print(f"DEBUG: Data columns after dropping: {df_temp.columns.tolist()}")

    # Remove target if present (in case monitoring data includes it)
    if 'Revenue' in df_temp.columns:
         print("DEBUG: Dropping 'Revenue' column.")
         df_temp = df_temp.drop('Revenue', axis=1)
         print(f"DEBUG: Data columns after dropping Revenue: {df_temp.columns.tolist()}")


    # Ensure 'Transaction_Date' is datetime and set as index for FE
    if 'Transaction_Date' in df_temp.columns:
        try:
            print("DEBUG: Converting 'Transaction_Date' to datetime and setting index.")
            df_temp['Transaction_Date'] = pd.to_datetime(df_temp['Transaction_Date'])
            df_temp.set_index('Transaction_Date', inplace=True)
            df_temp = df_temp.sort_index() # Sort by date
            print("DEBUG: 'Transaction_Date' converted and set as index.")
        except Exception as e:
             print(f"DEBUG: Error converting 'Transaction_Date' or setting index: {e}")
             raise ValueError(f"Error converting 'Transaction_Date' or setting index: {e}")
    elif not isinstance(df_temp.index, pd.DatetimeIndex):
         # Should not happen if the first check passes, but for safety
         print("DEBUG: Failed to set DatetimeIndex.")
         raise ValueError("Failed to set DatetimeIndex required for feature engineering.")

    print(f"DEBUG: Data columns before dynamic FE: {df_temp.columns.tolist()}")

    # --- Dynamic Feature Engineering (Time, Season) ---
    # These features are generated based on the date index
    if isinstance(df_temp.index, pd.DatetimeIndex):
        print("DEBUG: Adding dynamic time/season features.")
        df_temp['month'] = df_temp.index.month
        df_temp['dayofweek'] = df_temp.index.dayofweek
        try: df_temp['weekofyear'] = df_temp.index.isocalendar().week.astype(int)
        except AttributeError:
             # Fallback for older pandas versions
             warnings.warn("Using deprecated .weekofyear. Consider updating pandas.")
             df_temp['weekofyear'] = df_temp.index.weekofyear

        df_temp['quarter'] = df_temp.index.quarter
        df_temp['is_weekend'] = df_temp['dayofweek'].isin([5, 6]).astype(int)

        # Example: Holiday flag (Egyptian holidays, can be extended)
        # Note: This list should ideally be consistent between training and prediction
        holidays = pd.to_datetime(['2024-04-10', '2024-06-16']) # Example holidays
        df_temp['is_holiday'] = df_temp.index.isin(holidays).astype(int)

        def get_season(month):
            if month in [12, 1, 2]: return 'winter'
            if month in [3, 4, 5]: return 'spring'
            if month in [6, 7, 8]: return 'summer'
            if month in [9, 10, 11]: return 'autumn'
        df_temp['season'] = df_temp['month'].apply(get_season)
        # Dynamic get_dummies for season (matches FE script)
        # Handle potential pandas future warning for dtype
        with warnings.catch_warnings():
             warnings.simplefilter(action='ignore', category=FutureWarning)
             df_temp = pd.get_dummies(df_temp, columns=['season'], drop_first=True)
        print("DEBUG: Dynamic time/season features added.")
    else:
         print("DEBUG: DatetimeIndex not found, skipping dynamic FE.")


    print(f"DEBUG: Data columns after dynamic FE: {df_temp.columns.tolist()}")
    print(f"DEBUG: Data shape after dynamic FE: {df_temp.shape}")


    # --- Apply fitted Preprocessor Components (Encoding, Scaling) ---
    # Note: Based on your data_preprocessing.py, One-Hot Encoding is done *before*
    # saving the preprocessor dictionary. So, the input data *loaded* from
    # PreparedSalesData.csv already has one-hot encoded columns.
    # However, the API input is raw data. So, we need to re-apply encoding here.

    # Apply fitted OneHotEncoder to original categorical columns ('Category', 'Region')
    # Check if the original categorical columns are still present in df_temp
    # These columns might have been dropped if preprocess_data already encoded them
    # and saved the result to PreparedSalesData.csv.
    # If preprocess_data encodes and saves the *encoded* data, the API input needs
    # to be handled carefully. Let's assume API input is *raw* and needs full preprocessing.
    # The preprocessor dictionary contains the fitted encoder.

    # Identify categorical columns that were encoded during training based on the preprocessor
    # These are the original column names like 'Category', 'Region'.
    cat_cols_to_encode_now = [col for col in cat_cols_trained if col in df_temp.columns]
    print(f"DEBUG: Categorical columns to encode now: {cat_cols_to_encode_now}")


    if encoder and cat_cols_to_encode_now:
        try:
            print("DEBUG: Applying One-Hot Encoding transformation.")
            # Transform and create one-hot DataFrame
            # handle_unknown='ignore' in the encoder helps with unseen categories
            one_hot_encoded = encoder.transform(df_temp[cat_cols_to_encode_now])
            # Get feature names from the fitted encoder to ensure correct columns
            # Use original names for consistent column names - encoder.get_feature_names_out() is preferred
            # Fallback if get_feature_names_out is not available (older sklearn)
            try:
                one_hot_col_names = encoder.get_feature_names_out(cat_cols_trained)
            except AttributeError:
                 warnings.warn("Using deprecated .get_feature_names. Consider updating scikit-learn.")
                 one_hot_col_names = encoder.get_feature_names(cat_cols_trained)


            one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_col_names, index=df_temp.index)
            print(f"DEBUG: One-hot encoded columns: {one_hot_df.columns.tolist()}")

            # Drop original categorical columns and concatenate the one-hot encoded ones
            df_temp = df_temp.drop(cat_cols_to_encode_now, axis=1)
            # Concatenate, ensuring index alignment
            df_temp = pd.concat([df_temp, one_hot_df], axis=1)
            print(f"DEBUG: Data columns after One-Hot Encoding: {df_temp.columns.tolist()}")


        except Exception as e: # Catch potential errors during transform (e.g., data type issues)
            print(f"DEBUG: Error during categorical encoding transform: {e}")
            # Depending on robustness needed, could log error, skip row, etc.
            raise e # Re-raise for debugging
    elif cat_cols_to_encode_now:
         print(f"DEBUG: Encoder not loaded, but categorical columns {cat_cols_to_encode_now} are present. Skipping encoding.")
    else:
         print("DEBUG: No categorical columns to encode or encoder not needed/loaded.")


    # Apply scaling (log1p then minmax)
    # Apply log1p first (ensure cols exist and are numeric)
    # Use log_cols_trained from the preprocessor dictionary
    log_cols_apply = [col for col in log_cols_trained if col in df_temp.columns and pd.api.types.is_numeric_dtype(df_temp[col])]
    print(f"DEBUG: Columns for log1p scaling: {log_cols_apply}")
    if log_cols_apply:
        print("DEBUG: Applying log1p transformation.")
        # Ensure values are non-negative before log1p
        for col in log_cols_apply:
             # Check if any value is negative in the column slice
             if (df_temp[col] < 0).any(): # Corrected from .any().any() for Series/DataFrame slice
                  warnings.warn(f"Negative values found in '{col}'. log1p may produce NaNs. Replacing negatives with 0.")
                  df_temp[col] = df_temp[col].apply(lambda x: max(0, x) if pd.notna(x) else x) # Ensure non-negative, handle NaN
        try:
            df_temp[log_cols_apply] = df_temp[log_cols_apply].apply(np.log1p)
            print("DEBUG: Log1p transformation applied.")
        except Exception as e:
             print(f"DEBUG: Error during log1p transformation: {e}")
             raise e
    else:
         print("DEBUG: No columns for log1p scaling.")

    # Apply fitted MinMaxScaler (ensure cols exist, are numeric, and scaler is fitted)
    minmax_cols_apply = [col for col in minmax_cols_trained if
                         col in df_temp.columns and pd.api.types.is_numeric_dtype(df_temp[col])]
    print(f"DEBUG: Columns for MinMaxScaler scaling: {minmax_cols_apply}")

    if scaler and minmax_cols_apply:
        print("DEBUG: Applying MinMaxScaler transformation.")
        try:
            # Keep the original order from training
            cols_in_fitting_order = [col for col in minmax_cols_trained if col in df_temp.columns]

            if cols_in_fitting_order:
                cols_to_remove_from_scaling = []  # To collect columns that can't be converted

                for col in cols_in_fitting_order:
                    if not pd.api.types.is_float_dtype(df_temp[col]):
                        try:
                            df_temp[col] = df_temp[col].astype(float)
                        except ValueError:
                            warnings.warn(
                                f"Could not convert column '{col}' to float for scaling. Skipping scaling for this column.")
                            cols_to_remove_from_scaling.append(col)

                # Filter out bad columns
                cols_in_fitting_order = [col for col in cols_in_fitting_order if col not in cols_to_remove_from_scaling]

                if cols_in_fitting_order:
                    temp_df_for_scaling = df_temp[cols_in_fitting_order].copy()
                    df_temp[cols_in_fitting_order] = scaler.transform(temp_df_for_scaling)
                    print("DEBUG: MinMaxScaler transformation applied.")
                else:
                    print("DEBUG: No numeric columns left to scale after checking types.")

        except Exception as e:
            print(f"DEBUG: Error during scaling transform: {e}")
            raise e
    elif minmax_cols_apply:
         print(f"DEBUG: Scaler not loaded, but columns {minmax_cols_apply} need scaling. Skipping scaling.")
    else:
         print("DEBUG: No columns for MinMaxScaler scaling.")


    print(f"DEBUG: Data columns after preprocessing steps: {df_temp.columns.tolist()}")
    print(f"DEBUG: Data shape after preprocessing steps: {df_temp.shape}")


    # --- Final Feature Alignment ---
    # Ensure the processed DataFrame has the exact columns as the training data, in the correct order.
    # Add missing columns with a default value (0 is common for one-hot encoded features,
    # or engineered features like lags not added here).
    # Drop any extra columns not in the training set.
    final_feature_cols = train_feature_columns # This comes from model_loader

    print(f"DEBUG: Starting final feature alignment.")
    print(f"DEBUG: Current data columns: {df_temp.columns.tolist()}")
    print(f"DEBUG: Expected final feature columns: {final_feature_cols}")


    # Add missing columns with default value 0
    missing_cols = set(final_feature_cols) - set(df_temp.columns)
    print(f"DEBUG: Missing columns: {missing_cols}")
    for c in missing_cols:
        df_temp[c] = 0 # Add missing columns, default to 0
        print(f"DEBUG: Added missing column: {c}")


    # Drop any extra columns
    extra_cols = set(df_temp.columns) - set(final_feature_cols)
    print(f"DEBUG: Extra columns to drop: {extra_cols}")
    if extra_cols:
        # warnings.warn(f"Dropping extra columns not in training set: {extra_cols}")
        df_temp = df_temp.drop(columns=list(extra_cols))
        print(f"DEBUG: Dropped extra columns.")

    print(f"DEBUG: Data columns after adding/dropping: {df_temp.columns.tolist()}")

    # Ensure column order matches training
    # Use .loc to ensure index is preserved and columns are reordered
    try:
        df_final_features = df_temp.loc[:, final_feature_cols]
        print("DEBUG: Final feature alignment successful.")
        print(f"DEBUG: Final processed features columns: {df_final_features.columns.tolist()}")
        print(f"DEBUG: Final processed features shape: {df_final_features.shape}")

    except KeyError as e:
         print(f"DEBUG: KeyError during final feature alignment: {e}")
         print(f"DEBUG: Columns in df_temp: {df_temp.columns.tolist()}")
         print(f"DEBUG: Expected final_feature_cols: {final_feature_cols}")
         raise RuntimeError(f"Column mismatch during final feature alignment: {e}. Ensure feature_columns.pkl matches processed data.")
    except Exception as e:
         print(f"DEBUG: Unexpected error during final feature alignment: {e}")
         raise e


    return df_final_features


def predict_revenue(model, preprocessed_features, preprocessor):
    """
    Makes revenue predictions using the loaded transactional model on processed features.
    Applies inverse transformations if needed.
    """
    if preprocessed_features.empty:
         print("DEBUG: No features provided for prediction.")
         # Return empty array with a shape that indicates no predictions were made
         return np.array([])


    print("DEBUG: Making prediction(s)...")
    try:
        y_pred_scaled = model.predict(preprocessed_features)
        print("DEBUG: Model prediction complete.")
    except Exception as e:
         print(f"DEBUG: Error during model prediction: {e}")
         # Log the full traceback for better debugging
         import traceback
         traceback.print_exc()
         raise e


    # Inverse transform the predictions if Revenue was log-transformed during training
    # Check if 'Revenue' was in the original log_cols_trained list in the preprocessor
    log_cols_trained = preprocessor.get('log_cols', [])
    # Note: The actual target column name in the *processed* data before training
    # might just be 'Revenue', but the preprocessor stores info about original columns.
    # We need to know if the *target* column was one of the columns that underwent log transform.
    # The simplest way is to check if 'Revenue' is in the log_cols_trained list from the preprocessor.
    # This requires `data_preprocessing.py` to correctly identify 'Revenue' as a log-transformed column.

    # Let's assume 'Revenue' was correctly identified and transformed if present in LOG_COLS_PREPROC_CONTEXT
    # and log_cols_trained correctly reflects this.
    # Also, check if the scaler is present, as inverse transform might be needed even if only log1p was applied
    # if the target was part of the columns fed to the scaler.
    # However, for simplicity and assuming only log1p was applied to the target 'Revenue'
    # before the model saw it, we only inverse log1p if 'Revenue' is in log_cols_trained.
    print(f"DEBUG: Checking if 'Revenue' was log-transformed (in log_cols_trained): {'Revenue' in log_cols_trained}")
    if 'Revenue' in log_cols_trained:
        print("DEBUG: Applying inverse log1p transformation to predictions.")
        # Ensure predictions are non-negative before expm1
        y_pred_scaled[y_pred_scaled < 0] = 0 # Cap predictions at 0 before inverse log
        y_pred = np.expm1(y_pred_scaled)
    else:
        print("DEBUG: No inverse log1p transformation needed.")
        y_pred = y_pred_scaled # No inverse transform needed

    # Ensure predictions are non-negative after inverse transform
    y_pred[y_pred < 0] = 0
    print("DEBUG: Predictions capped at 0.")


    print("DEBUG: Prediction(s) complete.")
    return y_pred


def predict_sarima(sarima_model, start_date, end_date):
    """
    Generates time-series forecasts using the loaded SARIMA model.
    Assumes the SARIMA model is trained on daily aggregated data.
    sarima_model: The loaded SARIMA model object (e.g., from statsmodels.tsa.statespace.sarimax.SARIMAXResults).
    start_date: The start date for forecasting (datetime object).
    end_date: The end date for forecasting (datetime object).
    """
    if sarima_model is None:
        print("DEBUG: SARIMA model not loaded. Cannot generate forecast.")
        return pd.Series() # Return empty Series

    try:
        print(f"DEBUG: Generating SARIMA forecast from {start_date.date()} to {end_date.date()}")
        # The SARIMA model's predict method typically takes start and end indices.
        # For forecasting, the start index is usually the date immediately after the training data ends.
        # The end index is the requested end date.
        # Need to ensure the SARIMA model object (from statsmodels) has a predict method
        # that can handle date indices directly or calculate steps based on the training data end.

        # Assuming the model object has a predict method that takes start and end dates (as pandas Timestamp or datetime)
        # Need to ensure start_date and end_date are in the correct format (datetime objects)
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
             print("DEBUG: Invalid date types provided for SARIMA forecast.")
             return pd.Series()

        # Generate the forecast
        # The predict method should return a pandas Series with dates as index
        # The dates in the index should be pandas Timestamps.
        forecast_series = sarima_model.predict(start=start_date, end=end_date)
        print("DEBUG: SARIMA forecast generated.")

        # Ensure forecast values are non-negative if revenue cannot be negative
        forecast_series[forecast_series < 0] = 0
        print("DEBUG: SARIMA forecast values capped at 0.")


        return forecast_series

    except Exception as e:
        print(f"DEBUG: Error during SARIMA forecast prediction: {e}")
        import traceback
        traceback.print_exc() # Print traceback for debugging
        return pd.Series() # Return empty Series on error


if __name__ == "__main__":
    # Example Usage:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Use the model trained with all features for this example if available,
    # acknowledging it might be less accurate on real-time input without lags.
    MODEL_PATH_RF = os.path.join(PROJECT_ROOT, 'models_initial', 'best_random_forest_revenue_model.pkl')
    PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models_initial', 'preprocessor.pkl')
    SAMPLE_DATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'synthetic_ecommerce_data.csv')
    SARIMA_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models_initial', 'sarima_revenue_model.pkl') # Path to SARIMA model

    # Need to run the training script first to generate model and preprocessor files
    if not os.path.exists(MODEL_PATH_RF) or not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(SARIMA_MODEL_PATH):
        print("Error: One or more model/preprocessor files not found.")
        print(f"Ensure {MODEL_PATH_RF}, {PREPROCESSOR_PATH}, and {SARIMA_MODEL_PATH} exist.")
        print("Please run the training script (src/train.py or dvc repro) first.")
    else:
        try:
            # Load transactional model and preprocessor (for transactional prediction example)
            model_rf, preprocessor = load_model_and_preprocessor(MODEL_PATH_RF, PREPROCESSOR_PATH)

            # Load SARIMA model (for time-series forecast example)
            print(f"\nLoading SARIMA model from {SARIMA_MODEL_PATH}...")
            sarima_model = joblib.load(SARIMA_MODEL_PATH)
            print("SARIMA model loaded.")


            # --- Example of Transactional Prediction ---
            # Create some sample raw data for prediction
            # Load a few rows from the original data to simulate new input
            try:
                 # Load original raw data structure
                 sample_raw_df = pd.read_csv(SAMPLE_DATA_CSV)
                 # Take a small slice, e.g., the first 5 rows
                 sample_raw_df = sample_raw_df.head().copy() # Use .copy() to avoid issues

                 # Ensure it has expected original columns by explicitly defining them
                 # based on the dataset description
                 required_orig_cols = [
                    'Transaction_ID', 'Customer_ID', 'Product_ID', 'Transaction_Date',
                    'Category', 'Region', 'Units_Sold', 'Discount_Applied',
                    'Revenue', 'Clicks', 'Impressions', 'Conversion_Rate',
                    'Ad_CTR', 'Ad_CPC', 'Ad_Spend'
                 ]
                 # Filter sample_raw_df to only include these columns if they exist
                 sample_raw_df = sample_raw_df[[col for col in required_orig_cols if col in sample_raw_df.columns]]


                 print("\n--- Transactional Prediction Example ---")
                 print("\nSample raw data for prediction:")
                 print(sample_raw_df)

                 # Preprocess the sample data
                 # This call handles all steps internally
                 sample_processed_features = preprocess_for_prediction(sample_raw_df, preprocessor)

                 print("\nSample processed features for prediction:")
                 print(sample_processed_features)
                 print("\nProcessed features shape:", sample_processed_features.shape)
                 print("\nProcessed features columns:", sample_processed_features.columns.tolist())
                 # print("\nExpected training features columns:", preprocessor.get('feature_columns', 'N/A'))


                 # Make predictions
                 predictions_transactional = predict_revenue(model_rf, sample_processed_features, preprocessor)

                 print("\nTransactional Predictions:")
                 print(predictions_transactional)
                 print("\nTransactional Predictions (rounded):")
                 print(np.round(predictions_transactional, 2))

            except FileNotFoundError:
                print(f"Error: Sample data file not found at {SAMPLE_DATA_CSV}")
            except Exception as e:
                 print(f"An error occurred during the transactional prediction example: {e}")
                 import traceback
                 traceback.print_exc()


            # --- Example of Time-Series Forecast ---
            print("\n--- Time-Series Forecast Example (SARIMA) ---")
            # Define a forecast date range (e.g., next 7 days starting from a specific date)
            # For a real SARIMA forecast, the start date should be the day * after * the last date in the training data.
            # For this example, let's pick a date range for demonstration.
            # You would need to determine the actual last training date from your data/training process.
            forecast_start_date_example = datetime(2025, 1, 1) # Example start date
            forecast_end_date_example = datetime(2025, 1, 7)   # Example end date

            print(f"\nGenerating SARIMA forecast from {forecast_start_date_example.date()} to {forecast_end_date_example.date()}")

            try:
                 forecast_series = predict_sarima(sarima_model, forecast_start_date_example, forecast_end_date_example)

                 print("\nSARIMA Forecast (Predicted Daily Revenue):")
                 print(forecast_series)

            except Exception as e:
                 print(f"An error occurred during the SARIMA forecast example: {e}")
                 import traceback
                 traceback.print_exc()


        except FileNotFoundError as e:
            print(f"Error during setup: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during the script execution: {e}")
            import traceback
            traceback.print_exc()
