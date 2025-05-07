import pandas as pd
import numpy as np
import joblib
import os
# Import necessary preprocessing and feature engineering functions/classes
# Note: We need to *apply* the fitted preprocessor, not refit it.
# We also need to apply the *same* feature engineering steps (time/season for simplified real-time).
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # Needed for type hinting/structure
import warnings # To manage warnings if needed

# Define columns for different scaling methods (should match data_preprocessing)
# These are the names *before* any transformation or dropping.
LOG_COLS_PREPROC = ['Ad_Spend', 'Revenue', 'Conversion_Rate']
MINMAX_COLS_PREPROC = ['Discount_Applied', 'Clicks', 'Impressions', 'Ad_CTR', 'Ad_CPC', 'Units_Sold'] # Included Units_Sold for scaling consistency


def load_model_and_preprocessor(model_path, preprocessor_path):
    """
    Loads the trained model and the fitted preprocessor objects.
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

    # Ensure data is DataFrame and has 'Transaction_Date'
    if not isinstance(df_temp, pd.DataFrame):
         try:
            df_temp = pd.DataFrame([df_temp]) # Convert single data point (e.g., dict) to DataFrame
         except Exception as e:
              raise ValueError(f"Input data cannot be converted to DataFrame: {e}")


    if 'Transaction_Date' not in df_temp.columns and not isinstance(df_temp.index, pd.DatetimeIndex):
         raise ValueError("Input data must contain 'Transaction_Date' column or have a DatetimeIndex for feature engineering.")

    # --- Initial Cleaning/Preparation ---
    # Load preprocessor details
    encoder = preprocessor.get('encoder')
    scaler = preprocessor.get('scaler')
    cat_cols_trained = preprocessor.get('categorical_columns', [])
    log_cols_trained = preprocessor.get('log_cols', [])
    minmax_cols_trained = preprocessor.get('minmax_cols', [])
    dropped_corr_cols_trained = preprocessor.get('dropped_corr_cols', [])
    train_feature_columns = preprocessor.get('feature_columns') # Expected after all FE and preprocessing

    if train_feature_columns is None:
         raise ValueError("Preprocessor does not contain 'feature_columns'. Cannot align data.")

    # Drop columns that were dropped during initial training preprocessing
    df_temp = df_temp.drop(columns=dropped_corr_cols_trained, errors='ignore')
    # Remove target if present (in case monitoring data includes it)
    if 'Revenue' in df_temp.columns:
         df_temp = df_temp.drop('Revenue', axis=1)

    # Ensure 'Transaction_Date' is datetime and set as index for FE
    if 'Transaction_Date' in df_temp.columns:
        try:
            df_temp['Transaction_Date'] = pd.to_datetime(df_temp['Transaction_Date'])
            df_temp.set_index('Transaction_Date', inplace=True)
            df_temp = df_temp.sort_index() # Sort by date
        except Exception as e:
             raise ValueError(f"Error converting 'Transaction_Date' or setting index: {e}")
    elif not isinstance(df_temp.index, pd.DatetimeIndex):
         # Should not happen if the first check passes, but for safety
         raise ValueError("Failed to set DatetimeIndex required for feature engineering.")


    # --- Dynamic Feature Engineering (Time, Season) ---
    # These features are generated based on the date index
    if isinstance(df_temp.index, pd.DatetimeIndex):
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

        # Skip Lag/Rolling features for real-time prediction simplification
        # If the model requires these, they must be provided in the input data,
        # or this function needs access to historical data.
        # Assuming for this simplified real-time path, the model used doesn't require them
        # OR the model is robust to them being zero/missing. The training script is set up
        # to save a model trained *with* lags/rolling, which might perform poorly without them.
        # A better approach is to train a dedicated real-time model without these features.
        # For this example, we proceed, knowing the prediction might be less accurate
        # if the loaded model relies heavily on lags/rolling and they are not in the input/added here.


    # --- Apply fitted Preprocessor Components ---

    # Apply fitted OneHotEncoder to original categorical columns ('Category', 'Region')
    # Check if the original categorical columns are still present in df_temp
    cat_cols_present_for_encoding = [col for col in cat_cols_trained if col in df_temp.columns]

    if encoder and cat_cols_present_for_encoding:
        try:
            # Transform and create one-hot DataFrame
            # handle_unknown='ignore' in the encoder helps with unseen categories
            one_hot_encoded = encoder.transform(df_temp[cat_cols_present_for_encoding])
            # Get feature names from the fitted encoder to ensure correct columns
            one_hot_col_names = encoder.get_feature_names_out(cat_cols_trained) # Use original names for consistent column names

            one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_col_names, index=df_temp.index)

            # Drop original categorical columns and concatenate the one-hot encoded ones
            df_temp = df_temp.drop(cat_cols_present_for_encoding, axis=1)
            # Concatenate, ensuring index alignment
            df_temp = pd.concat([df_temp, one_hot_df], axis=1)

        except Exception as e: # Catch potential errors during transform (e.g., data type issues)
            print(f"Error during categorical encoding transform: {e}")
            # Depending on robustness needed, could log error, skip row, etc.
            raise e # Re-raise for debugging


    # Apply scaling (log1p then minmax)
    # Apply log1p first (ensure cols exist and are numeric)
    # Use LOG_COLS_PREPROC which has original names for consistency with scaler's expected input
    log_cols_apply = [col for col in LOG_COLS_PREPROC if col in df_temp.columns and pd.api.types.is_numeric_dtype(df_temp[col])]
    if log_cols_apply:
        # Ensure values are non-negative before log1p
        for col in log_cols_apply:
             if (df_temp[col] < 0).any().any(): # .any() chain to handle DataFrame slice
                  warnings.warn(f"Negative values found in '{col}'. log1p may produce NaNs. Replacing negatives with 0.")
                  df_temp[col] = df_temp[col].apply(lambda x: max(0, x)) # Ensure non-negative
        try:
            df_temp[log_cols_apply] = df_temp[log_cols_apply].apply(np.log1p)
        except Exception as e:
             print(f"Error during log1p transformation: {e}")
             raise e


    # Apply fitted MinMaxScaler (ensure cols exist, are numeric, and scaler is fitted)
    # Use MINMAX_COLS_PREPROC which has original names for consistency with scaler's expected input
    minmax_cols_apply = [col for col in MINMAX_COLS_PREPROC if col in df_temp.columns and pd.api.types.is_numeric_dtype(df_temp[col])]
    if scaler and minmax_cols_apply:
        try:
            # Create a temporary DataFrame with only the columns to scale, in the *original order*
            # as they were fed to the scaler during fitting. This order *must* be consistent.
            # Assuming minmax_cols_trained list preserves the order from data_preprocessing.
            # Only select columns that are both in the list used for fitting AND currently in the DataFrame
            cols_in_fitting_order = [col for col in minmax_cols_trained if col in df_temp.columns]

            if cols_in_fitting_order:
                # Ensure data types are appropriate before scaling
                 cols_to_remove_from_scaling = [] # List to collect columns that fail conversion
                 for col in cols_in_fitting_order:
                      if not pd.api.types.is_float_dtype(df_temp[col]):
                           try:
                               df_temp[col] = df_temp[col].astype(float)
                           except ValueError:
                               warnings.warn(f"Could not convert column '{col}' to float for scaling. Skipping scaling for this column.")
                               cols_to_remove_from_scaling.append(col) # Add to removal list

                 # Remove columns that failed conversion from the list used for scaling
                 cols_in_fitting_order = [col for col in cols_in_fitting_order if col not in cols_to_remove_from_scaling]


                 if cols_in_fitting_order:
                    temp_df_for_scaling = df_temp[cols_in_fitting_order].copy()
                    # Apply scaling
                    df_temp[cols_in_fitting_order] = scaler.transform(temp_df_for_scaling)

        except Exception as e:
            print(f"Error during scaling transform: {e}")
            raise e


    # --- Final Feature Alignment ---
    # Ensure the processed DataFrame has the exact columns as the training data, in the correct order.
    # Add missing columns with a default value (0 is common for one-hot encoded features,
    # or engineered features like lags not added here).
    # Drop any extra columns not in the training set.
    final_feature_cols = train_feature_columns

    # Add missing columns with default value 0
    missing_cols = set(final_feature_cols) - set(df_temp.columns)
    for c in missing_cols:
        df_temp[c] = 0 # Add missing columns, default to 0

    # Drop any extra columns
    extra_cols = set(df_temp.columns) - set(final_feature_cols)
    if extra_cols:
        # warnings.warn(f"Dropping extra columns not in training set: {extra_cols}")
        df_temp = df_temp.drop(columns=list(extra_cols))

    # Ensure column order matches training
    df_final_features = df_temp[final_feature_cols]

    return df_final_features


def predict_revenue(model, preprocessed_features, preprocessor):
    """
    Makes revenue predictions using the loaded model and preprocessed features.
    Applies inverse transformations if needed.
    """
    if preprocessed_features.empty:
         print("No features provided for prediction.")
         # Return empty array with a shape that indicates no predictions were made
         return np.array([])


    print("Making prediction(s)...")
    try:
        y_pred_scaled = model.predict(preprocessed_features)
    except Exception as e:
         print(f"Error during model prediction: {e}")
         raise e


    # Inverse transform the predictions if Revenue was log-transformed during training
    # Check if 'Revenue' was in the original log_cols_trained list in the preprocessor
    log_cols_trained = preprocessor.get('log_cols', [])
    # Note: The actual target column name in the *processed* data before training
    # might just be 'Revenue', but the preprocessor stores info about original columns.
    # We need to know if the *target* column was one of the columns that underwent log transform.
    # The simplest way is to check if 'Revenue' is in the log_cols_trained list from the preprocessor.
    # This requires `data_preprocessing.py` to correctly identify 'Revenue' as a log-transformed column.

    # Let's assume 'Revenue' was correctly identified and transformed if present in LOG_COLS_PREPROC
    # and log_cols_trained correctly reflects this.
    if 'Revenue' in log_cols_trained:
        # Ensure predictions are non-negative before expm1
        y_pred_scaled[y_pred_scaled < 0] = 0 # Cap predictions at 0 before inverse log
        y_pred = np.expm1(y_pred_scaled)
    else:
        y_pred = y_pred_scaled # No inverse transform needed

    # Ensure predictions are non-negative after inverse transform
    y_pred[y_pred < 0] = 0


    print("Prediction(s) complete.")
    return y_pred


# --- Helper functions used internally by preprocess_for_prediction ---
# (Copied from feature_engineering.py logic, but simplified for prediction flow)
# These are NOT separate callable functions from outside predict_utils.

# (No need to define add_time_features_for_prediction or add_lag_rolling_features_for_prediction)
# The logic is directly within preprocess_for_prediction


if __name__ == "__main__":
    # Example Usage:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Use the model trained with all features for this example if available,
    # acknowledging it might be less accurate on real-time input without lags.
    MODEL_PATH_RF = os.path.join(PROJECT_ROOT, 'models_initial', 'best_random_forest_revenue_model.pkl')
    PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models_initial', 'preprocessor.pkl')
    SAMPLE_DATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'synthetic_ecommerce_data.csv')

    # Need to run the training script first to generate model and preprocessor files
    if not os.path.exists(MODEL_PATH_RF) or not os.path.exists(PREPROCESSOR_PATH):
        print("Error: Model or preprocessor not found.")
        print(f"Ensure {MODEL_PATH_RF} and {PREPROCESSOR_PATH} exist.")
        print("Please run the training script (src/train.py or dvc repro) first.")
    else:
        try:
            model, preprocessor = load_model_and_preprocessor(MODEL_PATH_RF, PREPROCESSOR_PATH)

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


                 print("\nSample raw data for prediction:")
                 print(sample_raw_df)

                 # Preprocess the sample data
                 # This call handles all steps internally
                 sample_processed_features = preprocess_for_prediction(sample_raw_df, preprocessor)

                 print("\nSample processed features for prediction:")
                 print(sample_processed_features)
                 print("\nProcessed features shape:", sample_processed_features.shape)
                 print("\nProcessed features columns:", sample_processed_features.columns.tolist())
                 print("\nExpected training features columns:", preprocessor.get('feature_columns', 'N/A'))


                 # Make predictions
                 predictions = predict_revenue(model, sample_processed_features, preprocessor)

                 print("\nPredictions:")
                 print(predictions)
                 print("\nPredictions (rounded):")
                 print(np.round(predictions, 2))

            except FileNotFoundError:
                print(f"Error: Sample data file not found at {SAMPLE_DATA_CSV}")
            except Exception as e:
                 print(f"An error occurred during the prediction process: {e}")
                 import traceback
                 traceback.print_exc()

        except FileNotFoundError as e:
            print(f"Error during setup: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during the script execution: {e}")
            import traceback
            traceback.print_exc()

