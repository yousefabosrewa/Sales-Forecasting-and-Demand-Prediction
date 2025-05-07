import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Define columns for different scaling methods
LOG_COLS = ['Ad_Spend', 'Revenue', 'Conversion_Rate']
MINMAX_COLS = ['Discount_Applied', 'Clicks', 'Impressions', 'Ad_CTR', 'Ad_CPC'] # Excluded Units_Sold as it's not in notebook scaling list

def preprocess_data(input_csv_path, output_csv_path, preprocessor_output_path):
    """
    Loads raw data, cleans, preprocesses, handles outliers, scales,
    and saves the processed data and fitted preprocessor objects.
    """
    print(f"Loading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    print("Initial data shape:", df.shape)
    print("Dropping ID columns...")
    df.drop(['Transaction_ID', 'Customer_ID', 'Product_ID'], axis=1, inplace=True)

    print("Converting 'Transaction_Date' to datetime...")
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])

    # --- Feature Engineering relevant to preprocessing ---
    # Handle highly correlated features (based on notebook)
    print("Handling highly correlated features...")
    # Recalculate correlation after dropping IDs
    numerical_cols = df.select_dtypes(include=np.number).columns
    correlation_matrix = df[numerical_cols].corr()
    to_drop = set()
    # Check correlation with Revenue explicitly, and pairwise high correlation
    # Based on notebook analysis, Impressions, Ad_CTR, Ad_CPC, Conversion_Rate
    # are highly correlated with Clicks/Ad_Spend/Units_Sold/Revenue.
    # The notebook dropped some based on >0.8 correlation. Let's simulate dropping
    # Impressions and Ad_CTR based on potential high correlation with Clicks/Ad_CPC/Ad_Spend
    # as done implicitly in the notebook's dropping logic.
    # Note: The notebook's dropping logic is dynamic based on current correlations.
    # For a stable pipeline, it's better to define which columns to drop.
    # Let's stick to the notebook's likely outcome of dropping Impressions and Ad_CTR
    # based on the correlation matrix seen.
    # Example based on notebook:
    # Dropped columns: {'Ad_CTR', 'Impressions'} # or similar depending on exact data version
    columns_to_potentially_drop = ['Impressions', 'Ad_CTR'] # Based on common high correlation
    df.drop(columns=columns_to_potentially_drop, axis=1, errors='ignore', inplace=True)
    print("Dropped columns:", columns_to_potentially_drop) # Report what was attempted

    print("Rounding float columns...")
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].round(2)

    # --- Categorical Encoding ---
    print("Applying One-Hot Encoding to Category and Region...")
    categorical_columns = ['Category', 'Region']
    # Fit encoder and transform
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])

    # Create DataFrame from encoded features
    feature_names = encoder.get_feature_names_out(categorical_columns)
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=feature_names, index=df.index)

    # Drop original categorical columns and concatenate
    df = df.drop(categorical_columns, axis=1)
    df = pd.concat([df, one_hot_df], axis=1)


    # --- Outliers Handling (IQR) ---
    print("Handling outliers (replacing with median)...")
    numeric_cols_after_encoding = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude the newly created one-hot encoded columns from outlier detection
    # and also exclude 'Units_Sold' and 'Revenue' if they are targets/not features for RF/XGB features
    # Let's apply to all numerical features that will be used or scaled, except the targets themselves
    cols_for_outlier_handling = [col for col in numeric_cols_after_encoding if col not in ['Revenue', 'Units_Sold']]

    for column in cols_for_outlier_handling:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        median_val = df[column].median()
        df[column] = np.where(
            (df[column] < lower_bound) | (df[column] > upper_bound),
            median_val,
            df[column]
        )

    # --- Scaling ---
    print("Applying scaling (Log1p and Min-Max)...")
    # Ensure columns exist before applying transformations
    log_cols_exist = [col for col in LOG_COLS if col in df.columns]
    minmax_cols_exist = [col for col in MINMAX_COLS if col in df.columns]

    # Apply log1p transformation
    if log_cols_exist:
        # Add 1 before log for values that might be zero
        df[log_cols_exist] = df[log_cols_exist].apply(np.log1p)

    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    if minmax_cols_exist:
        df[minmax_cols_exist] = scaler.fit_transform(df[minmax_cols_exist])

    # --- Save Preprocessor ---
    # Save the fitted encoder and scaler
    preprocessor = {
        'encoder': encoder,
        'scaler': scaler,
        'categorical_columns': categorical_columns,
        'log_cols': log_cols_exist,
        'minmax_cols': minmax_cols_exist,
        'dropped_corr_cols': columns_to_potentially_drop
    }
    os.makedirs(os.path.dirname(preprocessor_output_path), exist_ok=True)
    joblib.dump(preprocessor, preprocessor_output_path)
    print(f"Fitted preprocessor saved to {preprocessor_output_path}")

    # --- Save Processed Data ---
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False) # Save without date index for now
    print(f"Processed data saved to {output_csv_path}")
    print("Final processed data shape:", df.shape)

if __name__ == "__main__":
    # Example Usage:
    # Define paths relative to the project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'synthetic', 'synthetic_ecommerce_data.csv')
    OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'PreparedSalesData.csv')
    PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, 'models_initial', 'preprocessor.pkl')

    # Ensure the input file exists (for direct script execution)
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found at {INPUT_CSV}")
        print("Please place synthetic_ecommerce_data.csv in data/synthetic/")
    else:
        preprocess_data(INPUT_CSV, OUTPUT_CSV, PREPROCESSOR_PATH)