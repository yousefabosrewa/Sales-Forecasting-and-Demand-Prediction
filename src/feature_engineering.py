import pandas as pd
import numpy as np
import os

def add_time_features(df):
    """
    Adds time-based features to the DataFrame.
    Assumes 'Transaction_Date' is a datetime column and sets it as index.
    """
    print("Adding time-based features...")
    # Ensure 'Transaction_Date' is datetime and set as index
    if 'Transaction_Date' not in df.columns:
         if not isinstance(df.index, pd.DatetimeIndex):
              raise ValueError("DataFrame must have 'Transaction_Date' column or a DatetimeIndex.")
         # If date is already index, ensure it's datetime
         df.index = pd.to_datetime(df.index)
    else:
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
        df.set_index('Transaction_Date', inplace=True)


    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek
    # Handle potential error with isocalendar().week on some pandas versions/dates
    try:
        df['weekofyear'] = df.index.isocalendar().week.astype(int)
    except AttributeError: # Fallback for older pandas versions
        df['weekofyear'] = df.index.weekofyear

    df['quarter'] = df.index.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Example: Holiday flag (Egyptian holidays, can be extended)
    holidays = pd.to_datetime([
        '2024-04-10',  # Eid al-Fitr (example)
        '2024-06-16',  # Eid al-Adha (example)
        # Add more relevant holidays
    ])
    df['is_holiday'] = df.index.isin(holidays).astype(int)

    # Example: Season
    def get_season(month):
        if month in [12, 1, 2]: return 'winter'
        if month in [3, 4, 5]: return 'spring'
        if month in [6, 7, 8]: return 'summer'
        if month in [9, 10, 11]: return 'autumn'
    df['season'] = df['month'].apply(get_season)
    df = pd.get_dummies(df, columns=['season'], drop_first=True)

    print("Time-based features added.")
    return df

def add_lag_rolling_features(df):
    """
    Adds lag and rolling window features to the DataFrame.
    Assumes DataFrame index is a DatetimeIndex and is sorted.
    """
    print("Adding lag and rolling window features...")
    # Ensure index is sorted for correct lag/rolling calculations
    df = df.sort_index()

    # Lag features
    for lag in [1, 7, 14, 30]:
        df[f'revenue_lag_{lag}'] = df['Revenue'].shift(lag)
        df[f'units_lag_{lag}'] = df['Units_Sold'].shift(lag)

    # Rolling window features
    for window in [7, 14, 30]:
        df[f'revenue_rollmean_{window}'] = df['Revenue'].rolling(window).mean()
        df[f'units_rollmean_{window}'] = df['Units_Sold'].rolling(window).mean()

    # Handle NAs created by lag/rolling features
    # print("Handling NAs created by lag/rolling features...")
    # df = df.dropna() # Dropping is done in train.py or calling script based on split

    print("Lag and rolling window features added.")
    return df

def aggregate_daily_revenue(df):
    """
    Aggregates data to a daily frequency, primarily for SARIMA modeling.
    Focuses on summing Revenue.
    Assumes DataFrame index is a DatetimeIndex.
    """
    print("Aggregating data to daily frequency for SARIMA...")
    # Ensure index is sorted for correct resampling
    df = df.sort_index()

    # Aggregate Revenue by summing
    daily_revenue = df['Revenue'].resample('D').sum()

    # Fill missing days using forward fill as in prompt's SARIMA fix
    # This is crucial for time series models expecting regular frequency
    daily_revenue = daily_revenue.asfreq('D', method='ffill')

    print("Daily aggregation complete.")
    return daily_revenue

if __name__ == "__main__":
    # Example Usage:
    # Define paths relative to the project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DATA_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'PreparedSalesData.csv')
    DAILY_AGGREGATED_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'daily_aggregated_revenue.csv') # New output for daily data

    # Ensure the processed data file exists (for direct script execution)
    if not os.path.exists(PROCESSED_DATA_CSV):
        print(f"Error: Processed data file not found at {PROCESSED_DATA_CSV}")
        print("Please run data_preprocessing.py first.")
    else:
        df_processed = pd.read_csv(PROCESSED_DATA_CSV)

        # Add time features
        df_features = add_time_features(df_processed)

        # Add lag and rolling features (dropna will be handled in train/calling script)
        df_features = add_lag_rolling_features(df_features)

        print("\nDataFrame with all features (before handling NAs from lags/rolling):")
        print(df_features.head())
        print("\nInfo after adding features:")
        df_features.info()


        # Aggregate data for SARIMA
        daily_revenue_series = aggregate_daily_revenue(df_processed.copy()) # Use a copy to not affect the main df

        # Save daily aggregated data
        os.makedirs(os.path.dirname(DAILY_AGGREGATED_CSV), exist_ok=True)
        daily_revenue_series.to_csv(DAILY_AGGREGATED_CSV, header=['Revenue'])
        print(f"\nDaily aggregated revenue data saved to {DAILY_AGGREGATED_CSV}")
        print("\nDaily aggregated revenue head:")
        print(daily_revenue_series.head())