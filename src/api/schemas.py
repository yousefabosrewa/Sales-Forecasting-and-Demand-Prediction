from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd # Needed for pandas compatibility note

# Define the schema for a single prediction input
# This should match the features the deployed model (base+time) expects
# derived from the *original* input columns plus engineered time/season features.
# It does NOT include lag/rolling features for the simplified real-time model.
class PredictionInput(BaseModel):
    # Original features (ensure these match the raw data column names)
    Transaction_Date: str # Expect date as string (e.g., "YYYY-MM-DD")
    Category: str
    Region: str
    Units_Sold: float # Or int, depending on data
    Discount_Applied: float
    Clicks: float # Or int
    Impressions: float # Included in raw data, but dropped in initial preprocessing.
                       # If real-time input includes it, preprocessor should handle dropping/ignoring.
                       # Let's define schema based on what the model *ultimately* uses after FE.
                       # The simplified real-time model uses base + time features.
                       # Base features from notebook: Units_Sold, Discount_Applied, Clicks, Impressions, Conversion_Rate, Ad_CTR, Ad_CPC, Ad_Spend
                       # Dropped Impressions and Ad_CTR in preprocessing.
                       # So, base features *after* preprocessing are: Units_Sold, Discount_Applied, Clicks, Conversion_Rate, Ad_CPC, Ad_Spend
                       # Time features: month, dayofweek, weekofyear, quarter, is_weekend, is_holiday
                       # Season features: season_autumn, season_spring, season_summer
                       # Encoded Categories/Regions: Category_Books, ..., Region_North America

    # Let's define the schema based on the *original* columns that are needed to
    # derive the features for the base+time model.
    # The `preprocess_for_prediction` function in predict_utils handles the
    # feature engineering and application of the preprocessor.

    # So, the API should accept the raw-like input structure:
    Transaction_Date: str # YYYY-MM-DD
    Category: str
    Region: str
    Units_Sold: float
    Discount_Applied: float
    Clicks: float
    Impressions: float # Even if dropped, accept it and let preprocessor handle
    Conversion_Rate: float
    Ad_CTR: float # Even if dropped, accept it
    Ad_CPC: float
    Ad_Spend: float
    # No Customer_ID, Product_ID, Transaction_ID as they were dropped initially.

    # Pydantic allows extra fields by default, but it's better to be explicit
    # or configure the model to allow or deny extra fields.
    # Example: Config = dict(extra='allow') or extra='ignore'

    # Note on Data Types: Pydantic helps validate basic types. Further validation
    # (e.g., ranges, categorical values) would be in the application logic.

# Define the schema for the prediction output
class PredictionOutput(BaseModel):
    predicted_revenue: float
    # Optional: Add other info like prediction date, confidence interval etc.


# Schema for batch prediction input (list of single inputs)
class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]

# Schema for batch prediction output (list of single outputs)
class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    # Optional: batch_id, processing_time etc.

# Note: For CSV file upload for batch, Pydantic schemas might not directly
# map to the file content. You would handle the file reading and row parsing
# within the Flask endpoint, then validate each row or the resulting DataFrame.
# The BatchPredictionInput/Output schemas are more suitable for JSON batch requests.