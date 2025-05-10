# Enterprise Technical Documentation: Sales Forecasting and Demand Prediction MLOps Project

## 1. Introduction and Business Context

**Project Goal:** To provide a robust, scalable, and maintainable MLOps solution for sales demand forecasting, enabling accurate predictions of sales revenue at both granular (transactional) and aggregated (daily time-series) levels.

**Business Value:**
*   **Optimized Inventory Management:** Reduce overstocking and stockouts by accurately predicting demand.
*   **Enhanced Financial Planning:** Improve budget accuracy with reliable sales forecasts.
*   **Strategic Marketing Campaigns:** Target promotions and campaigns more effectively based on predicted demand peaks.
*   **Improved Resource Allocation:** Optimize staffing and operational resources based on anticipated sales volumes.
*   **Data-Driven Decision Making:** Empower stakeholders with actionable insights derived from predictive analytics.

**Target Audience for this Document:**
*   **Software Engineers & DevOps:** For deployment, maintenance, and scaling of the system.
*   **Data Scientists & ML Engineers:** For understanding model development, evaluation, and retraining processes.
*   **Technical Architects:** For system design review and integration with other enterprise systems.
*   **New Contributors:** To quickly onboard and understand the codebase and MLOps practices.

## 2. System Architecture

The system is designed with a modular architecture, promoting separation of concerns and maintainability.

```
+-------------------------+      +-------------------------+      +-----------------------+
|     Data Sources        |----->|   MLOps Pipeline (DVC)  |<-----|   MLflow Tracking     |
| (e.g., CSV, Databases)  |      +-------------------------+      +-----------------------+
+-------------------------+      | - Data Ingestion        |
                                 | - Preprocessing         |
+-------------------------+      | - Feature Engineering   |      +-----------------------+
|   External Systems      |<-----| - Model Training        |----->|  Model & Preprocessor |
| (e.g., BI Tools, ERP)   |      | - Evaluation            |      |       Artifacts       |
+-------------------------+      | - Versioning (Data/Code)|      +-----------------------+
                                 +-----------|-------------+
                                             |
                                             v
+--------------------------------------------------------------------------------------+
|                                   Model Serving Layer                                  |
|--------------------------------------------------------------------------------------|
| +-------------------------+      +-------------------------------------------------+ |
| |      Flask API          |----->|  Streamlit Interactive Dashboard                  | |
| | (api.main:app)          |      |  (src/streamlit_app/app.py)                     | |
| | - /predict              |      |  - User Input Forms                             | |
| | - /predict_batch_csv    |      |  - Visualization (Plotly, Altair)               | |
| | - /forecast_sarima      |      |  - API Calls to Flask                           | |
| | - /model_info           |      |  - SHAP Explainability Display                  | |
| | - /shap_values          |      |  - Data Export                                  | |
| | - /status               |      +-------------------------------------------------+ |
| | - /reload_model         |                                                        | |
| +-------------------------+                                                        |
+--------------------------------------------------------------------------------------+
```

**Key Architectural Principles:**
*   **Modularity:** Core functionalities (data processing, training, API, UI) are separated into distinct modules.
*   **Reproducibility:** DVC and MLflow ensure that experiments and model training runs are reproducible.
*   **Scalability:** The Flask API is designed to be stateless, allowing for horizontal scaling. (Further considerations in Section 10).
*   **Maintainability:** Clear separation of concerns and use of established MLOps tools simplify maintenance and updates.
*   **Extensibility:** New models, features, or API endpoints can be added with minimal impact on existing components.

## 3. Technical Stack (Detailed)

### 3.1. Core Technologies
*   **Python (3.8+):** Primary programming language.
*   **Pandas (2.2.3):** Data manipulation and analysis.
*   **NumPy (1.26.4):** Numerical computations.
*   **Scikit-learn (1.4.1.post1):** Machine learning library (Random Forest, preprocessing, metrics).
*   **XGBoost (2.1.1):** Gradient boosting library for high-performance models.
*   **Statsmodels (0.14.1):** Statistical modeling, including SARIMA for time series.
*   **Joblib (1.3.2):** Efficiently saving and loading Python objects (models, preprocessors).

### 3.2. MLOps & Experimentation
*   **MLflow (2.12.1):**
    *   **Tracking:** Logging parameters, metrics, code versions, and artifacts.
    *   **Projects:** Packaging code for reproducible runs.
    *   **Models:** Managing model lifecycle (though Model Registry is a future enhancement).
    *   **UI:** Centralized dashboard for experiment review.
*   **DVC (Data Version Control) (3.48.0):**
    *   Versioning large data files, models, and intermediate artifacts outside Git.
    *   Defining and managing ML pipelines (`dvc.yaml`).
    *   Ensuring data and pipeline reproducibility.

### 3.3. API & Web Interface
*   **Flask (3.0.3):** Micro web framework for creating the REST API.
    *   **Flask-Pydantic (0.13.0):** Request and response validation using Pydantic.
*   **Pydantic (2.7.1):** Data validation and settings management.
*   **Streamlit (1.33.0):** Framework for building interactive web applications for ML.
*   **Waitress (2.1.2):** Production-ready WSGI server for Windows.
*   **Gunicorn (21.2.0):** Production-ready WSGI server for Linux/macOS.

### 3.4. Visualization & Explainability
*   **Plotly (5.18.0):** Interactive, publication-quality graphs.
*   **Altair (5.2.0):** Declarative statistical visualization library.
*   **Matplotlib (3.8.4) & Seaborn (0.13.2):** Static and statistical plotting.
*   **SHAP (Explainable AI) (0.43.0):** Model-agnostic explainability, calculating feature contributions.

### 3.5. Development & Code Quality
*   **Git:** Version control for code.
*   **Pytest (7.4.4):** Testing framework.
*   **Black (24.2.0):** Code formatter for consistent style.
*   **Flake8 (7.0.0):** Linter for identifying code errors and style issues.
*   **Mypy (1.8.0):** Static type checker.
*   **Python-dotenv (1.0.1):** Managing environment variables.

## 4. Detailed Project Structure & Components

Refer to Section 4 of the previous documentation for the directory structure.

### 4.1. `src/data_preprocessing.py`
*   **Purpose:** Handles initial data loading, cleaning, transformation, and saves processed data and preprocessor objects.
*   **Key Function:** `preprocess_data(raw_data_path, processed_data_path, preprocessor_output_path)`
    *   Loads raw data (e.g., `synthetic_ecommerce_data.csv`).
    *   Handles missing values (imputation or removal based on defined strategies).
    *   Converts data types (e.g., `Transaction_Date` to datetime).
    *   Performs one-hot encoding for categorical features (`Category`, `Region`).
    *   Applies transformations like log transform for skewed numerical features.
    *   Scales numerical features (e.g., `MinMaxScaler`).
    *   Saves the processed DataFrame to `processed_data_path` (e.g., `PreparedSalesData.csv`).
    *   Saves the fitted preprocessor components (encoder, scaler, column lists) as a dictionary using `joblib` to `preprocessor_output_path` (e.g., `models_initial/preprocessor.pkl`).
    *   **Outputs:** `PreparedSalesData.csv`, `preprocessor.pkl`.

### 4.2. `src/feature_engineering.py`
*   **Purpose:** Creates new predictive features from existing data.
*   **Key Functions:**
    *   `add_time_features(df)`:
        *   Input: DataFrame with a DatetimeIndex or `Transaction_Date` column.
        *   Extracts `month`, `dayofweek`, `weekofyear`, `quarter`, `is_weekend`.
        *   Adds holiday flags (configurable list of holidays).
        *   Creates 'season' feature and one-hot encodes it.
    *   `add_lag_rolling_features(df)`:
        *   Input: DataFrame sorted by time, with `Revenue` and `Units_Sold`.
        *   Creates lag features (e.g., `revenue_lag_1`, `revenue_lag_7`).
        *   Creates rolling window statistics (e.g., `revenue_rollmean_7`).
    *   `aggregate_daily_revenue(df)`:
        *   Input: DataFrame with DatetimeIndex and `Revenue`.
        *   Resamples data to daily frequency, summing `Revenue`.
        *   Fills missing days using forward fill (`ffill`).
        *   **Output:** Pandas Series with daily aggregated revenue.

### 4.3. `src/train.py`
*   **Purpose:** Orchestrates the model training, evaluation, and MLflow tracking pipeline.
*   **Key Function:** `train_models(...)`
    1.  **Initial Data Preprocessing:** Calls `preprocess_data`.
    2.  **SARIMA Data Preparation:**
        *   Loads *raw* data to avoid transformations affecting `Revenue` for aggregation.
        *   Calls `aggregate_daily_revenue` to get daily revenue series.
        *   Saves daily aggregated data (e.g., `data/processed/daily_aggregated_revenue.csv`).
    3.  **Transactional Model Data Preparation:**
        *   Loads the `PreparedSalesData.csv` from `preprocess_data`.
        *   Applies `add_time_features` and `add_lag_rolling_features`.
        *   Handles NaNs from lag/rolling features (typically by dropping).
        *   Performs time-based train/test split (e.g., last 6 months for test).
    4.  **Load Preprocessor Components:** Loads `preprocessor.pkl`.
    5.  **Manual Transformation Application:**
        *   Applies fitted log transformations and scalers from the loaded preprocessor to the train and test sets, ensuring column consistency. *Encoding is assumed to be part of the initial `preprocess_data` step and output columns are already encoded.*
    6.  **Model Training (Transactional - Random Forest, XGBoost):**
        *   Iterates through model configurations.
        *   Starts an MLflow run for each model: `mlflow.start_run()`.
        *   Logs parameters (`mlflow.log_params()`).
        *   Trains the model (e.g., `RandomForestRegressor().fit(X_train, y_train)`).
        *   Evaluates the model on `X_test`, `y_test` (calls `evaluate_model`).
        *   Logs metrics (`mlflow.log_metrics()`: MAE, MSE, RMSE, R²).
        *   Logs the trained model (`mlflow.sklearn.log_model()` or `mlflow.xgboost.log_model()`).
        *   Saves the "best" model locally using `joblib` based on a primary metric.
        *   Saves feature names used for training as a `.pkl` file.
    7.  **Model Training (Time Series - SARIMA):**
        *   Uses the daily aggregated revenue series.
        *   Performs train/test split on the time series.
        *   Starts an MLflow run.
        *   Logs SARIMA order (p,d,q)(P,D,Q,s) and other parameters.
        *   Trains `SARIMAX(train_series, order=..., seasonal_order=...).fit()`.
        *   Makes predictions on the test period.
        *   Evaluates (MAE, MSE, RMSE).
        *   Logs metrics and the SARIMA model (`mlflow.statsmodels.log_model()`).
        *   Saves the SARIMA model locally using `joblib`.
*   **Artifacts Produced:** Trained model files (`.pkl`), feature lists (`.pkl`), MLflow run data.

### 4.4. `src/evaluate.py`
*   **Purpose:** Calculates and returns performance metrics for trained models.
*   **Key Function:** `evaluate_model(model, X_test, y_test, model_type="transactional" or "timeseries")`
    *   Makes predictions using `model.predict(X_test)`.
    *   Calculates:
        *   Mean Absolute Error (MAE)
        *   Mean Squared Error (MSE)
        *   Root Mean Squared Error (RMSE)
        *   R-squared (R²)
    *   Returns a dictionary of these metrics.

### 4.5. `src/predict_utils.py`
*   **Purpose:** Contains utility functions for making predictions with trained models, used by the API.
*   **Key Functions:**
    *   `preprocess_for_prediction(input_df, preprocessor_components)`:
        *   Takes raw input DataFrame and loaded preprocessor components.
        *   Applies the *exact same* transformations (encoding, scaling, feature creation if necessary based on how preprocessor is structured) as used during training. Critical for consistency.
        *   Ensures feature order and names match the trained model's expectations.
    *   `predict_revenue(model, processed_features, preprocessor_components)`:
        *   Takes a trained transactional model and preprocessed features.
        *   Returns an array of revenue predictions.
    *   `predict_sarima(sarima_model, start_date, end_date)` (Conceptual, implementation might be in API directly or here):
        *   Takes a trained SARIMA model and forecast period.
        *   Returns a series of forecasted values.

### 4.6. `src/api/`
*   **`main.py`:** Defines Flask app and API endpoints. See Section 7 for detailed API specs.
*   **`model_loader.py`:**
    *   **Purpose:** Handles loading of trained models and preprocessor components.
    *   **Key Functions:**
        *   `load_latest_model(model_dir, model_name_pattern)`: Finds and loads the most recent version of a model.
        *   `get_model()`: Returns the loaded transactional model (e.g., RF or XGBoost).
        *   `get_preprocessor()`: Returns the loaded preprocessor components.
        *   `get_sarima_model()`: Returns the loaded SARIMA model.
        *   Implements caching or global variables to avoid reloading models on every request.
*   **`schemas.py`:** Defines Pydantic models for request/response validation. See Section 6.1.

### 4.7. `src/streamlit_app/app.py`
*   **Purpose:** Provides an interactive UI for data visualization, prediction, and model insights.
*   **Key Features:**
    *   **API Status Check:** Verifies connectivity to the Flask API.
    *   **Data Upload:** Allows users to upload CSV for visualization.
    *   **Historical Data Visualization:**
        *   Uses Plotly and Altair for trends, distributions, segmentation.
        *   Interactive filters (date range, category, region).
        *   KPI display.
    *   **Single Transaction Prediction:** Form to input features, calls `/predict` API endpoint.
    *   **Batch Prediction (CSV):** Upload CSV, calls `/predict_batch_csv` API endpoint, displays results.
    *   **SARIMA Forecasting:**
        *   Input for date range.
        *   Calls `/forecast_sarima` API endpoint.
        *   Visualizes forecast with confidence intervals (Plotly).
        *   Displays seasonality decomposition.
    *   **Model Explainability (SHAP):**
        *   Fetches feature importance from `/model_info`.
        *   Allows input for SHAP values for a specific prediction, calls `/shap_values`.
        *   Displays SHAP summary plots and force plots.
    *   **Data Export:** Option to download tables/forecasts as CSV/Excel.

### 4.8. `src/monitor.py` (Basic Monitoring)
*   **Purpose:** Script to perform basic model performance monitoring against new data or a baseline.
*   **Functionality:**
    *   Loads a trained model and preprocessor.
    *   Loads new data.
    *   Makes predictions on the new data.
    *   Compares performance metrics (e.g., MAE, R²) against baseline metrics (stored in a JSON file).
    *   Logs alerts if performance degrades significantly.
    *   Generates a performance report.
*   **Execution:** Run as a script, potentially scheduled.

## 5. Data Management

### 5.1. Data Sources
*   **Initial:** `data/synthetic/synthetic_ecommerce_data.csv` (example).
*   Designed to be adaptable to other structured data sources.

### 5.2. Data Schemas (Pydantic examples in `src/api/schemas.py`)

**Raw Input Data (Conceptual Example for `synthetic_ecommerce_data.csv`):**
*   `Transaction_ID`: String (e.g., "TXN123")
*   `Transaction_Date`: String (e.g., "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD")
*   `Product_ID`: String
*   `Product_Name`: String
*   `Category`: String (e.g., "Electronics", "Clothing")
*   `Units_Sold`: Integer
*   `Unit_Price`: Float
*   `Revenue`: Float (target variable for transactional models)
*   `Customer_ID`: String
*   `Region`: String (e.g., "North", "South")
*   Other relevant features...

**Processed Data (`PreparedSalesData.csv` - after `preprocess_data`):**
*   `Transaction_Date` (as index, datetime)
*   Numerical features scaled (e.g., `Units_Sold_scaled`, `Unit_Price_scaled`).
*   Categorical features one-hot encoded (e.g., `Category_Electronics`, `Region_North`).
*   Log-transformed features (e.g., `Revenue_log`).
*   Target variable `Revenue` (may or may not be scaled depending on modeling choice).

**Daily Aggregated Data (`daily_aggregated_revenue.csv`):**
*   `Transaction_Date` (as index, date, daily frequency)
*   `Revenue`: Float (sum of revenue for that day)

**API Schemas (see `src/api/schemas.py` for Pydantic definitions):**
*   `PredictionInput`: Features for a single transaction.
*   `PredictionOutput`: `predicted_revenue`.
*   `BatchPredictionInput`: `data: list[PredictionInput]`.
*   `BatchPredictionOutput`: `predictions: list[PredictionOutput]`.
*   `SarimaForecastInput`: `start_date: str`, `end_date: str`.
*   `SarimaForecastOutput`: `date: str`, `predicted_revenue: float`.
*   `BatchSarimaForecastOutput`: `forecasts: list[SarimaForecastOutput]`.
*   `SHAPInput`: Features for which to calculate SHAP values.
*   `SHAPOutput`: SHAP values structure.

### 5.3. Data Versioning (DVC)
*   **`dvc.yaml`:** Defines the DVC pipeline stages.
    *   **`preprocess` stage:**
        *   `cmd`: `python src/data_preprocessing.py ...`
        *   `deps`: `src/data_preprocessing.py`, `data/synthetic/raw_data.csv`
        *   `outs`: `data/processed/PreparedSalesData.csv`, `models_initial/preprocessor.pkl`
    *   **`feature_engineering_and_aggregate` stage:** (Conceptual, might be part of `train` or separate)
        *   `cmd`: `python src/feature_engineering.py ...` (or integrated into `train.py`)
        *   `deps`: `data/processed/PreparedSalesData.csv`, `src/feature_engineering.py`
        *   `outs`: `data/processed/daily_aggregated_revenue.csv`
    *   **`train` stage:**
        *   `cmd`: `python src/train.py ...`
        *   `deps`: `src/train.py`, `data/processed/PreparedSalesData.csv`, `data/processed/daily_aggregated_revenue.csv`, `models_initial/preprocessor.pkl`
        *   `outs`: `models_initial/best_random_forest_model.pkl`, `models_initial/best_xgboost_model.pkl`, `models_initial/sarima_model.pkl`, etc. (tracked by DVC if large, or by MLflow then pulled)
*   **`.dvc/config`:** Specifies remote storage for DVC-tracked files (e.g., S3, GCS, local).
*   **`.dvcignore`:** Files/directories DVC should ignore.
*   **Workflow:**
    1.  `dvc repro`: Executes the pipeline if dependencies change.
    2.  `dvc push`: Pushes tracked data/models to remote storage.
    3.  `dvc pull`: Fetches data/models from remote storage.
    4.  `git commit dvc.lock`: Commits the state of the pipeline outputs.

## 6. MLflow Integration

*   **Experiment Tracking:**
    *   Each model training run (RF, XGB, SARIMA) is an MLflow experiment.
    *   `mlflow.start_run(run_name="...", experiment_id="...")`
    *   **Parameters Logged:** Hyperparameters (e.g., `n_estimators`, `learning_rate`, SARIMA order), feature set details.
    *   **Metrics Logged:** MAE, MSE, RMSE, R².
    *   **Artifacts Logged:**
        *   Trained model files (via `mlflow.sklearn.log_model`, `mlflow.xgboost.log_model`, `mlflow.statsmodels.log_model`).
        *   Feature importance plots.
        *   `requirements.txt` or `conda.yaml` for environment reproducibility.
        *   Confusion matrices or residual plots if applicable.
*   **MLproject File:**
    *   Defines entry points for running the project (e.g., `train`, `preprocess`).
    *   Specifies environment (e.g., `conda.yaml`).
    *   Allows running project with `mlflow run . -e train_model --param-file params.yaml`.
*   **MLflow UI (localhost:5000):**
    *   View and compare runs.
    *   Examine parameters, metrics, and artifacts.
    *   Download model artifacts.

## 7. API Specification (`src/api/main.py`)

Base URL: `http://localhost:5001` (configurable)

### 7.1. `/status`
*   **Method:** `GET`
*   **Description:** Checks the health of the API and status of loaded models.
*   **Response (Success - 200 OK):**
    ```json
    {
      "status": "ok",
      "timestamp": "YYYY-MM-DDTHH:MM:SS.ffffff",
      "transactional_model_loaded": true,
      "preprocessor_loaded": true,
      "sarima_model_loaded": true,
      "transactional_model_type": "RandomForestRegressor" // or "XGBRegressor"
    }
    ```
*   **Response (Error - 503 Service Unavailable if models not loaded):**
    ```json
    {
      "status": "error",
      "details": "Transactional model not loaded." // or other specific error
    }
    ```

### 7.2. `/predict`
*   **Method:** `POST`
*   **Description:** Predicts revenue for a single transaction.
*   **Request Body (application/json):** Matches `PredictionInput` schema (defined in `src/api/schemas.py`). Example:
    ```json
    {
      "Units_Sold": 10,
      "Unit_Price": 5.99,
      "Category": "Electronics", // Example features
      "Region": "North",
      // ... other features required by the model
    }
    ```
*   **Response (Success - 200 OK):** Matches `PredictionOutput` schema.
    ```json
    {
      "predicted_revenue": 58.50
    }
    ```
*   **Response (Error - 400 Bad Request):** Invalid input data.
*   **Response (Error - 503 Service Unavailable):** Model not loaded.
*   **Response (Error - 500 Internal Server Error):** Unexpected error during prediction.

### 7.3. `/predict_batch_csv`
*   **Method:** `POST`
*   **Description:** Predicts revenue for multiple transactions provided in a CSV file.
*   **Request Body:** `multipart/form-data` with a file part named `file`. CSV columns must match training data features.
*   **Response (Success - 200 OK):**
    ```json
    {
      "predictions": [
        {"predicted_revenue": 58.50},
        {"predicted_revenue": 102.10},
        // ...
      ],
      "filename": "uploaded_data.csv",
      "num_records_processed": 100,
      "num_errors": 0 // if any records failed preprocessing
    }
    ```
    (Alternatively, may return a CSV file directly with predictions appended or a JSON list of `PredictionOutput`)
*   **Response (Error - 400 Bad Request):** No file, invalid file type, or CSV parsing error.
*   **Response (Error - 503 Service Unavailable):** Model not loaded.
*   **Response (Error - 500 Internal Server Error):** Unexpected error.

### 7.4. `/predict_batch_json`
*   **Method:** `POST`
*   **Description:** Predicts revenue for multiple transactions provided as a JSON list.
*   **Request Body (application/json):** Matches `BatchPredictionInput` schema.
    ```json
    {
      "data": [
        { "Units_Sold": 10, "Unit_Price": 5.99, "Category": "Electronics", ... },
        { "Units_Sold": 5, "Unit_Price": 20.99, "Category": "Books", ... }
      ]
    }
    ```
*   **Response (Success - 200 OK):** Matches `BatchPredictionOutput` schema.
    ```json
    {
      "predictions": [
        {"predicted_revenue": 58.50},
        {"predicted_revenue": 102.10}
      ]
    }
    ```
*   **Response (Error):** Similar to `/predict`.

### 7.5. `/forecast_sarima`
*   **Method:** `POST`
*   **Description:** Generates a SARIMA time-series forecast for daily revenue.
*   **Request Body (application/json):** Matches `SarimaForecastInput` schema.
    ```json
    {
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD"
    }
    ```
*   **Response (Success - 200 OK):** Matches `BatchSarimaForecastOutput` schema.
    ```json
    {
      "forecasts": [
        {"date": "YYYY-MM-DD", "predicted_revenue": 1234.56, "confidence_interval_lower": 1100.0, "confidence_interval_upper": 1350.0},
        // ...
      ]
    }
    ```
    (Confidence intervals added for enterprise-level)
*   **Response (Error):** Similar to `/predict`, including specific errors for invalid date ranges or SARIMA model issues.

### 7.6. `/model_info`
*   **Method:** `GET`
*   **Description:** Provides metadata about the currently loaded transactional model, including feature importance.
*   **Response (Success - 200 OK):**
    ```json
    {
      "model_type": "RandomForestRegressor", // or XGBRegressor
      "training_date": "YYYY-MM-DDTHH:MM:SS",
      "mlflow_run_id": "xxxxxxxxxxxx",
      "feature_importance": {
        "feature1_encoded": 0.25,
        "feature2_scaled": 0.15,
        // ...
      },
      "performance_metrics": { // Could be added from last eval
          "mae": 10.5, "rmse": 15.2, "r2": 0.85
      }
    }
    ```
*   **Response (Error - 503 Service Unavailable):** Model or metadata not available.

### 7.7. `/shap_values`
*   **Method:** `POST`
*   **Description:** Calculates and returns SHAP values for a given input instance.
*   **Request Body (application/json):** Similar to `PredictionInput`, containing the instance for explanation.
    ```json
    {
      // Instance features matching PredictionInput
      "Units_Sold": 10, "Unit_Price": 5.99, ...
    }
    ```
*   **Response (Success - 200 OK):**
    ```json
    {
      "base_value": 50.0, // Expected model output over the training data
      "shap_values": {
        "feature1_encoded": 5.5,
        "feature2_scaled": -2.0,
        // ...
      },
      "instance_features": { /* The input instance features */ },
      "prediction": 58.50
    }
    ```
*   **Response (Error):** Similar to `/predict`.

### 7.8. `/reload_model`
*   **Method:** `POST`
*   **Description:** Triggers a reload of the models (transactional, preprocessor, SARIMA) from disk. Useful for updating models without restarting the API.
*   **Response (Success - 200 OK):**
    ```json
    {
      "status": "success",
      "message": "Models reloaded successfully.",
      "reloaded_models": ["transactional", "preprocessor", "sarima"]
    }
    ```
*   **Response (Error - 500 Internal Server Error):** If reloading fails.

## 8. Deployment Strategy

### 8.1. Environment Configuration
*   **`.env` files:** Use `python-dotenv` to manage environment-specific configurations (API port, model paths, database URIs - if used).
    *   Example `.env`:
        ```
        FLASK_APP=src.api.main:app
        FLASK_ENV=development # or production
        API_PORT=5001
        MODEL_DIR=models_initial/
        PREPROCESSOR_PATH=models_initial/preprocessor.pkl
        # MLFLOW_TRACKING_URI=...
        ```
*   **Environment Variables:** Prioritize environment variables in production for security and flexibility.

### 8.2. Application Server
*   **Development:** `flask run --port 5001`
*   **Production (Windows):** `waitress-serve --listen=0.0.0.0:5001 src.api.main:app`
*   **Production (Linux/macOS):** `gunicorn -w 4 --bind 0.0.0.0:5001 src.api.main:app`
    *   `-w 4`: Number of worker processes (adjust based on server cores).

### 8.3. Containerization (Docker - Planned/Recommended)
*   **`Dockerfile` (Flask API):**
    ```dockerfile
    FROM python:3.8-slim

    WORKDIR /app

    COPY requiremnts.txt requiremnts.txt
    RUN pip install --no-cache-dir -r requiremnts.txt

    COPY . .

    # DVC pull step (if models/data are large and managed by DVC remote)
    # RUN dvc pull -R

    ENV FLASK_APP=src.api.main:app
    ENV API_PORT=5001
    # Add other ENV VARS

    # Expose API port
    EXPOSE 5001

    # Command to run the app using gunicorn or waitress
    CMD ["gunicorn", "-w", "4", "--bind", "0.0.0.0:5001", "src.api.main:app"]
    # For Waitress: CMD ["waitress-serve", "--listen=0.0.0.0:5001", "src.api.main:app"]
    ```
*   **`Dockerfile` (Streamlit App):**
    ```dockerfile
    FROM python:3.8-slim

    WORKDIR /app

    COPY requiremnts.txt requiremnts.txt # Or a specific requirements_streamlit.txt
    RUN pip install --no-cache-dir -r requiremnts.txt

    COPY src/streamlit_app /app/src/streamlit_app
    # Copy other necessary files like assets

    ENV API_URL=http://<flask_api_container_name_or_ip>:5001 # Important for Streamlit to reach API

    EXPOSE 8501

    CMD ["streamlit", "run", "src/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    ```
*   **`docker-compose.yml` (Recommended for local multi-container setup):**
    ```yaml
    version: '3.8'
    services:
      mlflow:
        image: ghcr.io/mlflow/mlflow:latest # Or a specific version
        ports:
          - "5000:5000"
        # volumes:
          # - ./mlruns:/mlflow/mlruns # To persist mlruns locally
        command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /mlflow/mlruns # Example with local backend

      flask_api:
        build:
          context: .
          dockerfile: Dockerfile.api # Assuming you name it Dockerfile.api
        ports:
          - "5001:5001"
        environment:
          - API_URL=http://flask_api:5001 # If Streamlit needs to call back for some reason
          - MODEL_DIR=/app/models_initial # Ensure models are accessible in container
          - PREPROCESSOR_PATH=/app/models_initial/preprocessor.pkl
        # volumes:
          # - ./models_initial:/app/models_initial # Mount models if not baked into image
          # - ./data:/app/data # Mount data if needed by API
        depends_on:
          - mlflow # Optional, if API needs to interact with MLflow server

      streamlit_app:
        build:
          context: .
          dockerfile: Dockerfile.streamlit # Assuming you name it Dockerfile.streamlit
        ports:
          - "8501:8501"
        environment:
          - API_URL=http://flask_api:5001 # Crucial: Streamlit calls Flask API by its service name
        depends_on:
          - flask_api
    ```

### 8.4. CI/CD Pipeline (Conceptual - Future Enhancement)
1.  **Trigger:** `git push` to `main` or `develop` branch, or on Pull Request.
2.  **Lint & Test:**
    *   Run `flake8` and `black --check`.
    *   Run `mypy`.
    *   Run `pytest` (unit and integration tests).
3.  **DVC Setup (if applicable):** `dvc pull` to get necessary data/model dependencies for integration tests or building.
4.  **Build Docker Images:** Build API and Streamlit images. Push to a container registry (e.g., Docker Hub, ECR, GCR).
5.  **(Optional) Model Retraining & DVC Repro:** If triggered by data changes or schedule:
    *   `dvc repro` to run the DVC pipeline.
    *   `dvc push` to save new artifacts.
    *   MLflow will log new model versions.
6.  **Deploy:**
    *   Deploy Docker images to staging/production environment (e.g., Kubernetes, ECS, App Service).
    *   Update API with newly registered model (if using MLflow Model Registry or a custom mechanism).
7.  **Smoke Tests:** Basic tests against the deployed application.

## 9. Security Considerations

*   **API Security:**
    *   **Input Validation:** Pydantic schemas (`src/api/schemas.py`) provide robust input validation against common injection attacks (though not a full WAF replacement).
    *   **HTTPS:** In production, deploy behind a reverse proxy (e.g., Nginx, Traefik) that handles SSL/TLS termination.
    *   **Authentication/Authorization (Future):** For sensitive endpoints, implement API key authentication or OAuth2.
    *   **Rate Limiting:** Consider implementing rate limiting to prevent abuse.
    *   **Error Handling:** Avoid leaking sensitive stack trace information in production error messages.
*   **Data Security:**
    *   **Sensitive Data:** Although current data is synthetic, if real PII is used, ensure appropriate masking, anonymization, or access controls are in place.
    *   **Secrets Management:** Store API keys, database credentials, etc., securely using tools like HashiCorp Vault, AWS Secrets Manager, or environment variables managed by the deployment platform (not hardcoded).
*   **Dependency Management:**
    *   Regularly scan dependencies for vulnerabilities (e.g., `safety check -r requiremnts.txt`, Snyk, GitHub Dependabot).
    *   Pin dependency versions in `requiremnts.txt` to avoid unexpected updates.
*   **Container Security:**
    *   Use minimal base Docker images (e.g., `python:3.8-slim`).
    *   Run containers as non-root users.
    *   Regularly scan Docker images for vulnerabilities.

## 10. Scalability and Performance

*   **Flask API:**
    *   **Stateless Design:** The API is largely stateless (model loading is cached but doesn't store per-request state), which is good for horizontal scaling.
    *   **WSGI Server:** Use a production-grade WSGI server like `gunicorn` (with multiple workers) or `waitress`.
    *   **Load Balancing:** Deploy multiple instances of the API container behind a load balancer (e.g., Nginx, ELB, GCLB).
    *   **Asynchronous Operations (Future):** For long-running batch predictions, consider using task queues (Celery, Redis Queue).
*   **Model Inference:**
    *   **Optimized Models:** Ensure models are optimized for inference speed (e.g., pruning, quantization if applicable, though less common for RF/XGB/SARIMA).
    *   **Hardware:** CPU-bound for current models. Sufficient CPU resources for API instances.
*   **Streamlit Application:**
    *   Streamlit itself is stateful per user session. Scaling involves running multiple Streamlit instances, possibly with sticky sessions if session state is critical and not managed externally.
    *   Most heavy lifting (predictions) is offloaded to the Flask API.
*   **Data Processing (DVC Pipeline):**
    *   Can be resource-intensive. Run on machines with sufficient CPU/RAM.
    *   DVC stages can be parallelized if dependencies allow (though `dvc repro` runs sequentially by default).
*   **MLflow:**
    *   The MLflow tracking server can become a bottleneck with many concurrent runs. For high-load scenarios, consider a robust backend store (PostgreSQL, MySQL) and artifact store (S3, GCS).

## 11. Testing Strategy

*   **Unit Tests (`pytest`):**
    *   Test individual functions in `data_preprocessing.py`, `feature_engineering.py`, `evaluate.py`, `predict_utils.py`, `api/model_loader.py`.
    *   Mock external dependencies (e.g., file I/O, API calls if testing Streamlit components locally).
    *   Example: Test `add_time_features` with known input and assert output columns/values.
*   **Integration Tests (`pytest`):**
    *   Test interactions between components.
    *   Example: Test the flow from `train.py` calling `preprocess_data` and then a model training function, checking for expected artifacts.
    *   Test API endpoint logic by making requests to a test instance of the Flask app (using `Flask.test_client()`).
*   **API Tests (e.g., `pytest` with `requests` or Postman/Newman):**
    *   Test deployed API endpoints with various valid and invalid inputs.
    *   Verify response codes, headers, and body content.
*   **Streamlit Application Testing (More challenging, can involve tools like Selenium or Playwright for E2E):**
    *   Basic checks: Ensure pages load, API calls are made correctly.
    *   Consider tools like `streamlit-testing-library` for component-level tests if available/mature.
*   **Code Coverage:** Aim for high code coverage, tracked using `pytest-cov`.

## 12. Monitoring and Logging

*   **Application Logging:**
    *   **Flask API:** Uses Python's `logging` module. Configure handlers for console output and file-based logging (rotating logs in production). Log request details, errors, and key events.
        ```python
        # src/api/main.py
        import logging
        app.logger.setLevel(logging.INFO) # or DEBUG
        handler = logging.StreamHandler() # or FileHandler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        app.logger.addHandler(handler)
        app.logger.info("API started")
        ```
    *   **Streamlit:** Uses `st.write`, `st.info`, `st.warning`, `st.error` for UI feedback. Server-side logs are standard Python logs.
*   **Model Performance Monitoring (`src/monitor.py` and Future Enhancements):**
    *   Scheduled execution of `src/monitor.py` to track metrics like MAE, R² on new data against a baseline.
    *   Log alerts to a file, console, or integrate with alerting systems (e.g., PagerDuty, Slack via webhooks).
    *   **Future:** Implement data drift detection (e.g., Population Stability Index, Kolmogorov-Smirnov test) and concept drift detection.
*   **Infrastructure Monitoring (Production):**
    *   Use tools like Prometheus/Grafana, Datadog, or cloud provider monitoring services (CloudWatch, Azure Monitor) to track:
        *   CPU/Memory/Disk utilization of API and Streamlit instances.
        *   API request latency and error rates (5xx, 4xx).
        *   Container health and restarts.

## 13. Contribution Guidelines (Example)

*   **Coding Standards:**
    *   Follow PEP 8.
    *   Use `black` for code formatting (run `black .` before committing).
    *   Use `flake8` for linting (run `flake8 .` and resolve issues).
    *   Add type hints and run `mypy .` to check.
*   **Branching Strategy:**
    *   `main`: Production-ready code.
    *   `develop`: Integration branch for upcoming release.
    *   Feature branches: `feature/<feature-name>` (branched from `develop`).
    *   Bugfix branches: `bugfix/<issue-id>` (branched from `develop` or `main` for hotfixes).
*   **Commit Messages:** Follow conventional commit guidelines (e.g., `feat: add new prediction endpoint`).
*   **Pull Requests (PRs):**
    *   Submit PRs to `develop` branch.
    *   Ensure all tests pass in CI.
    *   Require at least one code review before merging.
    *   Update documentation if changes affect public interfaces or core logic.
*   **DVC Usage:**
    *   If adding/modifying DVC stages, ensure `dvc.yaml` and `dvc.lock` are updated and committed.
    *   Run `dvc repro` to test pipeline changes locally.

## 14. Glossary of Terms

*   **API (Application Programming Interface):** A way for different software components to communicate.
*   **DVC (Data Version Control):** Tool for versioning data and ML pipelines.
*   **MLflow:** Platform for managing the ML lifecycle.
*   **Flask:** Micro web framework for Python.
*   **Streamlit:** Framework for building interactive ML web apps.
*   **SARIMA (Seasonal Autoregressive Integrated Moving Average):** Time series forecasting model.
*   **SHAP (SHapley Additive exPlanations):** Method for explaining model predictions.
*   **WSGI (Web Server Gateway Interface):** Standard interface between web servers and Python web applications.
*   **CI/CD (Continuous Integration/Continuous Delivery or Deployment):** Practices for automating software building, testing, and deployment.
*   **Docker:** Platform for developing, shipping, and running applications in containers.
*   **KPI (Key Performance Indicator):** A measurable value that demonstrates how effectively a company is achieving key business objectives.

