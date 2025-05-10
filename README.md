# **Sales Demand Forecasting MLOps Project**

## **📌 Project Overview**

This project implements a robust **Sales Demand Forecasting** solution leveraging **MLOps principles** to ensure **reproducibility**, **scalability**, and **efficient deployment**.

> The primary objective is to accurately predict future sales revenue at both transactional and aggregated daily levels, providing valuable insights for:

* **Business planning**
* **Inventory management**
* **Marketing strategies**

### **Key Technologies**

* Modular architecture
* **MLflow** for experiment tracking
* **DVC** for data & pipeline versioning
* **Flask API** for model serving
* **Streamlit** web interface for interaction

---

## **✨ Features**

### **🧹 Data Preprocessing**

* Raw data loading
* Cleaning & outlier treatment
* Initial transformations (e.g., **one-hot encoding**, **scaling**)

### **🧠 Feature Engineering**

* Time-based features: `day_of_week`, `month`, `quarter`, `holidays`
* Optional: **Lag** and **rolling window** features

### **🤖 Multiple Model Training**

* **Transactional Models**: `Random Forest`, `XGBoost`
* **Time-Series Model**: `SARIMA` for daily revenue trends

### **📈 MLflow Experiment Tracking**

* Logs **parameters**, **metrics** (`MAE`, `RMSE`, `R²`)
* Stores **model artifacts**
* Enables easy comparison between experiments

### **🧮 DVC Versioning**

* Tracks versions of **data**, **models**, and **pipeline**
* Ensures reproducibility and team collaboration

### **🌐 Flask REST API**

* Endpoints:

  * `POST /predict`: Single transaction prediction
  * `POST /predict_batch_csv`: Batch prediction via CSV
  * `POST /forecast_sarima`: Time-series forecast
  * `GET /status`: API status check

### **📊 Streamlit Web Application**

* Interactive dashboard for:

  * Data visualization
  * Running predictions
  * Viewing forecasts

### **🧪 Model Monitoring (Basic)**

* Script to compare model performance with baseline
* Can be expanded for **drift detection**

### **📦 Containerization (Planned)**

* Designed for **Docker-based** deployment environments

---

## **📁 Project Structure**

```
├── data/
│   ├── processed/             # Processed data, daily aggregated, monitoring reports
│   └── synthetic/             # Raw synthetic data
├── models_initial/            # Trained models and preprocessor components
├── src/
│   ├── api/                   # Flask API code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── evaluate.py
│   ├── monitor.py
│   ├── predict_utils.py
│   ├── streamlit_app/         # Streamlit app
│   └── train.py               # Training pipeline
├── .dvcignore
├── .gitignore
├── dvc.yaml                   # DVC pipeline
├── MLproject                  # MLflow config
├── requirements.txt
├── conda.yaml
└── README.md
```

---

## **⚙️ Setup Instructions**

### **🔧 Prerequisites**

* Python `3.8+`
* Git
* Conda or `venv`
* Docker *(optional)*

### **📥 1. Clone the Repository**

```bash
git clone <repository_url>
cd sales-demand-forecasting-mlops
```

### **🐍 2. Set up Environment**

#### Using Conda:

```bash
conda env create -f conda.yaml
conda activate sales-forecasting-mlops
```

#### Using venv:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### **📦 3. Install DVC**

```bash
pip install dvc[all]
```

### **▶️ 4. Initialize and Run DVC Pipeline**

```bash
dvc init --no-scm
dvc repro
```

---

## **📊 MLflow Tracking**

### **Start MLflow UI:**

```bash
mlflow ui
```

> Accessible at: `http://localhost:5000`

---

## **🧪 Usage**

### **Run Flask API**

```bash
waitress-serve --listen=0.0.0.0:5001 src.api.main:app
```

> Available at: `http://localhost:5001`

---

### **Run Streamlit App**

```bash
streamlit run src/streamlit_app/app.py
```

> Opens in browser: `http://localhost:8501`

---

### **API Endpoints Summary**

| Endpoint             | Method | Description                        |
| -------------------- | ------ | ---------------------------------- |
| `/predict`           | POST   | Single transaction prediction      |
| `/predict_batch_csv` | POST   | Upload CSV for batch predictions   |
| `/forecast_sarima`   | POST   | Forecast revenue for a date range  |
| `/status`            | GET    | Check API and model loading status |

---

### **Run Monitoring Script**

```bash
python src/monitor.py \
  --new_data_path data/processed/new_data_sample.csv \
  --model_path models_initial/best_random_forest_revenue_model.pkl \
  --preprocessor_path models_initial/preprocessor.pkl \
  --baseline_metrics_path data/processed/baseline_metrics.json \
  --report_output_dir data/processed/performance_reports \
  --alert_log_path data/processed/monitoring_alerts.log
```

---

## **⚙️ MLOps Components**

### **🧭 MLflow**

* Logs **hyperparameters**, **metrics**, and **artifacts**
* Organizes experiments for easy comparison

### **📂 DVC**

* **Data Versioning**: `dvc add` and `.dvc` files
* **Pipeline Versioning**: via `dvc.yaml`
* **Model Versioning**: models as DVC-tracked artifacts

---

## **🚀 Future Enhancements**

* Automated Monitoring & Alerts (email, Slack)
* CI/CD for training & deployment
* Docker-based containerization
* Cloud deployment (AWS, GCP, Azure)
* Drift detection (data & model)
* MLflow **Model Registry**
* Hyperparameter tuning (Optuna, Hyperopt)
* Deep learning models / Ensembles
* Testing suite (unit + integration)

---

## **📝 License**

This project is licensed under the **MIT License**.

---

## **📬 Contact**

For issues or suggestions, please open an issue on the [GitHub repository](#).

---


