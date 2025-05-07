import streamlit as st
import pandas as pd
import numpy as np
import requests # To call the Flask API
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the URL of your locally running Flask API
# If running Flask in Docker, this will be different (e.g., http://flask_container_name:5000)
API_URL = os.environ.get("API_URL", "http://localhost:5000")
PREDICT_SINGLE_URL = f"{API_URL}/predict"
PREDICT_BATCH_CSV_URL = f"{API_URL}/predict_batch_csv"
STATUS_URL = f"{API_URL}/status"


# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Demand Forecasting MLOps",
    page_icon="📈",
    layout="wide"
)

st.title("Sales Demand Forecasting MLOps")
st.write("Predict sales revenue and explore historical data.")

# --- Check API Status ---
@st.cache_data(ttl=5) # Cache status for 5 seconds
def get_api_status(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "details": "API connection failed."}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "details": f"API request failed: {e}"}

api_status = get_api_status(STATUS_URL)

if api_status.get("status") == "ok":
    st.sidebar.success("API Status: Connected and Ready")
else:
    st.sidebar.error(f"API Status: {api_status.get('status', 'unknown')}. Details: {api_status.get('details', 'N/A')}")
    st.warning("The prediction API is not available. Some functionalities may be limited.")


# --- Data Upload Section ---
st.header("Upload Data (for visualization)")
uploaded_file = st.file_uploader("Upload your sales data CSV (synthetic_ecommerce_data.csv)", type=["csv"])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Basic data info
        st.write("Data Info:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    except Exception as e:
        st.error(f"Error loading file: {e}")


# --- Data Visualization Section (based on uploaded data) ---
if df is not None:
    st.header("Data Visualizations")

    # Ensure date column is datetime
    if 'Transaction_Date' in df.columns:
        try:
            df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
            df = df.sort_values('Transaction_Date').set_index('Transaction_Date')

            # Monthly Revenue Trend
            st.subheader("Monthly Revenue Trend")
            monthly_revenue = df.resample('M')['Revenue'].sum()
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(monthly_revenue)
            ax1.set_title('Monthly Revenue Trend')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Revenue')
            st.pyplot(fig1)

            # Revenue by Category
            st.subheader("Revenue by Category")
            category_revenue = df.groupby('Category')['Revenue'].sum().sort_values()
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            category_revenue.plot(kind='barh', ax=ax2)
            ax2.set_title('Revenue by Category')
            ax2.set_xlabel('Revenue')
            st.pyplot(fig2)

            # Revenue by Region
            st.subheader("Revenue by Region")
            region_rev = df.groupby('Region')['Revenue'].sum()
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            region_rev.plot(kind='bar', color='teal', ax=ax3)
            ax3.set_title('Revenue by Region')
            ax3.set_ylabel('Revenue')
            st.pyplot(fig3)

            # Add more visualizations replicating the notebook if desired

        except Exception as e:
            st.warning(f"Could not generate visualizations. Ensure data format is correct. Error: {e}")


# --- Prediction Section ---
st.header("Make Predictions")

if api_status.get("status") == "ok":
    prediction_type = st.radio(
        "Select Prediction Type:",
        ("Single Prediction", "Batch Prediction (CSV Upload)")
    )

    # --- Single Prediction ---
    if prediction_type == "Single Prediction":
        st.subheader("Single Prediction")
        st.write("Enter features for a single transaction:")

        # Input fields for features (match API PredictionInput schema)
        # These should ideally reflect the *original* columns needed for preprocessing
        date_input = st.date_input("Transaction Date")
        category_input = st.selectbox("Category", ['Electronics', 'Clothing', 'Home Appliances', 'Books', 'Toys']) # Based on notebook data
        region_input = st.selectbox("Region", ['North America', 'Europe', 'Asia']) # Based on notebook data
        units_sold_input = st.number_input("Units Sold", min_value=0.0, value=1.0)
        discount_input = st.number_input("Discount Applied", min_value=0.0, max_value=1.0, value=0.0)
        clicks_input = st.number_input("Clicks", min_value=0.0, value=0.0)
        impressions_input = st.number_input("Impressions", min_value=0.0, value=0.0)
        conversion_rate_input = st.number_input("Conversion Rate", min_value=0.0, value=0.0)
        ad_ctr_input = st.number_input("Ad CTR", min_value=0.0, value=0.0)
        ad_cpc_input = st.number_input("Ad CPC", min_value=0.0, value=0.0)
        ad_spend_input = st.number_input("Ad Spend", min_value=0.0, value=0.0)


        predict_button = st.button("Predict Revenue")

        if predict_button:
            # Construct the request body matching PredictionInput schema
            input_data = {
                "Transaction_Date": date_input.strftime("%Y-%m-%d"), # Format date as string
                "Category": category_input,
                "Region": region_input,
                "Units_Sold": units_sold_input,
                "Discount_Applied": discount_input,
                "Clicks": clicks_input,
                "Impressions": impressions_input,
                "Conversion_Rate": conversion_rate_input,
                "Ad_CTR": ad_ctr_input,
                "Ad_CPC": ad_cpc_input,
                "Ad_Spend": ad_spend_input,
            }

            try:
                response = requests.post(PREDICT_SINGLE_URL, json=input_data)
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                prediction_result = response.json()

                st.success(f"Predicted Revenue: ${prediction_result['predicted_revenue']:.2f}")

            except requests.exceptions.RequestException as e:
                st.error(f"Error calling prediction API: {e}")
                if response:
                    st.error(f"API Response: {response.text}")


    # --- Batch Prediction ---
    elif prediction_type == "Batch Prediction (CSV Upload)":
        st.subheader("Batch Prediction (CSV Upload)")
        st.write("Upload a CSV file for batch predictions. The CSV should have the same columns as the original data (including 'Transaction_Date', 'Category', 'Region', etc.).")

        batch_upload_file = st.file_uploader("Upload batch prediction CSV", type=["csv"])

        if batch_upload_file is not None:
            batch_predict_button = st.button("Run Batch Prediction")

            if batch_predict_button:
                try:
                    # Send the file to the batch prediction endpoint
                    files = {'file': batch_upload_file.getvalue()} # Get file content as bytes
                    response = requests.post(PREDICT_BATCH_CSV_URL, files=files)
                    response.raise_for_status() # Raise an HTTPError for bad responses

                    batch_prediction_results = response.json()

                    if batch_prediction_results and 'predictions' in batch_prediction_results:
                        predictions_list = [item['predicted_revenue'] for item in batch_prediction_results['predictions']]
                        predictions_df = pd.DataFrame({'Predicted Revenue': predictions_list})

                        st.success("Batch predictions completed!")
                        st.write("Batch Predictions:")
                        st.dataframe(predictions_df)

                        # Optional: Allow downloading results
                        csv_output = predictions_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv_output,
                            file_name='batch_predictions.csv',
                            mime='text/csv',
                        )

                    else:
                        st.warning("Batch prediction completed but no predictions were returned.")


                except requests.exceptions.RequestException as e:
                    st.error(f"Error calling batch prediction API: {e}")
                    if response:
                         st.error(f"API Response: {response.text}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during batch prediction: {e}")

else:
    st.info("Connect to the prediction API to enable prediction features.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("Sales Demand Forecasting MLOps Project")
st.sidebar.write("Built with MLflow, DVC, Flask, Streamlit")