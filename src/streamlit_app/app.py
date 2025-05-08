import streamlit as st
import pandas as pd
import numpy as np
import requests # To call the Flask API
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io # Import io for StringIO/BytesIO
import altair as alt # Import Altair for interactive plots
from datetime import date, timedelta # Import date and timedelta for date inputs

# Define the URL of your locally running Flask API
# If running Flask in Docker, this will be different (e.g., http://flask_container_name:5000)
API_URL = os.environ.get("API_URL", "http://localhost:5001") # Use port 5001 for API
PREDICT_SINGLE_URL = f"{API_URL}/predict"
PREDICT_BATCH_CSV_URL = f"{API_URL}/predict_batch_csv"
STATUS_URL = f"{API_URL}/status"
FORECAST_SARIMA_URL = f"{API_URL}/forecast_sarima" # New endpoint for SARIMA forecast


# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Demand Forecasting MLOps",
    page_icon="📈",
    layout="wide" # Use wide layout
)

st.title("Sales Demand Forecasting MLOps")
st.write("Predict sales revenue and explore historical data.")

# --- Check API Status ---
@st.cache_data(ttl=5) # Cache status for 5 seconds
def get_api_status(url):
    """Checks the status of the prediction API."""
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
    # Check if SARIMA model is specifically loaded for forecasting section
    if not api_status.get("sarima_model_loaded", False):
         st.sidebar.warning("SARIMA model not loaded. Forecasting unavailable.")
else:
    st.sidebar.error(f"API Status: {api_status.get('status', 'unknown')}. Details: {api_status.get('details', 'N/A')}")
    st.warning("The prediction API is not available. Some functionalities may be limited.")


# --- Data Upload Section (for visualization) ---
st.header("Upload Data (for visualization)")
uploaded_file = st.file_uploader("Upload your sales data CSV (synthetic_ecommerce_data.csv) for visualization", type=["csv"])

df = None
if uploaded_file is not None:
    try:
        # Read the uploaded file into a DataFrame for visualization
        # Reset the file pointer to the beginning after reading for potential reuse
        uploaded_file.seek(0)
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
    st.header("Historical Data Visualizations")

    # Ensure date column is datetime and set as index
    if 'Transaction_Date' in df.columns:
        try:
            df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
            # Check if index is already datetime before setting
            if not isinstance(df.index, pd.DatetimeIndex):
                 df = df.sort_values('Transaction_Date').set_index('Transaction_Date')

            # --- Enterprise-Level Historical Analysis ---

            st.subheader("Historical Sales Analysis")

            # Add Filters to Sidebar for Historical Data
            st.sidebar.subheader("Historical Data Filters")

            min_date_hist = df.index.min().date()
            max_date_hist = df.index.max().date()
            date_range_hist = st.sidebar.date_input(
                "Select Date Range:",
                [min_date_hist, max_date_hist],
                min_value=min_date_hist,
                max_value=max_date_hist
            )

            filtered_hist_df = df.copy()
            if len(date_range_hist) == 2:
                start_date = min(date_range_hist)
                end_date = max(date_range_hist)
                filtered_hist_df = filtered_hist_df[(filtered_hist_df.index.date >= start_date) & (filtered_hist_df.index.date <= end_date)]
            elif len(date_range_hist) == 1:
                 filtered_hist_df = filtered_hist_df[filtered_hist_df.index.date == date_range_hist[0]] # Corrected variable name


            # Category Filter
            if 'Category' in filtered_hist_df.columns:
                 all_categories = ['All'] + filtered_hist_df['Category'].unique().tolist()
                 selected_categories = st.sidebar.multiselect("Select Categories:", all_categories, default=['All'])
                 if 'All' not in selected_categories:
                      filtered_hist_df = filtered_hist_df[filtered_hist_df['Category'].isin(selected_categories)]

            # Region Filter
            if 'Region' in filtered_hist_df.columns:
                 all_regions = ['All'] + filtered_hist_df['Region'].unique().tolist()
                 selected_regions = st.sidebar.multiselect("Select Regions:", all_regions, default=['All'])
                 if 'All' not in selected_regions:
                      filtered_hist_df = filtered_hist_df[filtered_hist_df['Region'].isin(selected_regions)]


            # Display KPIs based on Filtered Data
            if not filtered_hist_df.empty:
                 total_revenue_hist = filtered_hist_df['Revenue'].sum()
                 num_transactions_hist = len(filtered_hist_df)
                 avg_revenue_per_transaction = filtered_hist_df['Revenue'].mean()
                 num_unique_categories = filtered_hist_df['Category'].nunique() if 'Category' in filtered_hist_df.columns else 0
                 num_unique_regions = filtered_hist_df['Region'].nunique() if 'Region' in filtered_hist_df.columns else 0

                 st.subheader("Key Historical Metrics (Filtered Data)")
                 col1_hist, col2_hist, col3_hist, col4_hist = st.columns(4)
                 col1_hist.metric("Total Revenue", f"${total_revenue_hist:,.2f}")
                 col2_hist.metric("Number of Transactions", f"{num_transactions_hist:,}")
                 col3_hist.metric("Avg Revenue per Transaction", f"${avg_revenue_per_transaction:,.2f}")
                 col4_hist.metric("Unique Categories", f"{num_unique_categories:,}")
                 # Can add Unique Regions here or in another row

                 if 'Region' in filtered_hist_df.columns:
                      st.metric("Unique Regions", f"{num_unique_regions:,}")


                 # --- Historical Trend Analysis (using Altair) ---
                 st.subheader("Historical Revenue Trends")

                 # Monthly Trend
                 monthly_revenue_hist = filtered_hist_df.resample('ME')['Revenue'].sum().reset_index()
                 monthly_revenue_hist.columns = ['Date', 'Revenue']
                 st.write("Monthly Revenue Trend:")
                 chart_monthly_hist = alt.Chart(monthly_revenue_hist).mark_line().encode(
                     x=alt.X('Date:T', title='Date'),
                     y=alt.Y('Revenue:Q', title='Revenue'),
                     tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Revenue:Q', format='$,.2f')]
                 ).properties(
                     title='Monthly Historical Revenue Trend'
                 ).interactive()
                 st.altair_chart(chart_monthly_hist, use_container_width=True)

                 # Quarterly Trend
                 quarterly_revenue_hist = filtered_hist_df.resample('QE')['Revenue'].sum().reset_index()
                 quarterly_revenue_hist.columns = ['Date', 'Revenue']
                 st.write("Quarterly Revenue Trend:")
                 chart_quarterly_hist = alt.Chart(quarterly_revenue_hist).mark_line().encode(
                     x=alt.X('Date:T', title='Date'),
                     y=alt.Y('Revenue:Q', title='Revenue'),
                     tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Revenue:Q', format='$,.2f')]
                 ).properties(
                     title='Quarterly Historical Revenue Trend'
                 ).interactive()
                 st.altair_chart(chart_quarterly_hist, use_container_width=True)

                 # Yearly Trend
                 yearly_revenue_hist = filtered_hist_df.resample('YE')['Revenue'].sum().reset_index()
                 yearly_revenue_hist.columns = ['Date', 'Revenue']
                 st.write("Yearly Revenue Trend:")
                 chart_yearly_hist = alt.Chart(yearly_revenue_hist).mark_line().encode(
                     x=alt.X('Date:T', title='Date'),
                     y=alt.Y('Revenue:Q', title='Revenue'),
                     tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Revenue:Q', format='$,.2f')]
                 ).properties(
                     title='Yearly Historical Revenue Trend'
                 ).interactive()
                 st.altair_chart(chart_yearly_hist, use_container_width=True)


                 # --- Distribution Analysis ---
                 st.subheader("Historical Revenue Distribution")

                 st.write("Distribution of Transaction Revenue:")
                 fig_hist_dist, ax_hist_dist = plt.subplots()
                 sns.histplot(filtered_hist_df['Revenue'], kde=True, ax=ax_hist_dist)
                 ax_hist_dist.set_title("Histogram of Historical Transaction Revenue")
                 ax_hist_dist.set_xlabel("Revenue")
                 ax_hist_dist.set_ylabel("Frequency")
                 st.pyplot(fig_hist_dist)

                 if 'Category' in filtered_hist_df.columns:
                      st.write("Distribution of Revenue by Category (Boxplot):")
                      fig_hist_cat_box, ax_hist_cat_box = plt.subplots(figsize=(10, 6))
                      sns.boxplot(x='Category', y='Revenue', data=filtered_hist_df, ax=ax_hist_cat_box)
                      ax_hist_cat_box.set_title("Historical Revenue Distribution by Category")
                      ax_hist_cat_box.set_xlabel("Category")
                      ax_hist_cat_box.set_ylabel("Revenue")
                      plt.xticks(rotation=45, ha='right')
                      st.pyplot(fig_hist_cat_box)

                 if 'Region' in filtered_hist_df.columns:
                      st.write("Distribution of Revenue by Region (Boxplot):")
                      fig_hist_region_box, ax_hist_region_box = plt.subplots(figsize=(10, 6))
                      sns.boxplot(x='Region', y='Revenue', data=filtered_hist_df, ax=ax_hist_region_box)
                      ax_hist_region_box.set_title("Historical Revenue Distribution by Region")
                      ax_hist_region_box.set_xlabel("Region")
                      ax_hist_region_box.set_ylabel("Revenue")
                      plt.xticks(rotation=45, ha='right')
                      st.pyplot(fig_hist_region_box)


                 # --- Segment Analysis ---
                 st.subheader("Historical Revenue Segment Analysis")

                 if 'Category' in filtered_hist_df.columns and 'Region' in filtered_hist_df.columns:
                      st.write("Average Revenue by Category and Region (Heatmap):")
                      # Ensure data is numeric for heatmap
                      heatmap_data = filtered_hist_df.pivot_table(index='Category', columns='Region', values='Revenue', aggfunc='mean').fillna(0)
                      fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 7))
                      sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis", ax=ax_heatmap)
                      ax_heatmap.set_title("Average Historical Revenue by Category and Region")
                      st.pyplot(fig_heatmap)

                      st.write("Total Revenue by Category and Region (Stacked Bar Chart):")
                      # Ensure data is numeric for chart
                      stacked_bar_data = filtered_hist_df.groupby(['Category', 'Region'])['Revenue'].sum().reset_index()
                      chart_stacked_bar = alt.Chart(stacked_bar_data).mark_bar().encode(
                          x=alt.X('Category:N', title='Category'), # :N for Nominal
                          y=alt.Y('Revenue:Q', title='Total Revenue'),
                          color='Region:N',
                          tooltip=['Category', 'Region', alt.Tooltip('Revenue:Q', format='$,.2f')]
                      ).properties(
                          title='Total Historical Revenue by Category and Region'
                      ).interactive()
                      st.altair_chart(chart_stacked_bar, use_container_width=True)


                 # --- Raw Data Table (Filtered) ---
                 st.subheader("Filtered Historical Data")
                 st.dataframe(filtered_hist_df)


            else:
                 st.warning("No data available for the selected historical filters.")


        except Exception as e:
            st.error(f"An error occurred during historical data visualization: {e}")
            import traceback
            traceback.print_exc()


# --- Prediction Section ---
st.header("Make Predictions")

if api_status.get("status") == "ok":
    prediction_type = st.radio(
        "Select Prediction Type:",
        ("Single Prediction", "Batch Prediction (CSV Upload)", "Time-Series Forecast (SARIMA)") # Added SARIMA option
    )

    # --- Single Prediction (Transactional Model) ---
    if prediction_type == "Single Prediction":
        st.subheader("Single Prediction (Transactional Model)")
        st.write("Enter features for a single transaction to predict its revenue.")

        # Input fields for features (match API PredictionInput schema)
        # These should ideally reflect the *original* columns needed for preprocessing
        date_input = st.date_input("Transaction Date")
        # Provide options based on typical values, or read unique values from uploaded data if available
        category_options = ['Electronics', 'Clothing', 'Home Appliances', 'Books', 'Toys']
        region_options = ['North America', 'Europe', 'Asia']

        if df is not None: # Use uploaded data for options if available
             if 'Category' in df.columns:
                  category_options = df['Category'].unique().tolist()
             if 'Region' in df.columns:
                  region_options = df['Region'].unique().tolist()


        category_input = st.selectbox("Category", category_options)
        region_input = st.selectbox("Region", region_options)
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
            except Exception as e:
                 st.error(f"An unexpected error occurred during single prediction: {e}")


    # --- Batch Prediction (Transactional Model) ---
    elif prediction_type == "Batch Prediction (CSV Upload)":
        st.subheader("Batch Prediction (CSV Upload) (Transactional Model)")
        st.write("Upload a CSV file for batch predictions. The CSV should have the same columns as the original data (including 'Transaction_Date', 'Category', 'Region', etc.).")

        batch_upload_file = st.file_uploader("Upload batch prediction CSV", type=["csv"])

        if batch_upload_file is not None:
            batch_predict_button = st.button("Run Batch Prediction")

            if batch_predict_button:
                with st.spinner("Running batch prediction..."):
                    try:
                        # Read the original data from the uploaded file BEFORE sending to API
                        # Reset file pointer before reading
                        batch_upload_file.seek(0)
                        original_batch_df = pd.read_csv(batch_upload_file)
                        st.info(f"Read {len(original_batch_df)} records from the uploaded CSV.")

                        # Reset file pointer again before sending to API
                        batch_upload_file.seek(0)

                        # Send the file to the batch prediction endpoint
                        files = {'file': (batch_upload_file.name, batch_upload_file.getvalue())} # Send filename and content

                        response = requests.post(PREDICT_BATCH_CSV_URL, files=files)
                        response.raise_for_status() # Raise an HTTPError for bad responses

                        batch_prediction_results = response.json()

                        # --- Handle API Response ---
                        if batch_prediction_results and 'predictions' in batch_prediction_results:
                            predictions_list = [item['predicted_revenue'] for item in batch_prediction_results['predictions']]
                            predictions_df = pd.DataFrame({'Predicted Revenue': predictions_list})

                            # --- Combine Original Data and Predictions ---
                            # Ensure indices align. Assuming API returns predictions in input order.
                            # If original_batch_df had a meaningful index, reset it for concatenation.
                            original_batch_df_reset = original_batch_df.reset_index(drop=True)
                            combined_df = pd.concat([original_batch_df_reset, predictions_df], axis=1)

                            # Ensure 'Transaction_Date' is datetime in combined_df for time analysis
                            if 'Transaction_Date' in combined_df.columns:
                                try:
                                    combined_df['Transaction_Date'] = pd.to_datetime(combined_df['Transaction_Date'])
                                except Exception as e:
                                    st.warning(f"Could not convert 'Transaction_Date' to datetime in combined results: {e}")


                            st.success("Batch predictions completed!")

                            # --- Enterprise-Level Output Enhancements ---

                            st.subheader("Batch Prediction Summary")

                            # 1. Summary Metrics (KPIs)
                            total_predicted_revenue = predictions_df['Predicted Revenue'].sum()
                            num_records = len(predictions_df)
                            avg_predicted_revenue = predictions_df['Predicted Revenue'].mean()
                            min_predicted_revenue = predictions_df['Predicted Revenue'].min()
                            max_predicted_revenue = predictions_df['Predicted Revenue'].max()

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Predicted Revenue", f"${total_predicted_revenue:,.2f}")
                            col2.metric("Number of Records", f"{num_records:,}")
                            col3.metric("Average Predicted Revenue", f"${avg_predicted_revenue:,.2f}")
                            col4.metric("Min Predicted Revenue", f"${min_predicted_revenue:,.2f}")
                            # st.metric("Max Predicted Revenue", f"${max_predicted_revenue:,.2f}") # Can add max here or in stats


                            # 2. Detailed Results Table with Filtering and Sorting
                            st.subheader("Predicted Revenue per Record")
                            st.write("Explore individual predictions below. Use the filters and sorting options to analyze specific segments.")

                            # Allow filtering
                            filter_options = ['All'] + combined_df.columns.tolist()
                            filter_column = st.selectbox("Filter by column:", filter_options, key='batch_filter_col') # Add key for uniqueness

                            # Initialize filtered_df with the combined_df before filtering logic
                            filtered_df = combined_df.copy()

                            # Filtering logic
                            if filter_column != 'All':
                                # Ensure column exists before attempting to filter
                                if filter_column in combined_df.columns:
                                    unique_values = combined_df[filter_column].unique().tolist()
                                    # Handle different data types for filtering input
                                    if pd.api.types.is_numeric_dtype(combined_df[filter_column]):
                                        min_val = float(combined_df[filter_column].min())
                                        max_val = float(combined_df[filter_column].max())
                                        # Handle case where min_val == max_val to avoid slider error
                                        if min_val == max_val:
                                             st.write(f"Column '{filter_column}' has a single value: {min_val}")
                                             filtered_df = filtered_df[filtered_df[filter_column] == min_val]
                                        else:
                                             filter_value = st.slider(f"Select range for {filter_column}", min_val, max_val, (min_val, max_val), key='batch_filter_slider')
                                             filtered_df = filtered_df[(filtered_df[filter_column] >= filter_value[0]) & (filtered_df[filter_column] <= filter_value[1])]
                                    elif pd.api.types.is_datetime64_any_dtype(combined_df[filter_column]):
                                        # Convert to date for date picker
                                        min_date = combined_df[filter_column].min().date()
                                        max_date = combined_df[filter_column].max().date()
                                        # Handle case where min_date == max_date
                                        if min_date == max_date:
                                             st.write(f"Column '{filter_column}' has a single date: {min_date}")
                                             filtered_df = filtered_df[filtered_df[filter_column].dt.date == min_date]
                                        else:
                                             filter_date_range = st.date_input(f"Select date range for {filter_column}", [min_date, max_date], key='batch_filter_date')
                                             if len(filter_date_range) == 2:
                                                  # Ensure the dates are within the range of the column
                                                  start_date = min(filter_date_range)
                                                  end_date = max(filter_date_range)
                                                  filtered_df = filtered_df[(filtered_df[filter_column].dt.date >= start_date) & (filtered_df[filter_column].dt.date <= end_date)]
                                             elif len(filter_date_range) == 1:
                                                  filtered_df = filtered_df[filtered_df[filter_column].dt.date == filter_date_range[0]] # Handle single date selection
                                    else: # Categorical or other types
                                        filter_value = st.multiselect(f"Select value(s) for {filter_column}", unique_values, default=unique_values, key='batch_filter_multiselect')
                                        filtered_df = filtered_df[filtered_df[filter_column].isin(filter_value)]
                                else:
                                     st.warning(f"Column '{filter_column}' not found in the data.")


                            # Allow sorting on the filtered data (or full data if no filter)
                            # Use filtered_df for sorting if filtering was applied and resulted in data
                            df_to_sort = filtered_df if filter_column != 'All' and not filtered_df.empty else combined_df

                            sort_column = st.selectbox("Sort results by:", df_to_sort.columns.tolist(), key='batch_sort_col')
                            sort_order = st.radio("Sort order:", ("Ascending", "Descending"), key='batch_sort_order')
                            ascending = sort_order == "Ascending"

                            # Apply sorting
                            sorted_df_display = df_to_sort.sort_values(by=sort_column, ascending=ascending)

                            # Display the final sorted and filtered (or just sorted) DataFrame
                            if not sorted_df_display.empty:
                                st.dataframe(sorted_df_display)
                            else:
                                st.write("No records to display after filtering/sorting.")


                        # 3. Predicted Revenue Analysis (Enhanced)
                        st.subheader("Predicted Revenue Analysis")

                        # Descriptive Statistics
                        st.write("Descriptive Statistics:")
                        st.write(predictions_df['Predicted Revenue'].describe())

                        # Distribution of Predicted Revenue (Histogram)
                        st.write("Distribution of Predicted Revenue:")
                        fig_hist, ax_hist = plt.subplots()
                        sns.histplot(predictions_df['Predicted Revenue'], kde=True, ax=ax_hist)
                        ax_hist.set_title("Histogram of Predicted Revenue")
                        ax_hist.set_xlabel("Predicted Revenue")
                        ax_hist.set_ylabel("Frequency")
                        st.pyplot(fig_hist)

                        # Analysis by Category (Table and Chart)
                        if 'Category' in combined_df.columns and 'Predicted Revenue' in combined_df.columns:
                             st.subheader("Predicted Revenue by Category")
                             cat_analysis = combined_df.groupby('Category')['Predicted Revenue'].agg(['sum', 'mean', 'count']).reset_index()
                             cat_analysis.columns = ['Category', 'Total Predicted Revenue', 'Average Predicted Revenue', 'Number of Records']
                             st.write("Summary Table:")
                             st.dataframe(cat_analysis)

                             st.write("Average Predicted Revenue by Category:")
                             fig_cat_bar, ax_cat_bar = plt.subplots(figsize=(10, 6))
                             sns.barplot(x='Average Predicted Revenue', y='Category', data=cat_analysis.sort_values('Average Predicted Revenue', ascending=False), ax=ax_cat_bar)
                             ax_cat_bar.set_title("Average Predicted Revenue by Category")
                             ax_cat_bar.set_xlabel("Average Predicted Revenue")
                             ax_cat_bar.set_ylabel("Category")
                             plt.xticks(rotation=45, ha='right') # Rotate labels for readability
                             st.pyplot(fig_cat_bar)

                             st.write("Distribution by Category (Boxplot):")
                             fig_cat_box, ax_cat_box = plt.subplots(figsize=(10, 6))
                             sns.boxplot(x='Category', y='Predicted Revenue', data=combined_df, ax=ax_cat_box)
                             ax_cat_box.set_title("Predicted Revenue Distribution by Category")
                             ax_cat_box.set_xlabel("Category")
                             ax_cat_box.set_ylabel("Predicted Revenue")
                             plt.xticks(rotation=45, ha='right') # Rotate labels for readability
                             st.pyplot(fig_cat_box)


                        # Analysis by Region (Table and Chart)
                        if 'Region' in combined_df.columns and 'Predicted Revenue' in combined_df.columns:
                             st.subheader("Predicted Revenue by Region")
                             region_analysis = combined_df.groupby('Region')['Predicted Revenue'].agg(['sum', 'mean', 'count']).reset_index()
                             region_analysis.columns = ['Region', 'Total Predicted Revenue', 'Average Predicted Revenue', 'Number of Records']
                             st.write("Summary Table:")
                             st.dataframe(region_analysis)

                             st.write("Average Predicted Revenue by Region:")
                             fig_region_bar, ax_region_bar = plt.subplots(figsize=(10, 6))
                             sns.barplot(x='Average Predicted Revenue', y='Region', data=region_analysis.sort_values('Average Predicted Revenue', ascending=False), ax=ax_region_bar)
                             ax_region_bar.set_title("Average Predicted Revenue by Region")
                             ax_region_bar.set_xlabel("Average Predicted Revenue")
                             ax_region_bar.set_ylabel("Region")
                             st.pyplot(fig_region_bar)

                             st.write("Distribution by Region (Boxplot):")
                             fig_region_box, ax_region_box = plt.subplots(figsize=(10, 6))
                             sns.boxplot(x='Region', y='Predicted Revenue', data=combined_df, ax=ax_region_box)
                             ax_region_box.set_title("Predicted Revenue Distribution by Region")
                             ax_region_box.set_xlabel("Region")
                             ax_region_box.set_ylabel("Predicted Revenue")
                             plt.xticks(rotation=45, ha='right') # Rotate labels for readability
                             st.pyplot(fig_region_box)

                        # Analysis by Time (Monthly/Quarterly/Yearly) - Requires Transaction_Date
                        if 'Transaction_Date' in combined_df.columns and 'Predicted Revenue' in combined_df.columns:
                             st.subheader("Predicted Revenue Over Time")
                             combined_df['Transaction_Date'] = pd.to_datetime(combined_df['Transaction_Date']) # Ensure datetime

                             # Monthly Trend
                             monthly_pred_revenue = combined_df.resample('ME', on='Transaction_Date')['Predicted Revenue'].sum().reset_index()
                             monthly_pred_revenue.columns = ['Date', 'Predicted Revenue']
                             st.write("Monthly Predicted Revenue Trend:")
                             # Use Altair for interactive time series
                             chart_monthly = alt.Chart(monthly_pred_revenue).mark_line().encode(
                                 x=alt.X('Date:T', title='Date'), # :T for Temporal
                                 y=alt.Y('Predicted Revenue:Q', title='Predicted Revenue'), # :Q for Quantitative
                                 tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Predicted Revenue:Q', format='$,.2f')]
                             ).properties(
                                 title='Monthly Predicted Revenue Trend'
                             ).interactive() # Make the chart interactive (zoom, pan)
                             st.altair_chart(chart_monthly, use_container_width=True)


                             # Quarterly Trend
                             quarterly_pred_revenue = combined_df.resample('QE', on='Transaction_Date')['Predicted Revenue'].sum().reset_index()
                             quarterly_pred_revenue.columns = ['Date', 'Predicted Revenue']
                             st.write("Quarterly Predicted Revenue Trend:")
                             chart_quarterly = alt.Chart(quarterly_pred_revenue).mark_line().encode(
                                 x=alt.X('Date:T', title='Date'),
                                 y=alt.Y('Predicted Revenue:Q', title='Predicted Revenue'),
                                 tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Predicted Revenue:Q', format='$,.2f')]
                             ).properties(
                                 title='Quarterly Predicted Revenue Trend'
                             ).interactive()
                             st.altair_chart(chart_quarterly, use_container_width=True)

                             # Yearly Trend
                             yearly_pred_revenue = combined_df.resample('YE', on='Transaction_Date')['Predicted Revenue'].sum().reset_index()
                             yearly_pred_revenue.columns = ['Date', 'Predicted Revenue']
                             st.write("Yearly Predicted Revenue Trend:")
                             chart_yearly = alt.Chart(yearly_pred_revenue).mark_line().encode(
                                 x=alt.X('Date:T', title='Date'),
                                 y=alt.Y('Predicted Revenue:Q', title='Revenue'),
                                 tooltip=[alt.Tooltip('Date:T'), alt.Tooltip('Predicted Revenue:Q', format='$,.2f')]
                             ).properties(
                                 title='Yearly Predicted Revenue Trend'
                             ).interactive()
                             st.altair_chart(chart_yearly, use_container_width=True)


                        # 4. Top/Bottom N Records
                        st.subheader("Top/Bottom Predicted Revenue Records")
                        n_records = st.slider("Select number of records to display:", 5, 50, 10)

                        st.write(f"Top {n_records} Records by Predicted Revenue:")
                        st.dataframe(combined_df.nlargest(n_records, 'Predicted Revenue'))

                        st.write(f"Bottom {n_records} Records by Predicted Revenue:")
                        st.dataframe(combined_df.nsmallest(n_records, 'Predicted Revenue'))


                        # 5. Export Options (CSV and Excel)
                        st.subheader("Export Results")

                        # CSV Export
                        csv_output = combined_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_output,
                            file_name='batch_predictions_with_original_data.csv',
                            mime='text/csv',
                        )

                        # Excel Export (requires openpyxl: pip install openpyxl)
                        try:
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                combined_df.to_excel(writer, index=False, sheet_name='Predictions')
                            excel_buffer.seek(0)
                            st.download_button(
                                label="Download Results as Excel",
                                data=excel_buffer,
                                file_name='batch_predictions_with_original_data.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            )
                        except ImportError:
                            st.warning("Install 'openpyxl' (`pip install openpyxl`) to enable Excel export.")
                        except Exception as e:
                             st.error(f"Error during Excel export: {e}")


                        # This else is for the 'if batch_prediction_results and 'predictions' in batch_prediction_results:' block
                        else:
                             st.warning("Batch prediction completed but no predictions were returned.")


                    except requests.exceptions.RequestException as e:
                        st.error(f"Error calling batch prediction API: {e}")
                        if response:
                             st.error(f"API Response: {response.text}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during batch prediction: {e}")

    # --- Time-Series Forecast (SARIMA Model) ---
    elif prediction_type == "Time-Series Forecast (SARIMA)":
        st.subheader("Time-Series Forecast (SARIMA Model)")
        st.write("Forecast future daily revenue using the trained SARIMA model.")

        # Check if SARIMA model is loaded
        if api_status.get("sarima_model_loaded", False):
            st.info("SARIMA model is loaded and ready for forecasting.")

            # Input for forecast dates
            today = date.today()
            default_start_date = today + timedelta(days=1) # Start forecast from tomorrow
            default_end_date = today + timedelta(days=30) # Forecast for the next 30 days

            forecast_start_date = st.date_input("Forecast Start Date:", default_start_date)
            forecast_end_date = st.date_input("Forecast End Date:", default_end_date)

            forecast_button = st.button("Run Time-Series Forecast")

            if forecast_button:
                if forecast_start_date > forecast_end_date:
                    st.error("Forecast start date cannot be after end date.")
                else:
                    with st.spinner("Generating time-series forecast..."):
                        try:
                            # Prepare request body for the new API endpoint
                            forecast_input_data = {
                                "start_date": forecast_start_date.strftime("%Y-%m-%d"),
                                "end_date": forecast_end_date.strftime("%Y-%m-%d")
                            }

                            response = requests.post(FORECAST_SARIMA_URL, json=forecast_input_data)
                            response.raise_for_status() # Raise an HTTPError for bad responses

                            forecast_results = response.json()

                            if forecast_results and 'forecasts' in forecast_results:
                                # Convert results to DataFrame
                                forecast_df = pd.DataFrame(forecast_results['forecasts'])
                                forecast_df['date'] = pd.to_datetime(forecast_df['date'])
                                forecast_df = forecast_df.set_index('date').sort_index()

                                st.success("Time-series forecast completed!")

                                # Display forecast table
                                st.subheader("Forecasted Daily Revenue")
                                st.dataframe(forecast_df)

                                # Visualize forecast trend (using Altair)
                                st.subheader("Forecast Trend")
                                chart_forecast = alt.Chart(forecast_df.reset_index()).mark_line().encode(
                                    x=alt.X('date:T', title='Date'),
                                    y=alt.Y('predicted_revenue:Q', title='Predicted Revenue'),
                                    tooltip=[alt.Tooltip('date:T'), alt.Tooltip('predicted_revenue:Q', format='$,.2f')]
                                ).properties(
                                    title='SARIMA Predicted Daily Revenue Trend'
                                ).interactive()
                                st.altair_chart(chart_forecast, use_container_width=True)

                                # Optional: Combine historical data trend with forecast trend for visualization
                                if df is not None and 'Transaction_Date' in df.columns and 'Revenue' in df.columns:
                                     st.subheader("Historical and Forecasted Revenue Trend")
                                     # Aggregate historical data to daily for combining
                                     historical_daily_revenue = df.resample('D')['Revenue'].sum().reset_index()
                                     historical_daily_revenue.columns = ['date', 'Revenue']
                                     historical_daily_revenue['Type'] = 'Historical'
                                     forecast_df_display = forecast_df.reset_index()
                                     forecast_df_display.columns = ['date', 'Revenue'] # Rename for concatenation
                                     forecast_df_display['Type'] = 'Forecast'

                                     combined_trend_df = pd.concat([historical_daily_revenue, forecast_df_display])

                                     chart_combined = alt.Chart(combined_trend_df).mark_line().encode(
                                         x=alt.X('date:T', title='Date'),
                                         y=alt.Y('Revenue:Q', title='Revenue'),
                                         color='Type:N', # Color by type (Historical vs Forecast)
                                         tooltip=['date', alt.Tooltip('Revenue:Q', format='$,.2f'), 'Type']
                                     ).properties(
                                         title='Historical and Forecasted Daily Revenue Trend'
                                     ).interactive()
                                     st.altair_chart(chart_combined, use_container_width=True)


                                # Optional: Allow downloading forecast results
                                csv_output_forecast = forecast_df.to_csv().encode('utf-8') # Index is date, so no index=False needed
                                st.download_button(
                                    label="Download Forecast Results as CSV",
                                    data=csv_output_forecast,
                                    file_name='sarima_forecast.csv',
                                    mime='text/csv',
                                )


                            else:
                                st.warning("Forecast completed but no results were returned.")

                        except requests.exceptions.RequestException as e:
                            st.error(f"Error calling SARIMA forecast API: {e}")
                            if response:
                                 st.error(f"API Response: {response.text}")
                        except Exception as e:
                             st.error(f"An unexpected error occurred during SARIMA forecasting: {e}")

        else:
            st.info("SARIMA model is not loaded. Please ensure the training pipeline was run successfully.")


else:
    st.info("Connect to the prediction API to enable prediction features.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("Sales Demand Forecasting MLOps Project")
st.sidebar.write("Built with MLflow, DVC, Flask, Streamlit")
