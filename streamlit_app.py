import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import boto3
from io import StringIO

# Try to import FilesConnection
try:
    from streamlit_javascript import st_javascript
    from st_files_connection import FilesConnection
except ImportError:
    FilesConnection = None

# Set up page config
st.set_page_config(page_title="Prospect Call Time Predictor", layout="wide")

# Set up AWS credentials from secrets
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets.get("AWS_ACCESS_KEY_ID", "")
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets.get("AWS_SECRET_ACCESS_KEY", "")
os.environ['AWS_DEFAULT_REGION'] = st.secrets.get("AWS_DEFAULT_REGION", "")

# Create connection object for S3 if FilesConnection is available
if FilesConnection:
    try:
        conn = st.connection('s3', type=FilesConnection)
        st.success("S3 connection established")
    except Exception as e:
        st.error(f"Failed to establish S3 connection: {str(e)}")
        conn = None
else:
    st.warning("S3 connection is not available due to missing FilesConnection")
    conn = None

# Define ordered lists for dropdown menus
months_ordered = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
weekdays_ordered = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
time_spans_ordered = ['Early morning', 'Morning', 'Mid-day', 'Early afternoon', 'Afternoon', 'Late afternoon']

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('calibrated_random_forest_model.joblib')

model = load_model()

# Function to load data from S3
@st.cache_data
def load_data_from_s3(bucket_name, file_key):
    if conn:
        try:
            df = conn.read(f"s3://{bucket_name}/{file_key}", input_format="csv", ttl=600)
            return df
        except Exception as e:
            st.error(f"Failed to load data from S3: {str(e)}")
            return None
    else:
        st.error("S3 connection is not available")
        return None

# Main app
def main():
    st.title("Prospect Call Time Predictor")

    # User input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        month = st.selectbox("Select Month", months_ordered)
    with col2:
        weekday = st.selectbox("Select Weekday", weekdays_ordered)
    with col3:
        time_span = st.selectbox("Select Time Span", time_spans_ordered)

    if st.button("Predict Best Call Time"):
        # Load confidential dataset from S3
        df = load_data_from_s3('your-bucket-name', 'path/to/your/confidential_dataset.csv')
        
        if df is not None:
            # Prepare input data
            input_data = pd.DataFrame({
                'Month': [month],
                'Weekday': [weekday],
                'TimeSpan': [time_span]
            })

            # One-hot encode the input data
            input_encoded = pd.get_dummies(input_data, columns=['Month', 'Weekday', 'TimeSpan'])

            # Ensure all columns from training are present
            for col in model.feature_names_in_:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0

            # Reorder columns to match the training data
            input_encoded = input_encoded[model.feature_names_in_]

            # Make prediction
            prediction = model.predict_proba(input_encoded)[0]

            # Display results
            st.subheader("Prediction Results")
            result_df = pd.DataFrame({
                'Time Span': time_spans_ordered,
                'Probability': prediction
            })
            result_df = result_df.sort_values('Probability', ascending=False)

            st.bar_chart(result_df.set_index('Time Span'))
            
            best_time = result_df.iloc[0]['Time Span']
            best_prob = result_df.iloc[0]['Probability']
            
            st.success(f"The best time to call is during the {best_time} with a probability of {best_prob:.2f}")
        else:
            st.error("Failed to load data. Please try again later.")

if __name__ == "__main__":
    main()
