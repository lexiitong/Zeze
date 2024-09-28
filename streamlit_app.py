import streamlit as st
from st_files_connection import FilesConnection
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import io
import os
import joblib
import sys
import sklearn
import errno
import requests
import tempfile
import base64

# Try to import FilesConnection, but provide a fallback if it's not available
try:
    from st_files_connection import FilesConnection
    st.success("Successfully imported FilesConnection")
except ImportError:
    st.warning("Could not import FilesConnection. S3 connection might not be available.")
    FilesConnection = None

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

@st.cache_resource
def load_model():
    try:
        # GitHub API URL for the file
        api_url = "https://api.github.com/repos/lexiitong/Zeze/contents/calibrated_random_forest_model.joblib"
        
        st.write(f"Attempting to fetch model from GitHub API: {api_url}")
        
        # Fetch file content using GitHub API
        response = requests.get(api_url)
        response.raise_for_status()
        file_content = response.json()["content"]
        
        # Decode the base64 content
        decoded_content = base64.b64decode(file_content)
        
        st.write("Model content fetched successfully")
        
        # Load the model directly from the decoded content
        model = joblib.load(io.BytesIO(decoded_content))
        
        st.write(f"Model type: {type(model)}")
        st.write(f"scikit-learn version: {sklearn.__version__}")
        
        return model
    except requests.RequestException as e:
        st.error(f"Failed to fetch the model: {str(e)}")
    except ValueError as e:
        st.error(f"Error decoding model content: {str(e)}")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        st.write("Python version:", sys.version)
        st.write("joblib version:", joblib.__version__)
        st.write("scikit-learn version:", sklearn.__version__)
        raise
        
try:
    loaded_model = load_model()
    if loaded_model is not None:
        st.success("Model loaded successfully")
    else:
        st.error("Failed to load the model.")
        st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.stop()
    
# Load the cleaned dataset from S3
@st.cache_data
def load_data():
    df = conn.read("zezeapp/Felix_cleaned_dataset160824.csv", input_format="csv", ttl=600)
    df = df.drop(['Subject', 'Completed Date/Time', 'Completed Date', 'Completed Time', 'Seasonality'], axis=1)
    df_encoded = pd.get_dummies(df, columns=['Weekday', 'Month', 'Time-span'])
    X = df_encoded.drop(['Answer Status', 'Answered'], axis=1)
    return X

X = load_data()
X_columns = X.columns

# Define ordered options for dropdowns
months_ordered = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
weekdays_ordered = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
time_spans_ordered = ['Early morning', 'Morning', 'Mid-day', 'Early afternoon', 'Afternoon', 'Late afternoon']

# Create an Interactive Form
st.title("Call Outcome Predictor")

month = st.selectbox('Month:', months_ordered)
weekday = st.selectbox('Weekday:', weekdays_ordered)
time_span = st.selectbox('Time-span:', time_spans_ordered)

if st.button("Predict"):
    new_data = {column: 0 for column in X.columns}
    new_data[f'Month_{month}'] = 1
    new_data[f'Weekday_{weekday}'] = 1
    new_data[f'Time-span_{time_span}'] = 1
    new_data_encoded = pd.DataFrame([new_data])[X_columns]
    prob = loaded_model.predict_proba(new_data_encoded)[0][1]
    message = f'Your call has {prob*100:.2f}% chances of being answered.'
    if prob > 0.30:
        message += " Go for it, it is higher than usual!"
    st.success(message)

# Show the Heatmap for the Current Month
def show_current_month_heatmap():
    today = datetime.today()
    current_month = today.strftime("%B")
    probabilities = []

    for weekday in weekdays_ordered:
        weekday_probs = []
        for time_span in time_spans_ordered:
            new_data = {column: 0 for column in X.columns}
            new_data[f'Month_{current_month}'] = 1
            new_data[f'Weekday_{weekday}'] = 1
            new_data[f'Time-span_{time_span}'] = 1
            new_data_encoded = pd.DataFrame([new_data])[X_columns]
            prob = loaded_model.predict_proba(new_data_encoded)[0][1]
            weekday_probs.append(prob)
        probabilities.append(weekday_probs)

    prob_df = pd.DataFrame(probabilities, index=weekdays_ordered, columns=time_spans_ordered)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(prob_df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    plt.title(f"Probability of Calls Being Answered in {current_month}")
    plt.xlabel("Time-span")
    plt.ylabel("Weekday")
    return fig

st.subheader("Heatmap for Current Month")
st.pyplot(show_current_month_heatmap())
