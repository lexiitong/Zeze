import streamlit as st
from st_files_connection import FilesConnection
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import io
import os

# Set up AWS credentials from secrets
os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ['AWS_DEFAULT_REGION'] = st.secrets["AWS_DEFAULT_REGION"]

# Create connection object
conn = st.connection('s3', type=FilesConnection)

# Load the model
@st.cache_resource
def load_model():
    model_data = conn.read("zezeapp/calibrated_random_forest_model.pkl", input_format="binary", ttl=600)
    return joblib.load(io.BytesIO(model_data))

loaded_model = load_model()
st.success("Model loaded successfully")

# Load the cleaned dataset
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
