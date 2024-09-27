import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import boto3  # You'll need to install this

# Use Streamlit's secrets to access AWS credentials
s3 = boto3.client('s3',
    aws_access_key_id=st.secrets["aws"]["access_key_id"],
    aws_secret_access_key=st.secrets["aws"]["secret_access_key"]
)

# Load the model
@st.cache_resource
def load_model():
    with open('/tmp/model.pkl', 'wb') as f:
        s3.download_fileobj('your-bucket-name', 'calibrated_random_forest_model.pkl', f)
    return joblib.load('/tmp/model.pkl')

loaded_model = load_model()
st.success("Model loaded successfully")

# Load the cleaned dataset
@st.cache_data
def load_data():
    s3.download_file('your-bucket-name', 'Felix_cleaned_dataset160824.csv', '/tmp/data.csv')
    df = pd.read_csv('/tmp/data.csv')
    df = df.drop(['Subject', 'Completed Date/Time', 'Completed Date', 'Completed Time', 'Seasonality'], axis=1)
    df_encoded = pd.get_dummies(df, columns=['Weekday', 'Month', 'Time-span'])
    X = df_encoded.drop(['Answer Status', 'Answered'], axis=1)
    return X

X = load_data()
X_columns = X.columns

# Create an Interactive Form
def create_form():
    # Define ordered options for dropdowns
    months_ordered = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
    weekdays_ordered = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    time_spans_ordered = ['Early morning', 'Morning', 'Mid-day', 'Early afternoon', 'Afternoon', 'Late afternoon']

    month_dropdown = widgets.Dropdown(
        options=months_ordered,
        description='Month:'
    )

    weekday_dropdown = widgets.Dropdown(
        options=weekdays_ordered,
        description='Weekday:'
    )

    time_span_dropdown = widgets.Dropdown(
        options=time_spans_ordered,
        description='Time-span:'
    )

    predict_button = widgets.Button(description="Predict")
    output = widgets.Output()

    def predict_call_outcome(change):
        with output:
            clear_output()
            new_data = {column: 0 for column in X.columns}
            new_data[f'Month_{month_dropdown.value}'] = 1
            new_data[f'Weekday_{weekday_dropdown.value}'] = 1
            new_data[f'Time-span_{time_span_dropdown.value}'] = 1
            new_data_encoded = pd.DataFrame([new_data])[X_columns]
            prob = loaded_model.predict_proba(new_data_encoded)[0][1]
            message = f'Your call has {prob*100:.2f}% chances of being answered.'
            if prob > 0.30:
                message += " Go for it, it is higher than usual!"
            print(message)

    predict_button.on_click(predict_call_outcome)

    display(month_dropdown, weekday_dropdown, time_span_dropdown, predict_button, output)

# Create the form
create_form()

# Show the Heatmap for the Current Month
def show_current_month_heatmap():
    today = datetime.today()
    current_month = today.strftime("%B")
    probabilities = []

    weekdays_ordered = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    time_spans_ordered = ['Early morning', 'Morning', 'Mid-day', 'Early afternoon', 'Afternoon', 'Late afternoon']

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

    plt.figure(figsize=(12, 6))
    sns.heatmap(prob_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Probability of Calls Being Answered in {current_month}")
    plt.xlabel("Time-span")
    plt.ylabel("Weekday")
    plt.show()

# Show the heatmap for the current month
show_current_month_heatmap()
