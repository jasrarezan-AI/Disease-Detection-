
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model and label encoder
try:
    model = joblib.load('best_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model or Label Encoder file not found. Please ensure 'best_model.pkl' and 'label_encoder.pkl' are in the same directory.")
    st.stop()

st.title("Diagnosis Prediction App")

st.write("Enter the patient's characteristics to predict their diagnosis.")

csv_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if csv_file is not None:
    try:
        input_df = pd.read_csv(csv_file)
        st.write("Uploaded CSV file:")
        st.write(input_df.head())

        # Ensure columns match the training data
        expected_columns = [
            "Memory Recall (%)", "Gait Speed (m/s)", "Tremor Frequency (Hz)",
            "Speech Rate (wpm)", "Reaction Time (ms)",
            "Eye Movement Irregularities (saccades/s)", "Sleep Disturbance (scale 0-10)",
            "Cognitive Test Score (MMSE)", "Blood Pressure (mmHg)", "Cholesterol (mg/dL)",
            "Diabetes", "Severity", "Gender_Male"
        ]
        input_df = input_df[expected_columns]

        # Make the prediction
        predictions_encoded = model.predict(input_df)

        # Decode the predictions
        predicted_diagnoses = label_encoder.inverse_transform(predictions_encoded)

        # Add predictions to the DataFrame
        input_df['Predicted Diagnosis'] = predicted_diagnoses

        st.subheader("Predictions for Uploaded Data:")
        st.write(input_df[['Predicted Diagnosis']])

        st.subheader("Original Data with Predicted Diagnoses:")
        st.write(input_df)

        # Add download button
        @st.cache_data
        def convert_df_to_csv(df):
            # IMPORTANT: This is useful for Streamlit's download_button
            return df.to_csv(index=False).encode('utf-8')

        csv_for_download = convert_df_to_csv(input_df)

        st.download_button(
            label="Download Predictions as CSV",
            data=csv_for_download,
            file_name='predictions.csv',
            mime='text/csv',
        )




        # Further processing and prediction will go here
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")



# Create input fields for each feature
# You need to replace these with the actual feature names from your dataset
# Example:
memory_recall = st.number_input("Memory Recall (%)", min_value=0.0, max_value=100.0, value=80.0)
gait_speed = st.number_input("Gait Speed (m/s)", min_value=0.0, value=1.0)
tremor_frequency = st.number_input("Tremor Frequency (Hz)", value=5.0)
speech_rate = st.number_input("Speech Rate (wpm)", value=120.0)
reaction_time = st.number_input("Reaction Time (ms)", value=300.0)
eye_movement = st.number_input("Eye Movement Irregularities (saccades/s)", value=6.0)
sleep_disturbance = st.slider("Sleep Disturbance (scale 0-10)", 0, 10, 5)
cognitive_score = st.number_input("Cognitive Test Score (MMSE)", min_value=0.0, max_value=30.0, value=25.0)
blood_pressure = st.number_input("Blood Pressure (mmHg)", value=120.0)
cholesterol = st.number_input("Cholesterol (mg/dL)", value=200.0)
diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
severity = st.number_input("Severity (0-2)", min_value=0.0, max_value=2.0, value=1.0)
gender_male = st.selectbox("Gender", [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')


# Create a button to make predictions
if st.button("Predict Diagnosis"):
    # Create a pandas DataFrame from the input values
    input_data = pd.DataFrame([[
        memory_recall, gait_speed, tremor_frequency, speech_rate, reaction_time,
        eye_movement, sleep_disturbance, cognitive_score, blood_pressure,
        cholesterol, diabetes, severity, gender_male
    ]], columns=[
        "Memory Recall (%)", "Gait Speed (m/s)", "Tremor Frequency (Hz)",
        "Speech Rate (wpm)", "Reaction Time (ms)",
        "Eye Movement Irregularities (saccades/s)", "Sleep Disturbance (scale 0-10)",
        "Cognitive Test Score (MMSE)", "Blood Pressure (mmHg)", "Cholesterol (mg/dL)",
        "Diabetes", "Severity", "Gender_Male"
    ])

    # Make the prediction
    prediction_encoded = model.predict(input_data)

    # Decode the prediction
    predicted_diagnosis = label_encoder.inverse_transform(prediction_encoded)

    # Display the prediction
    st.subheader("Predicted Diagnosis:")
    st.info(predicted_diagnosis[0])

