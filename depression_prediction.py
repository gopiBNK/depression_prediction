import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

# Correct Model Architecture (DeepMLP)
class DepressionMLP(nn.Module):

    def __init__(self, input_size):

        super(DepressionMLP, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.network(x)


# Load saved files
@st.cache_resource
def load_resources():

    # Correct filenames
    feature_names = joblib.load("features.pkl")

    scaler = joblib.load("scale.pkl")

    encoders = joblib.load("encoder.pkl")

    # Load model
    model = DepressionMLP(len(feature_names))

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model, scaler, encoders, feature_names


model, scaler, encoders, feature_names = load_resources()

# UI Title
st.title("Depression Risk Prediction")

st.write(
    "Fill in the details to assess the likelihood of depression."
)

# Input Form
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Age", 18, 100, 25)

        gender = st.selectbox(
            "Gender",
            encoders['Gender'].classes_
        )

        city = st.selectbox(
            "City",
            encoders['City'].classes_
        )

        status = st.selectbox(
            "Working Professional or Student",
            encoders[
                'Working Professional or Student'
            ].classes_
        )

        profession = st.selectbox(
            "Profession",
            encoders['Profession'].classes_
        )

    with col2:

        sleep = st.selectbox(
            "Sleep Duration",
            encoders['Sleep Duration'].classes_
        )

        diet = st.selectbox(
            "Dietary Habits",
            encoders['Dietary Habits'].classes_
        )

        degree = st.selectbox(
            "Degree",
            encoders['Degree'].classes_
        )

        suicidal = st.selectbox(
            "Have you ever had suicidal thoughts ?",
            encoders[
                'Have you ever had suicidal thoughts ?'
            ].classes_
        )

        history = st.selectbox(
            "Family History of Mental Illness",
            encoders[
                'Family History of Mental Illness'
            ].classes_
        )

    st.subheader("Pressure & Stress Levels")

    p1, p2, p3 = st.columns(3)

    academic_p = p1.slider(
        "Academic Pressure",
        0,
        10,
        0
    )

    work_p = p2.slider(
        "Work Pressure",
        0,
        10,
        0
    )

    fin_stress = p3.slider(
        "Financial Stress",
        0,
        10,
        0
    )

    submit = st.form_submit_button("Predict")


# Prediction Logic
if submit:

    input_dict = {

        'Gender': gender,
        'Age': age,
        'City': city,

        'Working Professional or Student': status,
        'Profession': profession,

        'Academic Pressure': academic_p,
        'Work Pressure': work_p,

        'CGPA': 0,
        'Study Satisfaction': 0,
        'Job Satisfaction': 0,

        'Sleep Duration': sleep,
        'Dietary Habits': diet,
        'Degree': degree,

        'Have you ever had suicidal thoughts ?':
            suicidal,

        'Work/Study Hours': 8,

        'Financial Stress': fin_stress,

        'Family History of Mental Illness':
            history
    }

    input_df = pd.DataFrame([input_dict])

    # Encode categorical values
    for col, le in encoders.items():

        input_df[col] = le.transform(
            input_df[col].astype(str)
        )

    # Scale input
    input_scaled = scaler.transform(
        input_df[feature_names]
    )

    input_tensor = torch.tensor(
        input_scaled,
        dtype=torch.float32
    )

    # Predict
    with torch.no_grad():

        prob = model(input_tensor).item()

    st.divider()

    if prob > 0.5:

        st.error(
            f"High Risk Detected: {prob*100:.1f}%"
        )

        st.write(
            "Please consult a mental health professional."
        )

    else:

        st.success(
            f"Low Risk Detected: {prob*100:.1f}%"
        )
