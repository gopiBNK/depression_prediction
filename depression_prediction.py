import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

# Load Model Class
class DepressionMLP(nn.Module):
    def __init__(self, input_size):
        super(DepressionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# Load Artifacts
@st.cache_resource
def load_resources():
    feature_names = joblib.load('feature_names.pkl')
    model = DepressionMLP(len(feature_names))
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, joblib.load('scaler.pkl'), joblib.load('encoders.pkl'), feature_names

model, scaler, encoders, feature_names = load_resources()

st.title("🧠 Depression Risk Prediction")
st.write("Fill in the details to assess the likelihood of depression.")

# Form for user input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 25)
        gender = st.selectbox("Gender", encoders['Gender'].classes_)
        city = st.selectbox("City", encoders['City'].classes_)
        status = st.selectbox("Status", encoders['Working Professional or Student'].classes_)
        profession = st.selectbox("Profession", encoders['Profession'].classes_)
    with col2:
        sleep = st.selectbox("Sleep Duration", encoders['Sleep Duration'].classes_)
        diet = st.selectbox("Dietary Habits", encoders['Dietary Habits'].classes_)
        degree = st.selectbox("Degree", encoders['Degree'].classes_)
        suicidal = st.selectbox("Suicidal Thoughts?", encoders['Have you ever had suicidal thoughts ?'].classes_)
        history = st.selectbox("Family History?", encoders['Family History of Mental Illness'].classes_)

    st.subheader("Pressure & Stress Levels (0-10)")
    p1, p2, p3 = st.columns(3)
    academic_p = p1.slider("Academic Pressure", 0, 10, 0)
    work_p = p2.slider("Work Pressure", 0, 10, 0)
    fin_stress = p3.slider("Financial Stress", 0, 10, 0)
    
    submit = st.form_submit_button("Predict")

if submit:
    # Prepare input data matching the training format
    input_dict = {
        'Gender': gender, 'Age': age, 'City': city, 
        'Working Professional or Student': status, 'Profession': profession,
        'Academic Pressure': academic_p, 'Work Pressure': work_p, 'CGPA': 0,
        'Study Satisfaction': 0, 'Job Satisfaction': 0, 'Sleep Duration': sleep,
        'Dietary Habits': diet, 'Degree': degree,
        'Have you ever had suicidal thoughts ?': suicidal, 'Work/Study Hours': 8,
        'Financial Stress': fin_stress, 'Family History of Mental Illness': history
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Apply Encoding
    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col].astype(str))
    
    # Scale & Predict
    input_scaled = scaler.transform(input_df[feature_names])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        prob = model(input_tensor).item()
    
    st.divider()
    if prob > 0.5:
        st.error(f"High Risk Detected: {prob*100:.1f}%")
        st.write("Please consult a mental health professional.")
    else:
        st.success(f"Low Risk Detected: {prob*100:.1f}%")