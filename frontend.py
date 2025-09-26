#importing libraries
import pandas as pd
import streamlit as st
import joblib

#loading the model from model.pkl file
model=joblib.load('model.pkl')

#creating user interface for heart disease model
st.title("Heart Disease Prediction")
st.write("Enter patient data.")

#storing the input data from user
age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Converting the gender in numerical value to pass to model  Male,Female to 1 , 0 respectivly
sex_num = 1 if sex == "Male" else 0

# storing the input data in variable to give to model to Predict
input_data = pd.DataFrame([[age, sex_num, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.error(f"High risk of heart disease (Confidence: {probability:.2%})")
    else:
        st.success(f"Low risk of heart disease (Confidence: {probability:.2%})")


