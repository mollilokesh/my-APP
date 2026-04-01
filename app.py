import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load Dataset (Fixed)
# -----------------------------
@st.cache_data
def load_data():

    data = data = pd.read_csv(r"C:\Users\molli\Downloads\diabetes (2).csv")
    return data

data = load_data()

# -----------------------------
# Train Model
# -----------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# UI
# -----------------------------
st.title("🩺 Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 140, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, dpf, age]],
        columns=X.columns
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Diabetes Detected")
    else:
        st.success("✅ No Diabetes")