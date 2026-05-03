import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("Titanic_Model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details below:")

# Inputs
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.radio("Sex", ["Male", "Female"])
Age = st.slider("Age", 0, 80, 25)
SibSp = st.slider("Siblings/Spouses", 0, 5, 0)
Parch = st.slider("Parents/Children", 0, 5, 0)
Fare = st.slider("Fare", 0, 500, 50)
Embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert inputs
Sex = 1 if Sex == "Male" else 0
Embarked_Q = 1 if Embarked == "Q" else 0
Embarked_S = 1 if Embarked == "S" else 0
FamilySize = SibSp + Parch

# Arrange input (must match training order)
input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, FamilySize, Embarked_Q, Embarked_S]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")

    st.write(f"Confidence: {round(prob[0][1]*100, 2)}%")