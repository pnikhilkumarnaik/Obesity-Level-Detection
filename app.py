# app.py
import streamlit as st
from modules.save_load_model import load_model
from modules.predict_input import predict_from_input

# Load model
model = load_model("lgbm_model.joblib")

# Columns info
cat_cols = ["Gender", "Family_History", "Physical_Activity", "Diet"]
num_cols = ["Age", "Height", "Weight"]

# Streamlit UI
st.title("Obesity Level Prediction App")
st.markdown("Enter your details below:")

# Categorical inputs
input_dict = {}
input_dict['Gender'] = st.selectbox("Gender", ["Male", "Female"])
input_dict['Family_History'] = st.selectbox("Family History", ["Yes", "No"])
input_dict['Physical_Activity'] = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
input_dict['Diet'] = st.selectbox("Diet", ["Poor", "Average", "Good"])

# Numeric inputs
input_dict['Age'] = st.number_input("Age", min_value=1, max_value=120, value=25)
input_dict['Height'] = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
input_dict['Weight'] = st.number_input("Weight (kg)", min_value=10, max_value=200, value=70)

# Predict button
if st.button("Predict Obesity Level"):
    prediction = predict_from_input(model, input_dict, cat_cols)
    st.success(f"Predicted Obesity Level: {prediction}")
