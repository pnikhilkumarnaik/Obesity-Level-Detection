import streamlit as st
import pandas as pd
import joblib
from modules.predict_input import predict_from_input

# --- Load dataset ---
df = pd.read_csv("data/ObesityDataSet.csv")

# --- Load trained model and encoders ---
model = joblib.load("modules/lgbm_model.joblib")
encoders = joblib.load("modules/encoders.joblib")

# --- Define categorical columns ---
cat_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
            'family_history_with_overweight', 'CAEC', 'MTRANS']

st.title("Obesity Level Detection App")

# --- Numeric inputs with user-friendly units and realistic ranges ---
age = st.number_input("Age (years)", min_value=10, max_value=100, value=int(df["Age"].mean()))
height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=float(df["Height"].mean()*100))
weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=float(df["Weight"].mean()))
ncp = st.number_input("Number of main meals per day (NCP)", min_value=0, max_value=6, value=int(df["NCP"].mean()))
fcvc = st.number_input("Vegetable consumption (portions/day)", min_value=0, max_value=5, value=int(df["FCVC"].mode()[0]))
ch2o = st.number_input("Water intake (liters/day)", min_value=0.5, max_value=5.0, value=float(df["CH2O"].mean()))
faf = st.number_input("Physical activity (hours/week)", min_value=0.0, max_value=10.0, value=float(df["FAF"].mean()))
tue = st.number_input("Time using technology devices (hours/day)", min_value=0.0, max_value=10.0, value=float(df["TUE"].mean()))

# --- Categorical inputs with concise, one-line headers ---
def select_default(col_name, display_name=None):
    values = df[col_name].unique()
    default_index = list(values).index(df[col_name].mode()[0])
    if display_name is None:
        display_name = col_name
    return st.selectbox(display_name, values, index=default_index)

gender = select_default("Gender", "Gender")
favc = select_default("FAVC", "High Calorie Food Consumption")
scc = select_default("SCC", "Calories Monitoring")
smoke = select_default("SMOKE", "Smoking Habit")
family_history = select_default("family_history_with_overweight", "Family History of Overweight")
calc = select_default("CALC", "Alcohol Consumption")
caec = select_default("CAEC", "Eating Between Meals")
mtrans = select_default("MTRANS", "Transportation Mode")

# --- Prepare input dictionary with proper preprocessing ---
input_dict = {
    "Age": age,
    "Height": height / 100,  # Convert cm to meters for the model
    "Weight": weight,
    "NCP": ncp,
    "FCVC": fcvc,
    "CH2O": ch2o,
    "FAF": faf,
    "TUE": tue,
    "Gender": gender,
    "FAVC": favc,
    "SCC": scc,
    "SMOKE": smoke,
    "family_history_with_overweight": family_history,
    "CALC": calc,
    "CAEC": caec,
    "MTRANS": mtrans
}

# --- Prediction button ---
if st.button("Predict Obesity Level"):
    try:
        prediction = predict_from_input(model, input_dict, cat_cols, encoders)
        st.success(f"Predicted Obesity Level: {prediction}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
