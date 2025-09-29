# modules/predict_input.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class_mapping = {
    0.0: "Insufficient_Weight",
    1.0: "Normal_Weight",
    2.0: "Obesity_Type_I",
    3.0: "Obesity_Type_II",
    4.0: "Obesity_Type_III",
    5.0: "Overweight_Level_I",
    6.0: "Overweight_Level_II"
}

def predict_from_input(model, input_dict, cat_cols):
    """
    Predict obesity level based on a dictionary of inputs.
    """
    input_df = pd.DataFrame([input_dict])
    le = LabelEncoder()
    for col in cat_cols:
        if col in input_df:
            input_df[col] = le.fit_transform([str(input_df[col])])[0]

    prediction = model.predict(input_df)[0]
    print("Predicted Obesity Level:", class_mapping.get(prediction, "Unknown"))
    return prediction
