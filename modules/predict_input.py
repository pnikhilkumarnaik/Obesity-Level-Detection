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

    # Correctly encode categorical columns
    for col in cat_cols:
        if col in input_df:
            le = LabelEncoder()
            # Fit on the input value itself (single row)
            input_df[col] = le.fit([input_dict[col]]).transform([input_dict[col]])

    prediction = model.predict(input_df)[0]
    print("Predicted Obesity Level:", class_mapping.get(prediction, "Unknown"))
    return class_mapping.get(prediction, "Unknown")
