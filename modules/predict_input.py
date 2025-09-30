# modules/predict_input.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Mapping of numerical classes to readable labels
class_mapping = {
    0.0: "Insufficient_Weight",
    1.0: "Normal_Weight",
    2.0: "Obesity_Type_I",
    3.0: "Obesity_Type_II",
    4.0: "Obesity_Type_III",
    5.0: "Overweight_Level_I",
    6.0: "Overweight_Level_II"
}

def predict_from_input(model, input_dict, cat_cols, encoders):
    """
    Predicts obesity level for a single input dictionary.
    
    Parameters:
    - model: Trained LightGBM model
    - input_dict: Dictionary with feature names and their values
    - cat_cols: List of categorical column names
    - encoders: Dictionary of LabelEncoders for categorical columns
    
    Returns:
    - Predicted class as readable string
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Encode categorical columns using pre-fitted encoders
    for col in cat_cols:
        if col in input_df and col in encoders:
            input_df[col] = encoders[col].transform([input_df[col].iloc[0]])
    
    # Predict using the trained model
    prediction = model.predict(input_df)[0]
    
    # Convert numeric prediction to readable class
    return class_mapping.get(prediction, "Unknown")


def preprocess_and_test(model, csv_path, cat_cols, encoders, random_record_number):
    """
    Test a random record from dataset to check prediction accuracy.
    
    Parameters:
    - model: Trained LightGBM model
    - csv_path: Path to CSV dataset
    - cat_cols: List of categorical columns
    - encoders: Dictionary of LabelEncoders
    - random_record_number: Index of record to test
    
    Prints actual vs predicted value
    """
    df = pd.read_csv(csv_path)
    
    # Encode categorical columns using pre-fitted encoders
    for col in cat_cols:
        if col in df and col in encoders:
            df[col] = encoders[col].transform(df[col])
    
    # Select record
    record = pd.DataFrame(df.iloc[random_record_number]).T
    actual_class = record['NObeyesdad'].iloc[0]
    print("Actual Value:", class_mapping.get(actual_class, "Unknown"))
    
    record.drop(columns=['NObeyesdad'], inplace=True)
    predicted_class = model.predict(record)[0]
    print("Predicted Value:", class_mapping.get(predicted_class, "Unknown"))
