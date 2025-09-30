import os
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
csv_path = os.path.join(BASE_DIR, "data", "ObesityDataSet.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found at {csv_path}")

df = pd.read_csv(csv_path)

# Encode categorical columns
cat_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
            'family_history_with_overweight', 'CAEC', 'MTRANS']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and target
X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LGBMClassifier()
model.fit(x_train, y_train)

# Save model and encoders
joblib.dump(model, os.path.join(BASE_DIR, "modules", "lgbm_model.joblib"))
joblib.dump(encoders, os.path.join(BASE_DIR, "modules", "encoders.joblib"))

print("Model and encoders saved successfully.")
