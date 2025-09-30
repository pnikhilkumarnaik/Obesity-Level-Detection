import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from joblib import dump

# Load dataset
df = pd.read_csv("ObesityDataSet.csv")  # ensure CSV is in the same folder

# Encode categorical columns
cat_cols = [col for col in df.columns if df[col].dtype == "object"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Shuffle and split
X_res, y_res = shuffle(X_res, y_res, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train LightGBM
model = LGBMClassifier(random_state=42)
model.fit(x_train, y_train)

# Optional: print training accuracy
train_acc = model.score(x_train, y_train)
print(f"Training Accuracy: {train_acc:.2f}")

# Save the trained model
dump(model, "lgbm_model.joblib")
print("Model trained and saved as 'lgbm_model.joblib'.")
