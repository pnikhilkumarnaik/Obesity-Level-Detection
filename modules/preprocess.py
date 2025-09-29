# modules/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

def preprocess_data(df, target_col='NObeyesdad', sample_size=351, final_sample=1000):
    """
    Drop duplicates, encode categorical variables, apply SMOTE, shuffle, and sample dataset.
    """
    df.drop_duplicates(inplace=True)

    # Encode categorical columns
    cat_col = [col for col in df.columns if df[col].dtype == 'object']
    le = LabelEncoder()
    for col in cat_col:
        df[col] = le.fit_transform(df[col])

    # Features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # SMOTE oversampling
    smote = SMOTE(sampling_strategy={i: sample_size for i in range(len(y.unique()))})
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                              pd.DataFrame(y_resampled, columns=[target_col])], axis=1)
    df_resampled = shuffle(df_resampled, random_state=42).reset_index(drop=True)
    df_resampled = df_resampled.sample(n=final_sample, random_state=42).reset_index(drop=True)

    return df_resampled, cat_col
