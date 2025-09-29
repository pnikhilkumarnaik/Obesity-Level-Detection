# modules/train_model.py
import time
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier

def train_lgbm(x_train, x_test, y_train, y_test):
    """
    Train LGBMClassifier and evaluate it.
    """
    model = LGBMClassifier()
    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)

  
    return model, accuracy, y_pred
