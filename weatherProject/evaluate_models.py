import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('weather.csv')

# Load pre-trained models and label encoder
rain_model = joblib.load('rain_model.pkl')
le = joblib.load('label_encoder.pkl')

# Define features (same as training)
FEATURE_COLS = [
    'MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 
    'Pressure', 'Temp', 'CloudCover', 'Evaporation', 
    'Sunshine', 'WindSpeed_9am', 'WindSpeed_3pm', 'Humidity_9am', 
    'Humidity_3pm', 'Pressure_9am', 'Pressure_3pm'
]

# --------------------------
# Evaluate Rain Prediction Model (Classification)
# --------------------------
print('='*50)
print('RAIN PREDICTION MODEL EVALUATION')
print('='*50)

# Prepare data for rain prediction - target is 'RainToday'
target_col_rain = 'RainToday'

# Encode target column (Yes/No to 1/0)
df[target_col_rain] = df[target_col_rain].map({'Yes': 1, 'No': 0})

# Drop rows with missing values
df_rain = df[FEATURE_COLS + [target_col_rain]].dropna()

# Encode WindGustDir
df_rain['WindGustDir'] = le.transform(df_rain['WindGustDir'])

# Split data
X_rain = df_rain[FEATURE_COLS]
y_rain = df_rain[target_col_rain]
X_train_rain, X_test_rain, y_train_rain, y_test_rain = train_test_split(
    X_rain, y_rain, test_size=0.2, random_state=42
)

# Predict
y_pred_rain = rain_model.predict(X_test_rain)
y_pred_proba_rain = rain_model.predict_proba(X_test_rain)[:, 1]

# Calculate metrics
print(f'Accuracy: {accuracy_score(y_test_rain, y_pred_rain):.4f}')
print(f'Precision: {precision_score(y_test_rain, y_pred_rain):.4f}')
print(f'Recall: {recall_score(y_test_rain, y_pred_rain):.4f}')
print(f'F1-Score: {f1_score(y_test_rain, y_pred_rain):.4f}')
#print(f'ROC-AUC: {roc_auc_score(y_test_rain, y_pred_proba_rain):.4f}')
