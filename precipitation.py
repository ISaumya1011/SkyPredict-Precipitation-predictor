import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split  
import joblib

# Load the dataset
df = pd.read_csv("weatherAUS.csv")

# Separate features and target variable
x = df.drop(columns=['RainTomorrow']).values
Y = df['RainTomorrow'].values

# Handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
x = imputer.fit_transform(x)

# Convert categorical columns to numeric using LabelEncoder
categorical_columns = [3, 5, 6]  # Indices of categorical columns
le_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    x[:, col] = le.fit_transform(x[:, col])
    le_encoders[col] = le

# Encode the target variable
le_target = LabelEncoder()
Y = le_target.fit_transform(Y)

# Scale features
sc = StandardScaler()
x = sc.fit_transform(x)

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2, random_state=0)

# Train the XGBoost model
from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train, Y_train)

# Make predictions and evaluate the model
predictions = xg.predict(X_test)

# Save the trained model and other necessary objects
joblib.dump(xg, 'xgboost_rain_prediction_model.pkl')
joblib.dump(sc, 'scaler.pkl')
joblib.dump(le_encoders, 'label_encoders.pkl')
joblib.dump(le_target, 'label_encoder_target.pkl')

# --------- Inference/Predicting with New Data ---------

# Load the trained XGBoost model
xg = joblib.load('xgboost_rain_prediction_model.pkl')

# Load the scaler and label encoders used in training
scaler = joblib.load('scaler.pkl')
le_encoders = joblib.load('label_encoders.pkl')
le_target = joblib.load('label_encoder_target.pkl')

