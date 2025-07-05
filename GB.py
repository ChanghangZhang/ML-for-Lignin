# 梯度下降模型
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Read Excel file
data_frame = pd.read_excel(r'D:\projectPython\QA/k_lg_sol_1.xlsx')

# Extract input features and target values
X = data_frame.iloc[:, 4:-1].values
y = data_frame.iloc[:, 1].values
y = torch.Tensor(y.ravel())  # Convert to 1D tensor

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Define and train the model
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train_norm, y_train)

# Evaluate the model
y_pred = rf.predict(X_test_norm)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
correlation = np.corrcoef(y_pred, y_test)[0, 1]
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
print(f'Correlation Coefficient: {correlation:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')

# Read new feature data
feature_data_frame = pd.read_excel(r'D:\projectPython\QA/new_feature.xlsx')

# Prepare new data
new_features = feature_data_frame.iloc[:, 4:15].values

# Normalize new data
new_features_norm = scaler.transform(new_features)

# Make prediction
predicted_values = rf.predict(new_features_norm)

# Print all the predicted values
print("Predicted values:")
for i, value in enumerate(predicted_values):
    print(f"Prediction {i+1}: {value:.2f}")