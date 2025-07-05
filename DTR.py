# 梯度下降模型
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Read Excel file
data_frame = pd.read_excel(r'D:\projectPython\QA/k_lg_sol_0130.xlsx')

# 提取特征列 1-13 和 15-28
X_part1 = data_frame.iloc[:, 1:14].values  # 列 1-13（Python 中左闭右开，所以 1:14）
X_part2 = data_frame.iloc[:, 15:29].values  # 列 15-28（15:29）

# 将两部分特征合并为一个数组
X = np.hstack((X_part1, X_part2))  # 水平拼接

# 提取目标值列
y = data_frame.iloc[:, 0].values

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
feature_data_frame = pd.read_excel(r'D:\projectPython\QA/test_feature.xlsx')

# 提取特征列 1-13 和 14-28
x_part1 = feature_data_frame.iloc[:, 3:16].values  # 列 1-13（Python 中左闭右开，所以 1:14）
x_part2 = feature_data_frame.iloc[:, 17:31].values  # 列 15-28（14:29）

# 将两部分特征合并为一个数组
x = np.hstack((x_part1, x_part2))  # 水平拼接

# 提取目标值列
new_features = x

# Normalize new data
new_features_norm = scaler.transform(new_features)

# Make prediction
predicted_values = rf.predict(new_features_norm)

# Print all the predicted values
print("Predicted values:")
for i, value in enumerate(predicted_values):
    print(f"Prediction {i+1}: {value:.2f}")

# Ensure y_test and y_pred are 1D arrays
y_test = y_test.ravel()  # 确保 y_test 是一维数组
y_pred = y_pred.ravel()  # 确保 y_pred 是一维数组

# Create a parity plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predictions')  # Scatter plot of actual vs predicted values
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')  # Diagonal line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Parity Plot: Actual vs Predicted Values')

# Add error bounds (e.g., ±10%)
error_bound = 0.1 * np.mean(y_test)  # Use mean for relative error bound
plt.fill_between([min(y_test), max(y_test)],
                 [min(y_test) - error_bound, max(y_test) - error_bound],
                 [min(y_test) + error_bound, max(y_test) + error_bound],
                 color='gray', alpha=0.2, label='±10% Error Bound')

plt.legend()
plt.grid(True)
plt.show()