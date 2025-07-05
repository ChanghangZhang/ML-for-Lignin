# 支持向量模型
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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
X_test_norm = scaler.fit_transform(X_test)

# Define and train the model
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X_train_norm, y_train)

# Evaluate the model
y_pred = svr.predict(X_test_norm)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
correlation = np.corrcoef(y_pred, y_test)[0, 1]
rmse = np.sqrt(mse)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
print(f'Correlation Coefficient: {correlation:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')

# Plot the predicted vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted Values', s=100)

# Add the regression line
slope, intercept = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, slope * y_test + intercept, '-', lw=5, color='#7373FF', label='Regression Line')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=5, color='#FF9200', label='Actual Values')



import matplotlib.font_manager
# 设置 Times New Roman 字体
font_prop = matplotlib.font_manager.FontProperties(family='Times New Roman', size=24)
font_prop1 = matplotlib.font_manager.FontProperties(family='Times New Roman', size=20)

# 设置坐标轴刻度和标签的字体
plt.tick_params(axis='both', which='major', labelsize=18, labelcolor='black', labelrotation=0, labelright=False, labeltop=False, labelbottom=True, labelleft=True, pad=8, size=10, width=2, direction='in', colors='black', grid_color='grey', grid_alpha=0.5)
plt.tick_params(axis='both', which='minor', labelsize=12, labelcolor='black', labelrotation=0, labelright=False, labeltop=False, labelbottom=True, labelleft=True, pad=6, size=8, width=1, direction='in', colors='black', grid_color='grey', grid_alpha=0.3)
plt.setp(plt.gca().get_xticklabels(), fontproperties=font_prop1)
plt.setp(plt.gca().get_yticklabels(), fontproperties=font_prop1)

# 设置坐标轴标签和标题
plt.xlabel('Actual Values',fontproperties=font_prop)
plt.ylabel('Predicted Values',fontproperties=font_prop)
plt.title('Predicted vs Actual Values\nSupport Vector Regression ',fontproperties=font_prop, fontsize=24)

plt.legend(prop=font_prop)
plt.savefig('SVR.tif',dpi = 150)
plt.show()
