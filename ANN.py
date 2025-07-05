import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import numpy as np
import matplotlib.pyplot as plt
import shap

# 读取数据
df = pd.read_excel(r'D:\projectPython\QA\t1\k_lg_sol_shap.xlsx')
x = df.drop(['lnx'], axis=1).values
y = df['lnx'].values

# 应用 Robust Scaling
scaler = RobustScaler()
x = scaler.fit_transform(x)

# 划分训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# 定义带 Dropout 的神经网络模型
class AdvancedNN(nn.Module):
    def __init__(self, input_size):
        super(AdvancedNN, self).__init__()
        self.hidden1 = nn.Linear(input_size, 95)
        self.dropout1 = nn.Dropout(0.3)  # Dropout 层
        self.hidden2 = nn.Linear(95, 52)
        self.dropout2 = nn.Dropout(0.3)  # Dropout 层
        self.output = nn.Linear(52, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = self.dropout1(x)  # 应用 Dropout
        x = torch.relu(self.hidden2(x))
        x = self.dropout2(x)  # 应用 Dropout
        return self.output(x)

# 初始化模型
model = AdvancedNN(input_size=x.shape[1])

# 定义超参数
learning_rate = 0.0035837257746607065
num_epochs = 5000

# 定义损失函数和优化器（添加 L2 正则化）
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)  # L2 正则化

# 开始训练
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 验证
    val_outputs = model(x_val)
    val_loss = criterion(val_outputs, y_val)

    # 打印每 100 个 epoch 的损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# 使用模型进行预测
predicted_train = model(x_train)
predicted_val = model(x_val)

# 转换为 NumPy 数组
predicted_train_np = predicted_train.detach().numpy().flatten()
predicted_val_np = predicted_val.detach().numpy().flatten()

# 计算相关系数
correlation_train = np.corrcoef(predicted_train_np, y_train.numpy().flatten())[0, 1]
correlation_val = np.corrcoef(predicted_val_np, y_val.numpy().flatten())[0, 1]
print(f'Train Correlation Coefficient: {correlation_train:.4f}')
print(f'Validation Correlation Coefficient: {correlation_val:.4f}')

# 计算 R²
def calculate_r_squared(y_true, y_pred):
    mean_y = np.mean(y_true)
    sst = np.sum((y_true - mean_y) ** 2)
    sse = np.sum((y_pred - y_true) ** 2)
    return 1 - (sse / sst)

r_squared_train = calculate_r_squared(y_train.numpy(), predicted_train_np)
r_squared_val = calculate_r_squared(y_val.numpy(), predicted_val_np)
print(f'Train Coefficient of Determination (R²): {r_squared_train:.4f}')
print(f'Validation Coefficient of Determination (R²): {r_squared_val:.4f}')

# 绘图: 训练集和验证集的实际值与预测值对比
plt.figure(figsize=(14, 6))

# 训练集图
plt.subplot(1, 2, 1)
plt.scatter(y_train.numpy(), predicted_train_np, label='Predicted', s=100, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--', lw=2, color='orange', label='Actual Values')
plt.title('Training Set: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid()

# 验证集图
plt.subplot(1, 2, 2)
plt.scatter(y_val.numpy(), predicted_val_np, label='Predicted', s=100, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--', lw=2, color='orange', label='Actual Values')
plt.title('Validation Set: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('training_validation_comparison.png', dpi=600, transparent=True)
plt.show()

# SHAP 解释
def model_wrapper(x):
    x = torch.tensor(x, dtype=torch.float32)
    return model(x).detach().numpy()

# 创建 SHAP 解释器
explainer = shap.Explainer(model_wrapper, x_train.numpy())
shap_values = explainer(x_val.numpy())

# 绘制 SHAP 总结图
shap.summary_plot(shap_values, x_val.numpy(), feature_names=df.drop(['lnx'], axis=1).columns)