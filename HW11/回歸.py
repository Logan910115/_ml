from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 生成隨機回歸數據
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1  # y = 2x + 1 + 噪聲

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建並訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測和評估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"均方誤差: {mse:.4f}")

# 訓練後的參數
print(f"權重: {model.coef_[0][0]:.4f}")
print(f"偏置: {model.intercept_[0]:.4f}")