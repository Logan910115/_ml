from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np

# 生成隨機分類數據
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 創建並訓練模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測和評估
accuracy = model.score(X_test, y_test)
print(f"分類準確率: {accuracy:.2f}")

# 預測一個樣本
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)
print(f"樣本 {sample} 的預測類別: {prediction[0]}")