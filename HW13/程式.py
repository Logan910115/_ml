from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# 生成隨機數據（包含一些異常點）
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
# 添加一些異常點
outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.vstack([X, outliers])

# 創建並訓練 Isolation Forest 模型
model = IsolationForest(contamination=0.05, random_state=42)  # contamination 設定為 5% 異常點
model.fit(X)

# 預測異常分數和標籤 (-1 表示異常，1 表示正常)
y_pred = model.predict(X)
anomaly_scores = model.score_samples(X)

# 視覺化結果
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', label='數據點')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='異常標籤 (-1 = 異常, 1 = 正常)')
plt.show()

# 顯示部分異常點的索引
anomaly_indices = np.where(y_pred == -1)[0]
print(f"檢測到的異常點索引: {anomaly_indices[:5]}")
print(f"異常點數量: {len(anomaly_indices)}")