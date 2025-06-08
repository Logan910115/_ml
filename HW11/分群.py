from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

# 生成隨機分群數據
X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# 創建並訓練 K-Means 模型
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 獲取分群結果
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print(f"分群標籤: {labels}")
print(f"聚類中心: {centroids}")