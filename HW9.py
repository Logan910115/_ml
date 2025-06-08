import torch
import torch.nn as nn
import numpy as np

# 設置隨機種子以確保結果可重現
torch.manual_seed(0)

# 假設數據
# 生成隨機輸入數據 X (10 個樣本，1 個特徵) 和目標值 y
X = torch.randn(10, 1)
true_w = torch.tensor([[2.0]])  # 真實權重
true_b = torch.tensor([1.0])   # 真實偏置
y = X @ true_w + true_b + torch.randn(10, 1) * 0.1  # 添加一點噪聲

# 定義線性回歸模型
model = nn.Linear(1, 1)  # 輸入特徵 1 個，輸出 1 個

# 定義損失函數 (均方誤差)
criterion = nn.MSELoss()

# 定義優化器 (使用 SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 訓練循環
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向傳播
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # 反向傳播和優化
    optimizer.zero_grad()  # 清除之前的梯度
    loss.backward()        # 計算梯度
    optimizer.step()       # 更新權重
    
    # 每 100 個 epoch 打印損失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 訓練後的參數
print(f'\n訓練完成的權重: {model.weight.item():.4f}')
print(f'訓練完成的偏置: {model.bias.item():.4f}')
print(f'真實權重: {true_w.item():.4f}')
print(f'真實偏置: {true_b.item():.4f}')

# 預測
with torch.no_grad():
    y_pred = model(X)
    print(f'\n預測值: {y_pred.squeeze().detach().numpy()}')
    print(f'真實值: {y.squeeze().detach().numpy()}')