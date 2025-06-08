import torch

# 定義函數 f = x^2 + y^2 + z^2 - 2x - 4y - 6z + 8
def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

# 初始值，需設置 requires_grad=True 以跟踪梯度
x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)
z = torch.tensor(0.0, requires_grad=True)

# 學習率和訓練週期
learning_rate = 0.1
epochs = 100

# 梯度下降
for epoch in range(epochs):
    # 前向傳播
    fx = f(x, y, z)
    
    # 計算梯度
    fx.backward()
    
    # 更新參數（使用梯度下降）
    with torch.no_grad():  # 禁用梯度跟踪以更新參數
        x -= learning_rate * x.grad
        y -= learning_rate * y.grad
        z -= learning_rate * z.grad
    
    # 清除梯度
    x.grad.zero_()
    y.grad.zero_()
    z.grad.zero_()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, f(x,y,z) = {fx.item()}, x = {x.item()}, y = {y.item()}, z = {z.item()}")

# 訓練後的結果
print(f"\n訓練完成！最終值：")
print(f"x = {x.item()}, y = {y.item()}, z = {z.item()}")
print(f"f(x,y,z) = {f(x, y, z).item()}")