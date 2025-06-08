from micrograd.engine import Value

# 定義函數 f = x^2 + y^2 + z^2 - 2x - 4y - 6z + 8
def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

# 初始值
x = Value(0.0)
y = Value(0.0)
z = Value(0.0)

# 學習率和訓練週期
learning_rate = 0.1
epochs = 100

# 梯度下降
for epoch in range(epochs):
    # 前向傳播
    fx = f(x, y, z)
    
    # 計算梯度
    fx.backward()
    
    # 更新參數
    x.data -= learning_rate * x.grad
    y.data -= learning_rate * y.grad
    z.data -= learning_rate * z.grad
    
    # 清除梯度（準備下一次反向傳播）
    x.grad = 0.0
    y.grad = 0.0
    z.grad = 0.0
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, f(x,y,z) = {fx.data}, x = {x.data}, y = {y.data}, z = {z.data}")

# 訓練後的結果
print(f"\n訓練完成！最終值：")
print(f"x = {x.data}, y = {y.data}, z = {z.data}")
print(f"f(x,y,z) = {f(x, y, z).data}")