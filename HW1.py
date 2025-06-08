# 參數設定
x, y, z = 0.0, 0.0, 0.0   # 初始值
learning_rate = 0.1
epsilon = 1e-6            # 收斂條件
max_iterations = 1000

for i in range(max_iterations):
    # 計算梯度
    grad_x = 2 * x - 2
    grad_y = 2 * y - 4
    grad_z = 2 * z - 6
    
    # 更新變數
    x_new = x - learning_rate * grad_x
    y_new = y - learning_rate * grad_y
    z_new = z - learning_rate * grad_z
    
    # 判斷是否收斂
    if abs(x_new - x) < epsilon and abs(y_new - y) < epsilon and abs(z_new - z) < epsilon:
        break
    
    x, y, z = x_new, y_new, z_new

# 計算最低點對應的函數值
f_min = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

# 輸出結果
print(f"最低點位置: x = {x}, y = {y}, z = {z}")
print(f"最低點函數值: f = {f_min}")