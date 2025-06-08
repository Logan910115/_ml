import numpy as np

# 訓練數據
input_vectors = np.array([
    [1, 1, 1, 1, 1, 1, 0],  # 0
    [0, 1, 1, 0, 0, 0, 0],  # 1
    [1, 1, 0, 1, 1, 0, 1],  # 2
    [1, 1, 1, 1, 0, 0, 1],  # 3
    [0, 1, 1, 0, 0, 1, 1],  # 4
    [1, 0, 1, 1, 0, 1, 1],  # 5
    [1, 0, 1, 1, 1, 1, 1],  # 6
    [1, 1, 1, 0, 0, 0, 0],  # 7
    [1, 1, 1, 1, 1, 1, 1],  # 8
    [1, 1, 1, 1, 0, 1, 1]   # 9
])

target_outputs = np.array([
    [0, 0, 0, 0],  # 0
    [0, 0, 0, 1],  # 1
    [0, 0, 1, 0],  # 2
    [0, 0, 1, 1],  # 3
    [0, 1, 0, 0],  # 4
    [0, 1, 0, 1],  # 5
    [0, 1, 1, 0],  # 6
    [0, 1, 1, 1],  # 7
    [1, 0, 0, 0],  # 8
    [1, 0, 0, 1]   # 9
])

# 初始化權重
np.random.seed(0)  # 為了可重現性
w1 = np.random.rand(7, 4) * 0.1  # 輸入到隱藏層 (7x4)
w2 = np.random.rand(4, 4) * 0.1  # 隱藏層到輸出層 (4x4)

# 激活函數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 前向傳播
def forward(x, w1, w2):
    hidden = sigmoid(np.dot(x, w1))  # 隱藏層
    output = sigmoid(np.dot(hidden, w2))  # 輸出層
    return hidden, output

# 損失函數 (二元交叉熵)
def loss(output, target):
    return -np.mean(target * np.log(output + 1e-15) + (1 - target) * np.log(1 - output + 1e-15))

# 數值梯度 (簡化實現)
def numerical_gradient(f, w, x, t, eps=1e-5):
    grad = np.zeros_like(w)
    it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_value = w[idx]
        w[idx] = old_value + eps
        _, out_plus = forward(x, w1, w2)
        loss_plus = loss(out_plus, t)
        w[idx] = old_value - eps
        _, out_minus = forward(x, w1, w2)
        loss_minus = loss(out_minus, t)
        w[idx] = old_value
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad

# 訓練 (模擬成功)
learning_rate = 0.01
epochs = 1000

for _ in range(epochs):
    total_loss = 0
    for x, t in zip(input_vectors, target_outputs):
        x = x.reshape(1, -1)
        t = t.reshape(1, -1)
        _, output = forward(x, w1, w2)
        total_loss += loss(output, t)
        grad_w1 = numerical_gradient(lambda w: loss(forward(x, w, w2)[1], t), w1, x, t)
        grad_w2 = numerical_gradient(lambda w: loss(forward(x, w1, w)[1], t), w2, x, t)
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    if _ % 100 == 0:
        print(f"Epoch {_}, Loss: {total_loss / 10}")

# 假設訓練成功，預測函數
def predict(segment_input):
    segment_input = np.array(segment_input).reshape(1, -1)
    _, output = forward(segment_input, w1, w2)
    return np.round(output).astype(int).flatten()

# 測試預測
for num, segment in enumerate(input_vectors):
    binary_prediction = predict(segment)
    binary_str = "".join(map(str, binary_prediction))
    print(f"Input: {segment} -> Predicted Binary: {binary_str}  # {num}")