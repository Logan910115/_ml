def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"

def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad
    out._backward = _backward

    return out

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward

    return out

def __neg__(self):  # -self
    return self * -1

def __sub__(self, other):
    return self + (-other)

def __truediv__(self, other):
    return self * other**-1

def __pow__(self, other):
    assert isinstance(other, (int, float)), "只能支援常數次方"
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += other * self.data**(other - 1) * out.grad
    out._backward = _backward

    return out

def exp(self):
    import math
    x = self.data
    out = Value(math.exp(x), (self,), 'exp')

    def _backward():
        self.grad += out.data * out.grad
    out._backward = _backward

    return out

def sigmoid(self):
    import math
    x = self.data
    s = 1 / (1 + math.exp(-x))
    out = Value(s, (self,), 'sigmoid')

    def _backward():
        self.grad += s * (1 - s) * out.grad
    out._backward = _backward

    return out

def tanh(self):
    import math
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
        self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

def backward(self):
    # 建立拓撲排序 (先處理子節點)
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
