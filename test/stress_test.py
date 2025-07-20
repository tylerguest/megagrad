import numpy as np
from megatensor.tensor import Tensor

print("=== Megatensor Stress Test ===")

# Random input tensors
np.random.seed(42)
a_np = np.random.randn(3, 3)
b_np = np.random.randn(3, 3)
a = Tensor(a_np)
b = Tensor(b_np)

print("a:\n", a.data)
print("b:\n", b.data)

# Arithmetic
c = a + b
print("a + b:\n", c.data)
assert np.allclose(c.data, a_np + b_np)

d = a * b
print("a * b:\n", d.data)
assert np.allclose(d.data, a_np * b_np)

e = a - b
print("a - b:\n", e.data)
assert np.allclose(e.data, a_np - b_np)

f = a / (b + 1.5)  # avoid div by zero
print("a / (b + 1.5):\n", f.data)
assert np.allclose(f.data, a_np / (b_np + 1.5))

# Matrix multiplication
g = a @ b
print("a @ b:\n", g.data)
assert np.allclose(g.data, a_np @ b_np)

# Power and mean
h = (a ** 2).mean()
print("mean(a ** 2):", h.data)
assert np.allclose(h.data, (a_np ** 2).mean())

# Chained operations
y = ((a * b).sum() + (a - b).mean()) * 2
print("y:", y.data)
assert np.allclose(y.data, 2 * ((a_np * b_np).sum() + (a_np - b_np).mean()))
y = y * 2
print("y * 2:", y.data)
assert np.allclose(y.data, 4 * ((a_np * b_np).sum() + (a_np - b_np).mean()))

# Backward test
z = (a * b).sum()
z.backward()
print("a.grad after (a*b).sum().backward():\n", a.grad)
print("b.grad after (a*b).sum().backward():\n", b.grad)
assert np.allclose(a.grad, b_np)
assert np.allclose(b.grad, a_np)

# Reshape, flatten, relu
flat = a.flatten()
print("a.flatten():", flat.data)
assert np.allclose(flat.data, a_np.flatten())

reshaped = flat.reshape(3, 3)
print("flat.reshape(3, 3):\n", reshaped.data)
assert np.allclose(reshaped.data, a_np)

relu = a.relu()
print("a.relu():\n", relu.data)
assert np.allclose(relu.data, np.maximum(0, a_np))

print("All stress tests passed!")