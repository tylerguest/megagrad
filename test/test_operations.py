import unittest
import numpy as np
from megatensor.tensor import Tensor

class TestTensorOperations(unittest.TestCase):
  def test_add(self):
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    np.testing.assert_allclose(c.data, [5, 7, 9])

  def test_sub(self):
    a = Tensor([5, 7, 9])
    b = Tensor([1, 2, 3])
    c = a - b
    np.testing.assert_allclose(c.data, [4, 5, 6])

  def test_mul(self):
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a * b
    np.testing.assert_allclose(c.data, [4, 10, 18])

  def test_div(self):
    a = Tensor([4, 9, 16])
    b = Tensor([2, 3, 4])
    c = a / b
    np.testing.assert_allclose(c.data, [2, 3, 4])

  def test_pow(self):
    a = Tensor([2, 3, 4])
    c = a**2
    np.testing.assert_allclose(c.data, [4, 9, 16])

  def test_matmul(self):
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[2, 0], [1, 2]])
    c = a @ b
    np.testing.assert_allclose(c.data, [[4, 4], [10, 8]])

  def test_sum(self):
    a = Tensor([1, 2, 3])
    s = a.sum()
    self.assertEqual(s.data, 6)

  def test_mean(self):
    a = Tensor([2, 4, 6])
    m = a.mean()
    self.assertEqual(m.data, 4)

  def test_reshape(self):
    a = Tensor([1, 2, 3, 4])
    b = a.reshape(2, 2)
    np.testing.assert_allclose(b.data, [[1, 2], [3, 4]])

  def test_flatten(self):
    a = Tensor([1, 2], [3, 4])
    b = a.flatten()
    np.testing.assert_allclose(b.data, [1, 2, 3, 4])

  def test_relu(self):
    a = Tensor([-1, 0, 2])
    b = a.relu()
    np.testing.assert_allclose(b.data, [0, 0, 2])

  def test_exp_log(self):
    a = Tensor([1, 2, 3])
    b = a.exp()
    c = b.log()
    np.testing.assert_allclose(c.data, a.data, rtol=1e-5)

  def test_backward_add(self):
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    c.backward()
    np.testing.assert_allclose(a.grad, [1, 1, 1])
    np.testing.assert_allclose(b.grad, [1, 1, 1])

  def test_backward_mul(self):
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a * b
    c.backward()
    np.testing.assert_allclose(a.grad, b.data)
    np.testing.assert_allclose(b.grad, a.data)

  def test_broadcasting(self):
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = Tensor([1, 2, 3])
    c = a + b
    np.testing.assert_allclose(c.data, [[2, 4, 6], [5, 7, 9]])

  def test_scalar_ops(self):
    a = Tensor([1, 2, 3])
    b = a + 1
    np.testing.assert_allclose(b.data, [2, 3, 4])
    c = 2 * a
    np.testing.assert_allclose(c.data, [2, 4, 6])
    
if __name__ == "__main__": unittest.main()