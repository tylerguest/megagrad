import unittest
import numpy as np
from megatensor.tensor import Tensor

class TestTensorOverloads(unittest.TestCase):
  def test_add(self):
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    np.testing.assert_allclose(c.data, [5, 7, 9])
    c.backward()
    np.testing.assert_allclose(a.grad, [1, 1, 1])
    np.testing.assert_allclose(b.grad, [1, 1, 1])

  def test_sub(self):
    a = Tensor([5, 7, 9])
    b = Tensor([1, 2, 3])
    c = a - b
    np.testing.assert_allclose(c.data, [4, 5, 6])
    c.backward()
    np.testing.assert_allclose(a.grad, [1, 1, 1])
    np.testing.assert_allclose(b.grad, [-1, -1, -1])
  
  def test_mul(self):
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a * b
    np.testing.assert_allclose(c.data, [4, 10, 18])
    c.backward()
    np.testing.assert_allclose(a.grad, b.data)
    np.testing.assert_allclose(b.grad, a.data)

  def test_truediv(self):
    a = Tensor([4.0, 9.0, 16.0])
    b = Tensor([2.0, 3.0, 4.0])
    c = a / b
    np.testing.assert_allclose(c.data, [2, 3, 4])
    c.backward()
    np.testing.assert_allclose(a.grad, [0.5, 1/3., 0.25])
    np.testing.assert_allclose(b.grad, [-1.0, -1.0, -1.0])

  def test_pow(self):
    a = Tensor([2.0, 3.0, 4.0])
    c = a**2
    np.testing.assert_allclose(c.data, [4, 9, 16])
    c.backward()
    np.testing.assert_allclose(a.grad, [4.0, 6.0, 8.0])

  def test_matmul(self):
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[2, 0], [1, 2]])
    c = a @ b
    np.testing.assert_allclose(c.data, [[4, 4], [10, 8]])
    c.sum().backward()
    self.assertEqual(a.grad.shape, (2, 2))
    self.assertEqual(b.grad.shape, (2, 2))

  def test_getitem(self):
    a = Tensor([1, 2, 3])
    b = a[1]
    self.assertEqual(b.data, 2)
    b.backward()
    np.testing.assert_allclose(a.grad, [0, 1, 0])

  def test_inplace_ops(self):
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    a += b
    np.testing.assert_allclose(a.data, [5, 7, 9])
    a -= b
    np.testing.assert_allclose(a.data, [1, 2, 3])
    a *= b
    np.testing.assert_allclose(a.data, [4, 10, 18])
    a /= b
    np.testing.assert_allclose(a.data, [1, 2, 3])

  def test_neg_rdd_rsub_rmul_rtruediv(self):
    a = Tensor([1, 2, 3])
    b = -a
    np.testing.assert_allclose(b.data, [-1, -2, -3])
    c = 1 + a
    np.testing.assert_allclose(c.data, [2, 3, 4])
    d = 5 - a
    np.testing.assert_allclose(d.data, [4, 3, 2])
    e = 2 * a
    np.testing.assert_allclose(e.data, [2, 4, 6])
    f = 6 / a
    np.testing.assert_allclose(f.data, [6, 3, 2])

  def test_len_repr(self):
    a = Tensor([1, 2, 3])
    self.assertEqual(len(a), 3)
    self.assertIn("Tensor(shape=", repr(a))

if __name__ == "__main__": unittest.main()