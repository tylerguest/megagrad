import unittest
import numpy as np
from megatensor.tensor import Tensor

class TestTensorRealize(unittest.TestCase):
  def test_realize_add(self):
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    self.assertIsNone(c._data)
    realized = c.realize()
    np.testing.assert_allclose(realized, [5, 7, 9])
    self.assertIsNotNone(c._data)

  def test_realize_mull(self):
    a = Tensor([2, 3, 4])
    b = Tensor([5, 6, 7])
    c = a * b
    self.assertIsNone(c._data)
    realized = c.realize()
    np.testing.assert_allclose(realized, [10, 18, 28])

  def test_realize_pow(self):
    a = Tensor([2, 3, 4])
    c = a**2
    self.assertIsNone(c._data)
    realized = c.realize()
    np.testing.assert_allclose(realized, [4, 9, 16])

  def test_realize_matmul(self):
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[2, 0], [1, 2]])
    c = a @ b
    self.assertIsNone(c._data)
    realized = c.realize()
    np.testing.assert_allclose(realized, [[4, 4], [10, 8]])

  def test_realize_sum(self):
    a = Tensor([1, 2, 3])
    s = a.sum()
    self.assertIsNone(s._data)
    realized = s.realize()
    self.assertEqual(realized, 6)

  def test_realize_mean(self):
    a = Tensor([2, 4, 6])
    m = a.mean()
    self.assertIsNone(m._data)
    realized = m.realize()
    self.assertEqual(realized, 4)

  def test_realize_exp_log(self):
    a = Tensor([1, 2, 3])
    b = a.exp()
    c = b.log()
    np.testing.assert_allclose(c.realize(), a.realize(), rtol=1e-5)

  def test_realize_flatten(self):
    a = Tensor([[1, 2], [3, 4]])
    b = a.flatten()
    np.testing.assert_allclose(b.realize(), [1, 2, 3, 4])

  def test_realize_getitem(self):
    a = Tensor([1, 2, 3])
    b = a[1]
    self.assertEqual(b.realize(), 2)

if __name__ == "__main__": unittest.main()