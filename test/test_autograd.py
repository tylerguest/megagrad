import unittest
import numpy as np
from megatensor.tensor import Tensor

class TestAutograd(unittest.TestCase):
  def test_mul_sum_backward(self):
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    y = (a * b).sum()
    y.backward()
    self.assertTrue(np.allclose(a.grad, [4.0, 5.0, 6.0]))
    self.assertTrue(np.allclose(b.grad, [1.0, 2.0, 3.0]))

if __name__ == "__main__":
  unittest.main()
