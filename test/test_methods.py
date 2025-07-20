import unittest
import numpy as np
from megatensor.tensor import Tensor

class TestTensorMethods(unittest.TestCase):
  def test_to(self):
    a = Tensor([1, 2, 3])
    b = a.to('cpu')
    self.assertIs(b, a)

  def test_item(self):
    a = Tensor([42])
    self.assertEqual(a.item(), 42)
    b = Tensor(np.array([7.5]))
    self.assertAlmostEqual(b.item(), 7.5)

  def test_numpy(self):
    arr = np.array([1, 2, 3])
    a = Tensor(arr)
    np.testing.assert_allclose(a.numpy(), arr)
    self.assertIsInstance(a.numpy(), np.ndarray)

if __name__ == "__main__": unittest.main()