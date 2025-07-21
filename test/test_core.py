import unittest
import numpy as np
from megatensor.tensor import Tensor

class TestCore(unittest.TestCase):
  def test_tensor_init_from_list(self):
    t = Tensor([1, 2, 3])
    np.testing.assert_allclose(t.data, [1, 2, 3])
    self.assertEqual(t.shape, (3,))
    self.assertTrue(np.allclose(t.grad, [0, 0, 0]))

  def test_tensor_init_from_tensor(self):
    t1 = Tensor([1, 2, 3])
    t2 = Tensor(t1)
    np.testing.assert_allclose(t2.data, [1, 2, 3])

  def test_property_and_setter(self):
    t = Tensor([1, 2, 3])
    t.data = np.array([4, 5, 6])
    np.testing.assert_allclose(t.data, [4, 5, 6])

if __name__ == "__main__": unittest.main()