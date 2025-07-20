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

  def test_add_backward(self):
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    c = a + b
    c.backward()
    self.assertTrue(np.allclose(a.grad, [1.0, 1.0]))
    self.assertTrue(np.allclose(b.grad, [1.0, 1.0]))

  def test_chain_rule(self):
    a = Tensor([2.0])
    b = Tensor([3.0])
    c = a * b
    d = c + a
    d.backward()
    self.assertTrue(np.allclose(a.grad, [1.0 + 3.0]))
    self.assertTrue(np.allclose(b.grad, [2.0]))

  def test_zero_grad(self):
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    c = a * b
    c.backward()
    a.zero_grad()
    b.zero_grad()
    self.assertTrue(np.allclose(a.grad, [0.0, 0.0]))
    self.assertTrue(np.allclose(b.grad, [0.0, 0.0]))

  def test_stack_backward(self):
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    stacked = Tensor([a, b], _children=(a, b), _op='stack')
    s = stacked.sum()
    s.backward()
    self.assertTrue(np.allclose(a.grad, [1.0, 1.0]))
    self.assertTrue(np.allclose(b.grad, [1.0, 1.0]))
    
if __name__ == "__main__": unittest.main()
