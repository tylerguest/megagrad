import torch
import numpy as np
from megatensor.tensor import Tensor

class TestAgainstPyTorch:
  def test_complex_expression_vs_pytorch(self):
    x = Tensor(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y
    
    x_pt = torch.Tensor([-4.0]).double()
    x_pt.requires_grad = True
    z_pt = 2 * x_pt + 2 + x_pt
    q_pt = z_pt.relu() + z_pt * x_pt
    h_pt = (z_pt * z_pt).relu()
    y_pt = h_pt + q_pt + q_pt * x_pt
    y_pt.backward()
    
    assert abs(ymg.data - y_pt.data.item()) < 1e-10
    assert abs(xmg.grad - x_pt.grad.item()) < 1e-10
      
  def test_add(self):
    x = Tensor(40)
    y = Tensor(2)
    z = Tensor(43.5)
    a = x + y + z
    a += 10
    a.backward()
    assert a.data == 95.5
    assert a.grad == 1.0
  
  def test_plus_equals(self):
    a = Tensor([10, 10])
    b = Tensor([10, 10])
    c = a + b
    val1 = c.numpy()
    a += b
    val2 = a.numpy()
    np.testing.assert_allclose(val1, val2)

  def test_multiple_operations(self):
    a = Tensor(10, 20)
    b = Tensor(20, 10)
    c = a * b
  
  def test_relu(self):
    a = Tensor(-1)
    b = a.relu()
    assert b.data == 0

  def test_tensor_operations(self):
    x = Tensor(3)
    y = Tensor(10)
    result = x * y + x
    result.backward()
    
    assert result.shape == ()
    assert result.data == 33    
    assert x.grad.item() == 11  
    assert y.grad.item() == 3  

  def test_matmul(self):
    a = Tensor([[1,2], [3,4]])
    b = Tensor([[5,6], [7,8]])
    c = a @ b
    excepted = np.array([[19,22], [43,50]])
    np.testing.assert_allclose(c.data, excepted)
    c.sum().backward()

def test_repr_fix():
  v = Tensor(5.0)
  repr_str = repr(v)
  
  assert "data=5.0" in repr_str
  assert "grad=0" in repr_str
  assert "dtype" not in repr_str