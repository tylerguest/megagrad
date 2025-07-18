import numpy as np

class Tensor:
  def __init__(self, data, *args, _children=(), _op='', label=''):
    if args: data = (data,) + args
    self.data = np.array(data, dtype=float)
    self.grad = np.zeros_like(self.data)
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op 
    self.label = label
    self.shape = self.data.shape
  
  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, _children=(self, other), _op='+')
    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    return out
  
  def __sub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data - other.data, _children=(self, other), _op='-')
    def _backward():
      self.grad += out.grad
      other.grad -= out.grad
    out._backward = _backward
    return out
  
  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, _children=(self, other), _op='*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out
  
  def __truediv__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data / other.data, _children=(self, other), _op='/')
    def _backward():
      self.grad += (1 / other.data, (self, other), '/')
      other.grad -= (self.data / (other.data ** 2)) * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only int/float powers for now"
    out = Tensor(self.data**other, _children=(self,), _op=f'**{other}')
    def _backward(): self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out
  
  def __matmul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data @ other.data, _children=(self, other), _op='@')
    def _backward():
      self.grad += out.grad @ other.data.T
      other.grad += self.data.T @ out.grad
    out._backward = _backward
    return out
  
  def sum(self, axis=None, keepdims=False):
    out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _children=(self,), _op='sum')
    def _backward(): self.grad += np.ones_like(self.data) * out.grad
    out._backward = _backward
    return out
  
  def exp(self):
    out = Tensor(np.exp(self.data), _children=(self,), _op='exp')
    def _backward(): self.grad += np.exp(self.data) * out.grad
    out._backward = _backward
    return out
    
  def log(self):
    out = Tensor(np.log(self.data), _children=(self,), _op='log')
    def _backward(): self.grad += (1 / self.data) * out.grad
    out._backward = _backward
    return out
  
  def item(self): return self.data.item()

  def relu(self):
    out = Tensor(np.maximum(0, self.data), _children=(self,), _op='ReLU')
    def _backward(): self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out
  
  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev: build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = np.ones_like(self.data)
    for v in reversed(topo): v._backward()
  
  def zero_grad(self): self.grad = np.zeros_like(self.data)
  def numpy(self): return self.data
  def __iadd__(self, other): 
    other = other if isinstance(other, Tensor) else Tensor(other)
    self.data += other.data
    return self
  def __isub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    self.data -= other.data
    return self
  def __imul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    self.data *= other.data
    return self
  def __idiv__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    self.data /= other.data
    return self
  def __neg__(self): return self * -1
  def __radd__(self, other): return self + other
  def __sub__(self, other): return self + (-other)
  def __rsub__(self, other): return other + (-self)
  def __rmul__(self, other): return self * other
  def __truediv__(self, other): return self * other**-1
  def __rtruediv__(self, other): return other * self**-1
  def __repr__(self): return f"Tensor(shape={self.shape}, data={self.data}, grad={self.grad})"