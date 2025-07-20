import numpy as np
import urllib.request
import gzip, os

def unbroadcast(grad, shape):
    while len(grad.shape) > len(shape): grad = grad.sum(axis=0)
    for i, dim in enumerate(shape): 
      if dim == 1: grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
  def _stack_backward(self):
    for i, child in enumerate(self._prev):
      child.grad += self.grad[i]
  
  def __init__(self, data, *args, _children=(), _op='', label='', _lazy=False):
    if args: data = (data,) + args
    if _lazy:
      self._data = None
      self._left = data[0] if isinstance(data, tuple) and len(data) > 0 else data
      self._right = data[1] if isinstance(data, tuple) and len(data) > 1 else None
    else: 
      if isinstance(data, Tensor): data = data.data
      if isinstance(data, (list, tuple)) and all(isinstance(d, Tensor) for d in data): data = [d.data for d in data]
      self._data = np.array(data, dtype=float)
    self.grad = None if self._data is None else np.zeros_like(self._data)
    if _op == 'stack': self._backward = self._stack_backward
    else: self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op 
    self.label = label
    self.shape = None if self._data is None else self._data.shape

  @property
  def data(self):
    if self._data is None: return self.realize()
    return self._data
  
  @data.setter
  def data(self, value): self._data = value
  
  def realize(self):
    if self._data is not None: return self._data
    if self._op == '+':
      left = self._left.realize() if isinstance(self._left, Tensor) else self._left
      right = self._right.realize() if isinstance(self._right, Tensor) else self._right
      self._data = left + right
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == '-':
      left = self._left.realize() if isinstance(self._left, Tensor) else self._left
      right = self._right.realize() if isinstance(self._right, Tensor) else self._right
      self._data = left - right
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == '*':
      left = self._left.realize() if isinstance(self._left, Tensor) else self._left
      right = self._right.realize() if isinstance(self._right, Tensor) else self._right
      self._data = left * right
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == '/':
      left = self._left.realize() if isinstance(self._left, Tensor) else self._left
      right = self._right.realize() if isinstance(self._right, Tensor) else self._right
      self._data = left / right
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op.startswith('**'):
      base = self._left.realize() if hasattr(self, '_left') and isinstance(self._left, Tensor) else self._left
      power = float(self._op[2:]) if self._op.startswith('**') else None
      self._data = base ** power
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == '@':
      left = self._left.realize() if isinstance(self._left, Tensor) else self._left
      right = self._right.realize() if isinstance(self._right, Tensor) else self._right
      self._data = left @ right
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == 'sum':
      base = self._left.realize()
      axis = getattr(self, '_axis', None)
      keepdims = getattr(self, '_keepdims', False)
      self._data = base.sum(axis=axis, keepdims=keepdims)
      self.shape = () if np.isscalar(self._data) else self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == 'mean':
      base = self._left.realize()
      axis = getattr(self, '_axis', None)
      keepdims = getattr(self, '_keepdims', False)
      self._data = base.mean(axis=axis, keepdims=keepdims)
      self.shape = () if np.isscalar(self._data) else self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == 'exp':
      base = self._left.realize() if hasattr(self, '_left') and isinstance(self._left, Tensor) else self._left
      self._data = np.exp(base)
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == 'log':
      base = self._left.realize() if hasattr(self, '_left') and isinstance(self._left, Tensor) else self._left
      self._data = np.log(base)
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == 'flatten':
      base = self._left.realize()
      axis = getattr(self, '_axis', None)
      keepdims = getattr(self, '_keepdims', False)
      self._data = base.flatten(axis=axis, keepdims=keepdims)
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == 'ReLU':
      base = self._left.realize() if hasattr(self, '_left') and isinstance(self._left, Tensor) else self._left
      self._data = np.maximum(0, base)
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == 'astype':
      base = self._left.realize()
      axis = getattr(self, '_axis', None)
      keepdims = getattr(self, '_keepdims', False)
      self._data = base.astype(axis=axis, keepdims=keepdims)
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == 'reshape':
      base = self._left.realize()
      axis = getattr(self, '_axis', None)
      keepdims = getattr(self, '_keepdims', False)
      self._data = base.reshape(axis=axis, keepdims=keepdims)
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    if self._op == 'getitem':
      base = self._left.realize() if hasattr(self, '_left') and isinstance(self._left, Tensor) else self._left
      self._data = np.squeeze(base[self._idx]) if hasattr(self, '_idx') else base
      self.shape = self._data.shape
      self.grad = np.zeros_like(self._data)
      return self._data
    raise NotImplementedError(f"Lazy op {self._op} not implemented in realize()")
  
  
  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor((self, other), _children=(self, other), _op='+', _lazy=True)
    def _backward():
      out_data = out.realize()
      self_data = self.realize()
      other_data = other.realize()
      if self.grad is None: self.grad = np.zeros_like(self_data)
      if other.grad is None: other.grad = np.zeros_like(other_data)
      self.grad += unbroadcast(out.grad, self_data.shape)
      other.grad += unbroadcast(out.grad, other_data.shape)
    out._backward = _backward
    return out
  
  def __sub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor((self, other), _children=(self, other), _op='-', _lazy=True)
    def _backward():
      out_data = out.realize()
      self_data = self.realize()
      other_data = other.realize()
      if self.grad is None: self.grad = np.zeros_like(self_data)
      if other.grad is None: other.grad = np.zeros_like(other_data)
      self.grad += out.grad
      other.grad -= out.grad
    out._backward = _backward
    return out
  
  def __mul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor((self, other), _children=(self, other), _op='*', _lazy=True)
    def _backward():
      self_data = self.realize()
      other_data = other.realize()
      if self.grad is None: self.grad = np.zeros_like(self_data)
      if other.grad is None: other.grad = np.zeros_like(other_data)
      grad_self = other.data * out.grad
      grad_other = self.data * out.grad
      while grad_self.shape != self.grad.shape: grad_self = grad_self.sum(axis=0)
      while grad_other.shape != other.grad.shape: grad_other = grad_other.sum(axis=0)
      self.grad += grad_self
      other.grad += grad_other
    out._backward = _backward
    return out
  
  def __truediv__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor((self, other), _children=(self, other), _op='/', _lazy=True)
    def _backward():
      self_data = self.realize()
      other_data = other.realize()
      if self.grad is None: self.grad = np.zeros_like(self_data)
      if other.grad is None: other.grad = np.zeros_like(other_data)
      self.grad += (1 / other_data) * out.grad
      other.grad -= (self_data / (other_data ** 2)) * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Tensor((self, other), _children=(self,), _op=f'**{other}', _lazy=True)
    def _backward(): 
      self_data = self.realize()
      if self.grad is None: self.grad = np.zeros_like(self_data)
      self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    return out
  
  def __matmul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor((self, other), _children=(self, other), _op='@', _lazy=True)
    def _backward():
      self_data = self.realize()
      other_data = other.realize()
      if self.grad is None: self.grad = np.zeros_like(self_data)
      if other.grad is None: other.grad = np.zeros_like(other_data)
      self.grad += out.grad @ other.data.T
      other.grad += self.data.T @ out.grad
    out._backward = _backward
    return out
  
  def __getitem__(self, idx):
    data = self.realize()[idx]
    data = np.squeeze(data)
    out = Tensor(data, _children=(self,), _op='getitem')
    def _backward():
      grad = np.zeros_like(self.realize())
      grad[idx] = out.grad
      self.grad += grad
    out._backward = _backward
    return out

  def astype(self, dtype):
    out = Tensor(self.realize().astype(dtype), _children=(self,), _op='astype')
    def _backward(): self.grad += out.grad.astype(self.grad.dtype)
    out._backward = _backward
    return out

  def sum(self, axis=None, keepdims=False):
    out = Tensor(self, _children=(self,), _op='sum', _lazy=True)
    out._axis = axis
    out._keepdims = keepdims
    def _backward(): 
      self_data = self.realize()
      if self.grad is None: self.grad = np.zeros_like(self_data)
      self.grad += np.ones_like(self_data) * out.grad
    out._backward = _backward
    return out
  
  def exp(self):
    out = Tensor(np.exp(self.realize()), _children=(self,), _op='exp')
    def _backward(): self.grad += np.exp(self.realize()) * out.grad
    out._backward = _backward
    return out
    
  def log(self):
    out = Tensor(np.log(self.realize()), _children=(self,), _op='log')
    def _backward(): self.grad += (1 / self.realize()) * out.grad
    out._backward = _backward
    return out
  
  def flatten(self, start_dim=0):
    data = self.realize()
    new_shape =(int(np.prod(data.shape[start_dim:])),) if start_dim == 0 else data.shape[:start_dim] + (-1,)
    out = Tensor(data.reshape(new_shape), _children=(self,), _op='flatten')
    def _backward(): self.grad += out.grad.reshape(data.shape)
    out._backward = _backward
    return out
  
  def mean(self, axis=None, keepdims=False):
    data = self.realize()
    out = Tensor(data.mean(axis=axis, keepdims=keepdims), _children=(self,), _op='mean')
    def _backward():
      n = data.size if axis is None else data.shape[axis]
      self.grad += np.ones_like(data) * out.grad / n
    out._backward = _backward
    return out
  
  def argmax(self, axis=None): return int(np.argmax(self.realize(), axis=axis))
  
  def zero_grad(self): self.grad = np.zeros_like(self.realize())

  def sequential(self, layers):
    x = self
    for layer in layers: x = layer(x)
    return x

  def relu(self):
    data = self.realize()
    out = Tensor(np.maximum(0, data), _children=(self,), _op='ReLU')
    def _backward(): self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out
  
  @classmethod
  def from_url(cls, url, gunzip=False):
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    fname = url.split("/")[-1]
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
      print(f"Downloading {fname} to {fpath}...")
      urllib.request.urlretrieve(url, fpath)
    with open(fpath, "rb") as f: data = f.read()
    if gunzip: data = gzip.decompress(data)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cls(arr)
  
  def reshape(self, *shape):
    data = self.realize()
    out = Tensor(data.reshape(*shape), _children=(self,), _op='reshape')
    def _backward(): self.grad += out.grad.reshape(data.shape)
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
    self.grad = np.ones_like(self.realize())
    for v in reversed(topo): v._backward()

  def to(self, device=None): return self

  def item(self): return self.realize().item()
  
  def numpy(self): return self.realize()
  
  def __iadd__(self, other): 
    other = other if isinstance(other, Tensor) else Tensor(other)
    self.realize()
    other.realize()
    self._data += other.data
    return self
  def __isub__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    self.realize()
    other.realize()
    self._data -= other.data
    return self
  def __imul__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    self.realize()
    other.realize()
    self._data *= other.data
    return self
  def __idiv__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    self.realize()
    other.realize()
    self._data /= other.data
    return self
  def __len__(self): return len(self.data)
  def __neg__(self): return self * -1
  def __radd__(self, other): return self + other
  def __sub__(self, other): return self + (-other)
  def __rsub__(self, other): return other + (-self)
  def __rmul__(self, other): return self * other
  def __truediv__(self, other): return self * other**-1
  def __rtruediv__(self, other): return other * self**-1
  def __repr__(self): return f"Tensor(shape={self.shape}, data={self.data}, grad={self.grad})"