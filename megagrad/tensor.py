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
  
  def __init__(self, data, *args, _children=(), _op='', label=''):
    if args: data = (data,) + args
    if isinstance(data, Tensor): data = data.data
    if isinstance(data, (list, tuple)) and all(isinstance(d, Tensor) for d in data): data = [d.data for d in data]
    self.data = np.array(data, dtype=float)
    self.grad = np.zeros_like(self.data)
    if _op == 'stack': self._backward = self._stack_backward
    else: self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op 
    self.label = label
    self.shape = self.data.shape
  
  def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, _children=(self, other), _op='+')
    def _backward():
      self.grad += unbroadcast(out.grad, self.data.shape)
      other.grad += unbroadcast(out.grad, other.data.shape)
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
  
  def __getitem__(self, idx):
    data = self.data[idx]
    data = np.squeeze(data)
    out = Tensor(data, _children=(self,), _op='getitem')
    def _backward():
      grad = np.zeros_like(self.data)
      grad[idx] = out.grad
      self.grad += grad
    out._backward = _backward
    return out

  def astype(self, dtype):
    out = Tensor(self.data.astype(dtype), _children=(self,), _op='astype')
    def _backward(): self.grad += out.grad.astype(self.grad.dtype)
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
  
  def flatten(self, start_dim=0):
    new_shape =(int(np.prod(self.data.shape[start_dim:])),) if start_dim == 0 else self.data.shape[:start_dim] + (-1,)
    out = Tensor(self.data.reshape(new_shape), _children=(self,), _op='flatten')
    def _backward(): self.grad += out.grad.reshape(self.data.shape)
    out._backward = _backward
    return out
  
  def argmax(self, axis=None): return int(np.argmax(self.data, axis=axis))

  def mean(self, axis=None, keepdims=False):
    out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), _children=(self,), _op='mean')
    def _backward():
      n = self.data.size if axis is None else self.data.shape[axis]
      self.grad += np.ones_like(self.data) * out.grad / n
    out._backward = _backward
    return out

  def item(self): return self.data.item()

  def sequential(self, layers):
    x = self
    for layer in layers: x = layer(x)
    return x

  def relu(self):
    out = Tensor(np.maximum(0, self.data), _children=(self,), _op='ReLU')
    def _backward(): self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out
  
  @classmethod
  def from_url(cls, url, gunzip=False):
    # Download to data/ directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)
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
    out = Tensor(self.data.reshape(*shape), _children=(self,), _op='reshape')
    def _backward(): self.grad += out.grad.reshape(self.data.shape)
    out._backward = _backward
    return out
  
  def to(self, device=None): return self
  
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
  def __len__(self): return len(self.data)
  def __neg__(self): return self * -1
  def __radd__(self, other): return self + other
  def __sub__(self, other): return self + (-other)
  def __rsub__(self, other): return other + (-self)
  def __rmul__(self, other): return self * other
  def __truediv__(self, other): return self * other**-1
  def __rtruediv__(self, other): return other * self**-1
  def __repr__(self): return f"Tensor(shape={self.shape}, data={self.data}, grad={self.grad})"