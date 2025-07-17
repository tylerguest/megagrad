from megagrad.engine import Value

class Tensor:

  def __init__(self, data, shape=None):
    if isinstance(data, (int, float)):
      self.data = [Value(data)]
      self.shape = ()
    elif isinstance(data, list):
      self.data = self._flatten_and_convert(data)
      self.shape = self._infer_shape(data) if shape is None else shape
    else: raise ValueError("Data must be a number or list")
    
  def _flatten_and_convert(self, data):
    """Recursively flatten nested lists and convert to Value objects"""
    if isinstance(data, (int, float)): return [Value(data)]
    elif isinstance(data, list):
      result = []
      for item in data: result.extend(self._flatten_and_convert(item))
      return result
    else: raise ValueError("Invalid data type in tensor")

  def _infer_shape(self, data):
    """Infer shape from nested list structure"""
    if isinstance(data, (int, float)): return ()
    elif isinstance(data, list):
      if not data: return (0,)
      shape = [len(data)]
      if isinstance(data[0], list):
        shape.extend(self._infer_shape(data[0]))
        return tuple(shape)
      
  def __getitem__(self, idx):
    """Basic indexing for 1D tensors"""
    if self.shape == ():
      if idx != 0: raise IndexError("Scalar tensor only has index 0")
      return self.data[0]
    return self.data[idx]
  
  def __add__(self, other):
    """Element-wise addition"""
    if isinstance(other, Tensor):
      if self.shape != other.shape: raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
      result_data = [a + b for a, b in zip(self.data, other.data)]
    else: result_data = [val + other for val in self.data]
    result = Tensor.__new__(Tensor)
    result.data = result_data
    result.shape = self.shape
    return result 
  
  def __mul__(self, other):
    """Element-wise multiplication"""
    if isinstance(other, Tensor):
      if self.shape != other.shape: raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
      result_data = [a * b for a, b in zip(self.data, other.data)]
    else: result_data = [val * other for val in self.data]
    result = Tensor.__new__(Tensor)
    result.data = result_data
    result.shape = self.shape
    return result
  
  def sum(self):
    """Sum all elements to a scalar Value"""
    result = Value(0)
    for val in self.data: result = result + val
    return result
  
  def backward(self):
    """Backward pass - only works for scalar tensors"""
    if self.shape != (): raise ValueError("Can only call backward() on scalar tensors")
    self.data[0].backward()

  @property
  def grad(self):
    """Get gradients as a new tensor with same shape"""
    grad_data = [val.grad for val in self.data]
    result = Tensor.__new__(Tensor)
    result.data = [Value(g) for g in grad_data]
    result.shape = self.shape
    return result
  
  def zero_grad(self):
    """Zero out all gradients"""
    for val in self.data: val.grad = 0

  def __repr__(self):
    if self.shape == (): return f"Tensor({self.data[0].data}, shape={self.shape})"
    else:
      data_vals = [val.data for val in self.data]
      return f"Tensor({data_vals}, shape={self.shape})"