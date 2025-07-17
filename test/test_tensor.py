"""Integration tests that match the testbook.ipynb examples"""
import torch
from megagrad.engine import Value
from megagrad.tensor import Tensor

class TestAgainstPyTorch:
    def test_complex_expression_vs_pytorch(self):
        """Test the exact example from testbook.ipynb"""
        print("\n=== Testing Complex Expression vs PyTorch ===")
        
        # Nanograd computation
        print("Running nanograd computation...")
        x = Value(-4.0)
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        print(f"Before backward: x.data={x.data}, y.data={y.data}")
        y.backward()
        xmg, ymg = x, y
        print(f"After backward: x.grad={xmg.grad}, y.data={ymg.data}")
        
        # PyTorch computation
        print("Running PyTorch computation...")
        x_pt = torch.Tensor([-4.0]).double()
        x_pt.requires_grad = True
        z_pt = 2 * x_pt + 2 + x_pt
        q_pt = z_pt.relu() + z_pt * x_pt
        h_pt = (z_pt * z_pt).relu()
        y_pt = h_pt + q_pt + q_pt * x_pt
        print(f"Before backward: x_pt.data={x_pt.data.item()}, y_pt.data={y_pt.data.item()}")
        y_pt.backward()
        print(f"After backward: x_pt.grad={x_pt.grad.item()}, y_pt.data={y_pt.data.item()}")
        
        # Compare results
        print("Comparing results...")
        print(f"Nanograd y: {ymg.data}")
        print(f"PyTorch y:  {y_pt.data.item()}")
        print(f"Difference: {abs(ymg.data - y_pt.data.item())}")
        print(f"Nanograd x.grad: {xmg.grad}")
        print(f"PyTorch x.grad:  {x_pt.grad.item()}")
        print(f"Grad difference: {abs(xmg.grad - x_pt.grad.item())}")
        
        assert abs(ymg.data - y_pt.data.item()) < 1e-10
        assert abs(xmg.grad - x_pt.grad.item()) < 1e-10
        print("✓ Complex expression test PASSED!")
        
    def test_simple_arithmetic_operations(self):
        """Test basic operations from testbook.ipynb"""
        print("\n=== Testing Simple Arithmetic Operations ===")
        
        x = Value(40)
        y = Value(2)
        print(f"Initial values: x={x.data}, y={y.data}")
        
        z = x * y
        print(f"After x * y: z={z.data}")
        
        z += 10
        print(f"After z += 10: z={z.data}")
        
        z.backward()
        print(f"After backward:")
        print(f"  z.data = {z.data}")
        print(f"  x.grad = {x.grad} (should be {y.data})")
        print(f"  y.grad = {y.grad} (should be {x.data})")
        
        assert z.data == 90
        assert x.grad == 2  # gradient of x in x*y is y
        assert y.grad == 40  # gradient of y in x*y is x
        print("✓ Simple arithmetic test PASSED!")
        
    def test_tensor_operations(self):
        """Test tensor operations from testbook.ipynb"""
        print("\n=== Testing Tensor Operations ===")
        
        x = Tensor(3)
        y = Tensor(10)
        print(f"Initial tensors:")
        print(f"  x = {x}")
        print(f"  y = {y}")
        
        result = x * y + x
        print(f"After x * y + x: result = {result}")
        print(f"Expected: 3 * 10 + 3 = 33")
        
        result.backward()
        print(f"After backward:")
        print(f"  result.shape = {result.shape}")
        print(f"  result.data[0].data = {result.data[0].data}")
        print(f"  x.data[0].grad = {x.data[0].grad} (should be 11: dy/dx = 10 + 1)")
        print(f"  y.data[0].grad = {y.data[0].grad} (should be 3: dy/dy = x)")
        
        assert result.shape == ()
        assert result.data[0].data == 33  # 3*10 + 3 = 33
        
        # Check gradients
        assert x.data[0].grad == 11  # gradient is 10 + 1 = 11
        assert y.data[0].grad == 3   # gradient is 3
        print("✓ Tensor operations test PASSED!")

def test_repr_fix():
    """Test that the __repr__ bug is fixed"""
    print("\n=== Testing __repr__ Fix ===")
    
    v = Value(5.0)
    print(f"Creating Value(5.0)...")
    
    # This should not raise AttributeError about 'dtype'
    repr_str = repr(v)
    print(f"repr(v) = {repr_str}")
    
    print("Checking repr contents...")
    print(f"  Contains 'data=5.0': {'data=5.0' in repr_str}")
    print(f"  Contains 'grad=0': {'grad=0' in repr_str}")
    print(f"  Contains 'dtype': {'dtype' in repr_str} (should be False)")
    
    assert "data=5.0" in repr_str
    assert "grad=0" in repr_str
    # Should not contain 'dtype' since that attribute doesn't exist
    assert "dtype" not in repr_str
    print("✓ __repr__ fix test PASSED!")