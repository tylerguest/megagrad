import numpy as np
import gzip
import os
from megagrad.tensor import Tensor

def load_mnist(path=None, kind="train"):
  """Load MNIST data from `data/` directory by default"""
  if path is None:
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    path = os.path.abspath(data_dir)
  labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
  images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
  with gzip.open(labels_path, 'rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
  with gzip.open(images_path, 'rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28*28)
  return images, labels 

def mnist(device=None, fashion=False):
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/" if fashion else "https://storage.googleapis.com/cvdf-datasets/mnist/"
    def _mnist(file): return Tensor.from_url(base_url+file, gunzip=True)
    return _mnist("train-images-idx3-ubyte.gz")[0x10:].reshape(-1,1,28,28).to(device), \
           _mnist("train-labels-idx1-ubyte.gz")[8:].to(device), \
           _mnist("t10k-images-idx3-ubyte.gz")[0x10:].reshape(-1,1,28,28).to(device), \
           _mnist("t10k-labels-idx1-ubyte.gz")[8:].to(device)