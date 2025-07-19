from megagrad.tensor import Tensor
import numpy as np

def cross_entropy(logits: Tensor, target: int):
    exps = logits.exp()
    probs = exps / exps.sum()
    return -(probs[target]).log()