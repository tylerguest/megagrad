from megagrad.tensor import Tensor
import numpy as np

def cross_entropy(logits: Tensor, target: int):
    max_logit = logits.data.max()
    log_sum_exp = (logits - max_logit).exp().sum().log() + max_logit
    log_probs = logits - log_sum_exp
    loss = -log_probs[target]
    return loss