from megagrad.utils import mnist
from megagrad.tensor import Tensor
import numpy as np

# load data
X_train, Y_train, X_test, Y_test = mnist()

# normalize and flatten
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

from megagrad.nn import MLP
model = MLP(28*28, [128, 64, 10])

from megagrad.losses import cross_entropy
lr = 0.01
epochs = 3

for epoch in range(epochs):
    correct = 0
    total_loss = 0
    for i in range(len(X_train)):
        x = X_train[i]
        y = int(Y_train[i].item())
        out = model(Tensor(x))
        loss = cross_entropy(out, y).mean()
        model.zero_grad()
        loss.backward()
        for p in model.parameters(): p.data -= lr * p.grad
        total_loss += loss.item()
        if out.argmax() == y: correct += 1
        if (i+1) % 1000 == 0: print(f"Train step {i+1}/{len(X_train)} loss: {total_loss/(i+1):.4f} acc: {correct/(i+1):.4f}")
        print(f"Epoch {epoch+1}: Train loss {total_loss/len(X_train):.4f}, acc {correct/len(X_train):.4f}")

correct = 0
for i in range(len(X_test)):
    x = X_test[i]
    y = int(Y_test[i].item())
    out = model(Tensor(x))
    if out.argmax() == y: correct += 1
print(f"Test accuracy: {correct/len(X_test):.4f}")
