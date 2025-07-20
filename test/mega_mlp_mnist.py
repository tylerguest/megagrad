from megatensor.utils import mnist
from megatensor.tensor import Tensor
from megatensor.nn import MLP
from megatensor.losses import cross_entropy
import numpy as np

X_train, Y_train, X_test, Y_test = mnist()
X_train = (X_train / 255.0)[:10]
Y_train = Y_train[:10]
X_test = (X_test / 255.0)[:20]
Y_test = Y_test[:20]

model = MLP(28*28, [128, 64, 10])
lr = 0.01
epochs = 1

for epoch in range(epochs):
    print(f"\n------ Epoch {epoch+1} start ------")
    correct = 0
    total_loss = 0
    for i in range(len(X_train)):
        x = X_train[i].reshape(-1)
        y = int(Y_train[i].item())
        out = model(Tensor(x))
        loss = cross_entropy(out, y)
        model.zero_grad()
        loss.backward()
        for p in model.parameters(): p.data -= lr * p.grad
        total_loss += loss.item()
        if out.argmax() == y: correct += 1
        print(f"Training step {i+1}/{len(X_train)}: loss={loss.item():.4f}, acc={correct/(i+1):.4f}")
    print(f"------ Epoch {epoch+1} end: Train loss {total_loss/len(X_train):.4f}, acc {correct/len(X_train):.4f} ------\n")

print("\n------ Starting test loop ------")
correct = 0
for i in range(len(X_test)):
    x = X_test[i].reshape(-1)
    y = int(Y_test[i].item())
    out = model(Tensor(x))
    if out.argmax() == y: correct += 1
    print(f"Test step {i+1}/{len(X_test)}: current acc={correct/(i+1):.4f}")
print(f"------ Test accuracy: {correct/len(X_test):.4f} ------\n")
