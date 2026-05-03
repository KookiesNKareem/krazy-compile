import numpy as np
from compile import compile, backward
from ops import *

np.random.seed(0)

# Synthetic 3-class dataset
CENTERS = np.array([
    [-2.0,  0.0,  0.0,  0.0],
    [ 0.0, -2.0,  0.0,  0.0],
    [ 0.0,  2.0,  2.0,  0.0],
])
N_CLASSES = 3
IN_DIM = 4
HIDDEN = 8
BATCH = 64

def make_batch(n=BATCH):
    labels = np.random.randint(0, N_CLASSES, size=n)
    x = CENTERS[labels] + 0.5 * np.random.randn(n, IN_DIM)
    y_oh = np.eye(N_CLASSES)[labels] # one-hot encoded
    return x, y_oh, labels

# Have to manually write IR for now
ops = [
    Matmul(a="x",  b="W1", out="t0"),
    Broadcast(a="b1", a_shape=(1, HIDDEN), out_shape=(BATCH, HIDDEN), out="b1_bcast"),
    Add(a="t0", b="b1_bcast", out="t1"),
    ReLU(a="t1", out="t2"),
    Matmul(a="t2", b="W2",out="t3"),
    Broadcast(a="b2", a_shape=(1, N_CLASSES), out_shape=(BATCH, N_CLASSES), out="b2_bcast"),
    Add(a="t3", b="b2_bcast",out="pred"),
    MSELoss(pred="pred", target="y_target", out="loss"),
]
input_names = ["x", "W1", "b1", "W2", "b2", "y_target"]

extended_ops, ext_inputs, ext_outputs = backward(
    ops, input_names, output_name="loss",
    params=["W1", "b1", "W2", "b2"],
)

train_step = compile(extended_ops, ext_inputs, ext_outputs)

W1 = np.random.randn(IN_DIM,  HIDDEN) * np.sqrt(2.0 / IN_DIM)
b1 = np.zeros((1, HIDDEN))
W2 = np.random.randn(HIDDEN,  N_CLASSES) * np.sqrt(2.0 / HIDDEN)
b2 = np.zeros((1, N_CLASSES))

lr = 0.05
n_steps = 1000

for step in range(n_steps):
    x, y_oh, _ = make_batch(BATCH)
    d_loss = np.array(1.0)
    loss, d_W1, d_b1, d_W2, d_b2 = train_step(x, W1, b1, W2, b2, y_oh, d_loss)

    W1 = W1 - lr * d_W1
    b1 = b1 - lr * d_b1
    W2 = W2 - lr * d_W2
    b2 = b2 - lr * d_b2

    if step % 100 == 0:
        print(f"step {step:4d}  loss={loss:.5f}")

def predict(x_test):
    t1 = x_test @ W1 + b1
    t2 = np.maximum(t1, 0)
    return t2 @ W2 + b2

# Held-out eval
x_test, y_test_oh, labels_test = make_batch(1000)
preds = predict(x_test)
acc = (preds.argmax(axis=1) == labels_test).mean()
print(f"\nFinal loss: {loss:.5f}")
print(f"Test accuracy: {acc * 100:.1f}%")
assert acc >= 0.95, f"accuracy {acc:.3f} below 0.95"
print("PASS")
