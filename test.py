import numpy as np
from compile import compile, backward
from ops import *

# Synthetic data: y = 3x + 2
true_W = np.array([[3.0]])
true_B = np.array([[2.0]])

def make_batch(n=64):
    x = np.random.randn(n, 1)
    y = x @ true_W + true_B
    return x, y

ops = [
    Matmul(   a="x",  b="W", out="t0"),
    Broadcast(a="B",  a_shape=(1, 1), out_shape=(64, 1), out="B_bcast"),
    Add(      a="t0", b="B_bcast", out="pred"),
    MSELoss(  pred="pred", target="y_target", out="loss"),
]
input_names = ["x", "W", "B", "y_target"]

extended_ops, ext_inputs, ext_outputs = backward(
    ops, input_names, output_name="loss", params=["W", "B"]
)
train_step = compile(extended_ops, ext_inputs, ext_outputs)

W = np.random.randn(1, 1) * 0.1
B = np.random.randn(1, 1) * 0.1
lr = 0.5

for step in range(200):
    x, y = make_batch(64)
    d_loss = np.array(1.0)
    loss, d_W, d_B = train_step(x, W, B, y, d_loss)

    W = W - lr * d_W
    B = B - lr * d_B

    # if loss <= 0.01:
    #     break
    if step % 20 == 0:
        print(f"step {step:4d}  loss={loss:.5f}  W={W.flatten()}  B={B.flatten()}")

print(f"\nLearned W={W.flatten()} (true=3.0)")
print(f"Learned B={B.flatten()} (true=2.0)")
assert abs(W.item() - 3.0) < 0.05
assert abs(B.item() - 2.0) < 0.05
print("PASS")
