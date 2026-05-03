import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from compile import compile, backward
from compile_cpp import compile_cpp
from ops import *
from sklearn.datasets import fetch_openml
from skimage.transform import resize
import time

np.random.seed(0)

IN_DIM = 64
HIDDEN = 1024
N_CLASSES = 10
BATCH = 256

N=100000
X = np.random.randn(N, 64)
# X_88 = np.array([resize(img.reshape(28, 28), (8, 8), anti_aliasing=True).flatten() for img in X])
y = np.random.randint(0, N_CLASSES, size=N)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

X_train_bin = np.where(X_train > 0.5, 1, -1).astype(np.int8)
X_test_bin  = np.where(X_test  > 0.5, 1, -1).astype(np.int8)
print(f"Train: {X_train_bin.shape}, Test: {X_test_bin.shape}")

ops = [
    Sign(a="W1_latent", out="W1"),
    BinaryMatmul(a="x",  b="W1", out="t0"),
    Broadcast(a="b1", a_shape=(1, HIDDEN), out_shape=(BATCH, HIDDEN), out="b1_bcast"),
    Add(a="t0", b="b1_bcast", out="t1"),
    Sign(a="t1", out="t2"),
    Sign(a="W2_latent", out="W2"),
    BinaryMatmul(a="t2", b="W2", out="t3"),
    Broadcast(a="b2", a_shape=(1, N_CLASSES), out_shape=(BATCH, N_CLASSES), out="b2_bcast"),
    Add(a="t3", b="b2_bcast", out="pred"),
    SoftmaxCrossEntropy(pred="pred", target="y_target", out="loss")]
input_names = ["x", "W1_latent", "b1", "W2_latent", "b2", "y_target"]

extended_ops, ext_inputs, ext_outputs = backward(ops, input_names, output_name="loss", params=["W1_latent", "b1", "W2_latent", "b2"])
input_specs = {
    "x": np.zeros((BATCH, IN_DIM)),
    "W1_latent": np.zeros((IN_DIM, HIDDEN)),
    "b1": np.zeros((1, HIDDEN)),
    "W2_latent": np.zeros((HIDDEN, N_CLASSES)),
    "b2": np.zeros((1, N_CLASSES)),
    "y_target": np.zeros((BATCH, N_CLASSES)),
    "d_loss": np.zeros(())}

# benchmark compile time of python vs cpp backend
train_step_py = compile(extended_ops, ext_inputs, ext_outputs)
train_step_cpp = compile_cpp(extended_ops, ext_inputs, ext_outputs, input_specs)

start = time.time()
train_step_py(np.zeros((BATCH, IN_DIM)), np.zeros((IN_DIM, HIDDEN)), np.zeros((1, HIDDEN)), np.zeros((HIDDEN, N_CLASSES)), np.zeros((1, N_CLASSES)), np.zeros((BATCH, N_CLASSES)), np.array(1.0))
py_time = time.time() - start

start = time.time()
train_step_cpp(np.zeros((BATCH, IN_DIM)), np.zeros((IN_DIM, HIDDEN)), np.zeros((1, HIDDEN)), np.zeros((HIDDEN, N_CLASSES)), np.zeros((1, N_CLASSES)), np.zeros((BATCH, N_CLASSES)), np.array(1.0))
cpp_time = time.time() - start

print(f"Python backend: {py_time:.4f}s")
print(f"C++ backend: {cpp_time:.4f}s")
print(f"Speedup: {py_time/cpp_time:.2f}x")

W1_latent = np.random.randn(IN_DIM, HIDDEN) * 0.01
b1 = np.zeros((1, HIDDEN))
W2_latent = np.random.randn(HIDDEN, N_CLASSES) * 0.01
b2 = np.zeros((1, N_CLASSES))
m_W1 = np.zeros_like(W1_latent)
m_b1 = np.zeros_like(b1)
m_W2 = np.zeros_like(W2_latent)
m_b2 = np.zeros_like(b2)
beta = 0.9

def predict(x_bin):
    W1_bin = np.where(W1_latent > 0, 1, -1).astype(np.int8)
    W2_bin = np.where(W2_latent > 0, 1, -1).astype(np.int8)
    t0 = x_bin.astype(np.int32) @ W1_bin.astype(np.int32)
    t1 = t0 + b1
    t2 = np.where(t1 > 0, 1, -1).astype(np.int8)
    t3 = t2.astype(np.int32) @ W2_bin.astype(np.int32)
    return t3 + b2

lr = 0.0001
n_epochs = 300
steps_per_epoch = len(X_train_bin) // BATCH
# time one epoch of training with python vs cpp backend
W1_init = W1_latent.copy()
b1_init = b1.copy()
W2_init = W2_latent.copy()
b2_init = b2.copy()

def run_trial(train_fn, run_epochs=1):
    W1 = W1_init.copy(); b1x = b1_init.copy(); W2 = W2_init.copy(); b2x = b2_init.copy()
    mW1 = np.zeros_like(W1); mb1 = np.zeros_like(b1x); mW2 = np.zeros_like(W2); mb2 = np.zeros_like(b2x)
    start = time.time()
    for epoch in range(run_epochs):
        perm = np.random.permutation(len(X_train_bin))
        X_shuf, y_shuf = X_train_bin[perm], y_train[perm]
        for step in range(steps_per_epoch):
            i = step * BATCH
            x_batch = X_shuf[i:i+BATCH]
            y_batch = y_shuf[i:i+BATCH]
            y_oh = np.eye(N_CLASSES)[y_batch]
            d_loss = np.array(1.0)
            loss, d_W1_latent, d_b1, d_W2_latent, d_b2 = train_fn(x_batch, W1, b1x, W2, b2x, y_oh, d_loss)
            mW1 = beta * mW1 + (1 - beta) * d_W1_latent
            mb1 = beta * mb1 + (1 - beta) * d_b1
            mW2 = beta * mW2 + (1 - beta) * d_W2_latent
            mb2 = beta * mb2 + (1 - beta) * d_b2
            W1 = np.clip(W1 - lr * mW1, -1.0, 1.0)
            b1x = b1x - lr * mb1
            W2 = np.clip(W2 - lr * mW2, -1.0, 1.0)
            b2x = b2x - lr * mb2
    elapsed = time.time() - start
    preds = predict(X_test_bin)  # predict uses global W1_latent/W2_latent; override temporarily
    return elapsed

# run and time
py_time = run_trial(train_step_py, run_epochs=1)
cpp_time = run_trial(train_step_cpp, run_epochs=1)

print(f"Python training (1 epoch): {py_time:.4f}s")
print(f"C++ training (1 epoch): {cpp_time:.4f}s")
print(f"Speedup: {py_time/cpp_time:.2f}x")