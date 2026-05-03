import numpy as np
from compile import compile, backward
from compile_cpp import compile_cpp
from ops import *
from sklearn.datasets import fetch_openml
from skimage.transform import resize

np.random.seed(0)

print("Loading MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data.astype(np.float32) / 255.0
X_88 = np.array([resize(img.reshape(28, 28), (8, 8), anti_aliasing=True).flatten() for img in X])
y = mnist.target.astype(np.int64)

X_train, X_test = X_88[:60000], X_88[60000:]
y_train, y_test = y[:60000], y[60000:]

X_train_bin = np.where(X_train > 0.5, 1, -1).astype(np.int8)
X_test_bin  = np.where(X_test  > 0.5, 1, -1).astype(np.int8)
print(f"Train: {X_train_bin.shape}, Test: {X_test_bin.shape}")

IN_DIM = 64
HIDDEN = 256
N_CLASSES = 10
BATCH = 256

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

train_step = compile_cpp(extended_ops, ext_inputs, ext_outputs, input_specs)

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

# Training
lr = 0.0001
n_epochs = 300
steps_per_epoch = len(X_train_bin) // BATCH

for epoch in range(n_epochs):
    perm = np.random.permutation(len(X_train_bin))
    X_shuf, y_shuf = X_train_bin[perm], y_train[perm]
    epoch_loss = 0.0

    for step in range(steps_per_epoch):
        i = step * BATCH
        x_batch = X_shuf[i:i+BATCH]
        y_batch = y_shuf[i:i+BATCH]
        y_oh = np.eye(N_CLASSES)[y_batch]

        d_loss = np.array(1.0)
        loss, d_W1_latent, d_b1, d_W2_latent, d_b2 = train_step(x_batch, W1_latent, b1, W2_latent, b2, y_oh, d_loss)

        m_W1 = beta * m_W1 + (1 - beta) * d_W1_latent
        m_b1 = beta * m_b1 + (1 - beta) * d_b1
        m_W2 = beta * m_W2 + (1 - beta) * d_W2_latent
        m_b2 = beta * m_b2 + (1 - beta) * d_b2

        W1_latent = np.clip(W1_latent - lr * m_W1, -1.0, 1.0)
        b1 = b1 - lr * m_b1
        W2_latent = np.clip(W2_latent - lr * m_W2, -1.0, 1.0)
        b2 = b2 - lr * m_b2
        
        epoch_loss += loss

    avg_loss = epoch_loss / steps_per_epoch
    test_preds = predict(X_test_bin)
    test_acc = (test_preds.argmax(axis=1) == y_test).mean()
    print(f"Epoch {epoch+1:2d}/{n_epochs}  avg_loss={avg_loss:.4f}  test_acc={test_acc*100:.2f}%")

np.savez("bnn_8x8_trained.npz", W1_latent=W1_latent, b1=b1, W2_latent=W2_latent, b2=b2)