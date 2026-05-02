import numpy as np
from compile import compile, backward
from ops import *

ops = [
    BinaryMatmul(a="x", b="W", out="t0"),
    Add(         a="t0", b="b", out="t1"),
    Sign(        a="t1",        out="t2"),
    ReLU(        a="t2",        out="y"),
]
input_names = ["x", "W", "b"]
output_name = "y"

extended_ops, ext_inputs, ext_outputs = backward(ops, input_names, output_name)
f = compile(extended_ops, ext_inputs, ext_outputs)

x   = np.random.choice([-1, 1], size=(2, 4)).astype(np.int8)
W   = np.random.choice([-1, 1], size=(4, 3)).astype(np.int8)
b   = np.random.randn(2, 3)
d_y = np.random.randn(2, 3)

y, d_x, d_W, d_b = f(x, W, b, d_y)

# Reference
def ref():
    t0 = x.astype(np.int32) @ W.astype(np.int32)
    t1 = t0 + b
    t2 = np.where(t1 > 0, 1, -1).astype(np.int8)
    y_ref = np.maximum(t2, 0)

    # Backward
    d_t2 = np.where(t2 > 0, d_y, 0)            # ReLU backward
    d_t1 = d_t2                                  # Sign backward (STE)
    d_t0 = d_t1                                  # Add backward
    d_b_ref = d_t1                               # Add backward
    d_x_ref = d_t0 @ W.astype(np.int32).T        # Matmul backward
    d_W_ref = x.astype(np.int32).T @ d_t0        # Matmul backward

    return y_ref, d_x_ref, d_W_ref, d_b_ref

y_ref, d_x_ref, d_W_ref, d_b_ref = ref()

assert np.allclose(y,   y_ref)
assert np.allclose(d_x, d_x_ref)
assert np.allclose(d_W, d_W_ref)
assert np.allclose(d_b, d_b_ref)
print("PASS")
