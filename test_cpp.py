import numpy as np
from compile_cpp import compile_cpp
from ops import *

W1 = np.random.choice([-1.0, 1.0], size=(4, 8))
W2 = np.random.choice([-1.0, 1.0], size=(8, 3))

ops = [
    Const(value=W1, out="W1"),
    Const(value=W2, out="W2"),
    BinaryMatmul(a="x",  b="W1", out="t0"),
    Sign(        a="t0",         out="t2"),
    BinaryMatmul(a="t2", b="W2", out="y"),
]

input_specs = {"x": np.zeros((1, 4))}
f = compile_cpp(ops, ["x"], "y", input_specs)

def ref(x):
    t0 = x @ W1
    t2 = np.where(t0 > 0, 1.0, -1.0)
    return t2 @ W2

for _ in range(50):
    x = np.random.choice([-1.0, 1.0], size=(1, 4))
    assert np.allclose(f(x), ref(x))

print("PASS")
