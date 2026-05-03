import numpy as np
from compile import compile, backward
from compile_cpp import compile_cpp
from ops import *

# Forward
ops = [
    Matmul(a="x", b="W", out="t0"),
    Broadcast(a="b", a_shape=(1, 3), out_shape=(4, 3), out="b_bcast"),
    Add(a="t0", b="b_bcast", out="t1"),
    ReLU(a="t1", out="y"),
]

input_specs = {
    "x": np.zeros((4, 5)),
    "W": np.zeros((5, 3)),
    "b": np.zeros((1, 3)),
}
f_py = compile(ops, ["x", "W", "b"], "y")
f_cpp = compile_cpp(ops, ["x", "W", "b"], "y", input_specs)

for _ in range(50):
    x = np.random.randn(4, 5)
    W = np.random.randn(5, 3)
    b = np.random.randn(1, 3)
    py = f_py(x, W, b)
    cp = f_cpp(x, W, b)
    assert np.allclose(py, cp), (py, cp)
print("PASS — forward agrees")

# Backward
ext_ops, ext_inputs, ext_outputs = backward(
    ops, ["x", "W", "b"], output_name="y", params=["x", "W", "b"]
)
input_specs_bwd = {
    "x": np.zeros((4, 5)),
    "W": np.zeros((5, 3)),
    "b": np.zeros((1, 3)),
    "d_y": np.zeros((4, 3)),
}
g_py = compile(ext_ops, ext_inputs, ext_outputs)
g_cpp = compile_cpp(ext_ops, ext_inputs, ext_outputs, input_specs_bwd)

for _ in range(50):
    x = np.random.randn(4, 5)
    W = np.random.randn(5, 3)
    b = np.random.randn(1, 3)
    d_y = np.random.randn(4, 3)
    py_outs = g_py(x, W, b, d_y)
    cp_outs = g_cpp(x, W, b, d_y)
    assert len(py_outs) == len(cp_outs)
    for p, c in zip(py_outs, cp_outs):
        assert np.allclose(p, c), (p, c)
print("PASS — backward agrees")
