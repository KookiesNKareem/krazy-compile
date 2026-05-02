import numpy as np
from compile import compile
from compile_verilog import compile_verilog
from ops import Const, Add, BinaryMatmul, Sign

W1 = np.random.choice([-1, 1], size=(16, 8)).astype(np.int8)
W2 = np.random.choice([-1, 1], size=(8, 4)).astype(np.int8)
b1 = np.random.randint(-3, 3, size=(1, 8), dtype=np.int8)
b2 = np.random.randint(-3, 3, size=(1, 4), dtype=np.int8)

ops = [
    Const(value=W1, out="W1"),
    Const(value=b1, out="b1"),
    Const(value=W2, out="W2"),
    Const(value=b2, out="b2"),
    BinaryMatmul(a="x",  b="W1", out="t0"),
    Add(         a="t0", b="b1", out="t1"),
    Sign(        a="t1",         out="t2"),
    BinaryMatmul(a="t2", b="W2", out="t3"),
    Add(         a="t3", b="b2", out="y"),
]

# Reference x — note shape (1, 16) for matmul (M=1, K=16)
x_spec = np.zeros((1, 16), dtype=np.int8)

f_py  = compile(ops, ["x"], "y")
f_sv  = compile_verilog(ops, ["x"], "y", {"x": x_spec})

for _ in range(20):
    x = np.random.choice([-1, 1], size=(1, 16)).astype(np.int8)
    py_out = f_py(x)
    sv_out = f_sv(x)
    assert np.array_equal(py_out, sv_out), (py_out, sv_out)
print("PASS — both backends agree")