import numpy as np
from compile_cpp import compile_cpp
from ops import Add

ops = [
    Add(a="a", b="b", out="t0"),
    Add(a="t0", b="c", out="y"),
]
input_specs = {
    "a": np.zeros(5),
    "b": np.zeros(5),
    "c": np.zeros(5),
}

f = compile_cpp(ops, ["a", "b", "c"], "y", input_specs)

a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
c = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
y = f(a, b, c)

assert np.allclose(y, a + b + c)
print("PASS")