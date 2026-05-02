import numpy as np
from compile import compile
from compile_verilog import compile_verilog
from ops import *


# scalar add (smoke test for the original use case)
specs = {"a": np.zeros((), dtype=np.int8), "b": np.zeros((), dtype=np.int8)}
f = compile_verilog([Add("a", "b", "y")], ["a", "b"], "y", specs)
assert f(5, -3) == 2
assert f(-5, -3) == -8


# vector add
specs = {"a": np.zeros((4,), dtype=np.int8), "b": np.zeros((4,), dtype=np.int8)}
f = compile_verilog([Add("a", "b", "y")], ["a", "b"], "y", specs)
a = np.array([1, 2, 3, 4], dtype=np.int8)
b = np.array([10, -20, 30, -40], dtype=np.int8)
np.testing.assert_array_equal(f(a, b), a.astype(np.int16) + b.astype(np.int16))


# matmul (2,3) @ (3,4) = (2,4)
specs = {"a": np.zeros((2, 3), dtype=np.int8), "b": np.zeros((3, 4), dtype=np.int8)}
f = compile_verilog([Matmul("a", "b", "y")], ["a", "b"], "y", specs)
a = np.array([[1, 2, 3], [-1, 0, 4]], dtype=np.int8)
b = np.array([[1, -2, 3, 4], [5, 6, 7, 8], [-1, 1, 0, -2]], dtype=np.int8)
np.testing.assert_array_equal(f(a, b), (a.astype(np.int32) @ b.astype(np.int32)))


# matmul + add + relu chain: y = relu(a @ b + c)
specs = {
    "a": np.zeros((2, 3), dtype=np.int8),
    "b": np.zeros((3, 2), dtype=np.int8),
    "c": np.zeros((2, 2), dtype=np.int16),
}
ops = [Matmul("a", "b", "ab"), Add("ab", "c", "abc"), ReLU("abc", "y")]
f = compile_verilog(ops, ["a", "b", "c"], "y", specs)
a = np.array([[1, 2, -3], [4, -5, 6]], dtype=np.int8)
b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
c = np.array([[100, -200], [-50, 25]], dtype=np.int16)
expected = np.maximum(a.astype(np.int32) @ b.astype(np.int32) + c.astype(np.int32), 0)
np.testing.assert_array_equal(f(a, b, c), expected)


# const folded into a matmul
weight = np.array([[1, -1], [2, 3], [0, 4]], dtype=np.int8)
specs = {"x": np.zeros((1, 3), dtype=np.int8)}
ops = [Const(weight, "w"), Matmul("x", "w", "y")]
f = compile_verilog(ops, ["x"], "y", specs)
x = np.array([[2, -1, 3]], dtype=np.int8)
np.testing.assert_array_equal(f(x), x.astype(np.int32) @ weight.astype(np.int32))


print("ok")
