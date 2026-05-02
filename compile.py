from ops import *
import numpy as np

def compile(ops, input_names, output_name):
    f_str = "def f("
    f_str += ",".join(input_names)
    f_str += "):"

    ns = {"np": np}
    consts = 0
    for op in ops:
        if isinstance(op, Matmul):
            f_str += f"\n    {op.out} = {op.a} @ {op.b}"
        elif isinstance(op, BinaryMatmul):
            f_str += f"\n    {op.out} = ({op.a}.astype(np.int32) @ {op.b}.astype(np.int32))"
        elif isinstance(op, Sign):
            f_str += f"\n    {op.out} = np.where({op.a} > 0, 1, -1).astype(np.int8)"
        elif isinstance(op, Add):
            f_str += f"\n    {op.out} = {op.a} + {op.b}"
        elif isinstance(op, ReLU):
            f_str += f"\n    {op.out} = np.maximum({op.a}, 0)"
        elif isinstance(op, Const):
            f_str += f"\n    {op.out} = _const{consts}_"
            ns[f"_const{consts}_"] = op.value
            consts += 1

    f_str += f"\n    return {output_name}"

    exec(f_str, ns)
    return ns["f"]