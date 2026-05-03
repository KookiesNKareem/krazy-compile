from ops import *
import numpy as np

def _reduce_to_shape(a, target_shape):
    a = np.asarray(a)
    while a.ndim > len(target_shape):
        a = a.sum(axis=0)
    for i in range(a.ndim):
        if target_shape[i] == 1 and a.shape[i] != 1:
            a = a.sum(axis=i, keepdims=True)
    return a

def compile(ops, input_names, output_name):
    single_output = 1 if isinstance(output_name, str) else 0
    if single_output:
        output_name = list(output_name)

    f_str = "def f("
    f_str += ",".join(input_names)
    f_str += "):"

    ns = {"np": np, "_reduce_to_shape": _reduce_to_shape}
    consts = 0
    for op in ops:
        if isinstance(op, Matmul):
            f_str += f"\n\t{op.out} = {op.a} @ {op.b}"
        elif isinstance(op, BinaryMatmul):
            f_str += f"\n\t{op.out} = ({op.a}.astype(np.int32) @ {op.b}.astype(np.int32))"
        elif isinstance(op, Sign):
            f_str += f"\n\t{op.out} = np.where({op.a} > 0, 1, -1).astype(np.int8)"
        elif isinstance(op, Add):
            f_str += f"\n\t{op.out} = {op.a} + {op.b}"
        elif isinstance(op, ReLU):
            f_str += f"\n\t{op.out} = np.maximum({op.a}, 0)"
        elif isinstance(op, Const):
            f_str += f"\n\t{op.out} = _const{consts}_"
            ns[f"_const{consts}_"] = op.value
            consts += 1
        elif isinstance(op, Identity):
            f_str += f"\n\t{op.out} = {op.a}"
        elif isinstance(op, Transpose):
            f_str += f"\n\t{op.out} = {op.a}.T"
        elif isinstance(op, ReLUBackward):
            f_str += f"\n\t{op.out} = np.where({op.a} > 0, {op.dy}, 0)"
        elif isinstance(op, MSELoss):
            f_str += f"\n\t{op.out} = np.mean(({op.pred} - {op.target}) ** 2)"
        elif isinstance(op, MSELossBackward):
            f_str += f"\n\t{op.out} = ({op.dy} * 2.0 / {op.pred}.size) * ({op.pred} - {op.target})"
        elif isinstance(op, Broadcast):
            f_str += f"\n\t{op.out} = np.broadcast_to({op.a}, {op.out_shape}).copy()"
        elif isinstance(op, ReduceSumToShape):
            f_str += f"\n\t{op.out} = _reduce_to_shape({op.a}, {op.target_shape})"



    if single_output:
        f_str += f"\n\treturn {output_name}"
    else:
        f_str += f"\n\treturn ({', '.join(output_name)},)"

    exec(f_str, ns)
    return ns["f"]


def backward(ops, input_names, output_name, params=None):
    gradients = dict()
    gradients[output_name] = f"d_{output_name}"

    backward_ops = []
    for op in reversed(ops):
        if isinstance(op, Const):
              continue
        g_out = gradients[op.out]

        if isinstance(op, Add):
            backward_ops.append(Identity(a=g_out, out=f"d_{op.a}"))
            backward_ops.append(Identity(a=g_out, out=f"d_{op.b}"))
            gradients[op.a] = f"d_{op.a}"
            gradients[op.b] = f"d_{op.b}"
        elif isinstance(op, Identity):
            backward_ops.append(Identity(a=g_out, out=f"d_{op.a}"))
            gradients[op.a] = f"d_{op.a}"
        elif isinstance(op, Matmul):
            # d_a = d_y @ b.T
            b_T = f"_bT_{op.b}_for_{op.out}"
            backward_ops.append(Transpose(a=op.b, out=b_T))
            backward_ops.append(Matmul(a=g_out, b=b_T, out=f"d_{op.a}"))
            gradients[op.a] = f"d_{op.a}"

            # d_b = a.T @ d_y
            a_T = f"_aT_{op.a}_for_{op.out}"
            backward_ops.append(Transpose(a=op.a, out=a_T))
            backward_ops.append(Matmul(a=a_T, b=g_out, out=f"d_{op.b}"))
            gradients[op.b] = f"d_{op.b}"
        elif isinstance(op, BinaryMatmul):
            # d_a = d_y @ b.T
            b_T = f"_bT_{op.b}_for_{op.out}"
            backward_ops.append(Transpose(a=op.b, out=b_T))
            backward_ops.append(Matmul(a=g_out, b=b_T, out=f"d_{op.a}"))
            gradients[op.a] = f"d_{op.a}"

            # d_b = a.T @ d_y
            a_T = f"_aT_{op.a}_for_{op.out}"
            backward_ops.append(Transpose(a=op.a, out=a_T))
            backward_ops.append(Matmul(a=a_T, b=g_out, out=f"d_{op.b}"))
            gradients[op.b] = f"d_{op.b}"
        elif isinstance(op, Sign):
            backward_ops.append(Identity(a=g_out, out=f"d_{op.a}"))
            gradients[op.a] = f"d_{op.a}"
        elif isinstance(op, ReLU):
            backward_ops.append(ReLUBackward(a=op.a, dy=g_out, out=f"d_{op.a}"))
            gradients[op.a] = f"d_{op.a}"
        elif isinstance(op, MSELoss):
            backward_ops.append(MSELossBackward(pred=op.pred, target=op.target, dy=g_out, out=f"d_{op.pred}"))
            gradients[op.pred] = f"d_{op.pred}"
        elif isinstance(op, Broadcast):
            backward_ops.append(
                ReduceSumToShape(a=g_out, target_shape=op.a_shape, out=f"d_{op.a}")
            )
            gradients[op.a] = f"d_{op.a}"
        else:
            raise NotImplementedError(f"backward not implemented for {type(op).__name__}")

    diff_set = params if params is not None else input_names
    extended_ops = ops + backward_ops
    extended_input_names = input_names + [f"d_{output_name}"]
    extended_output_names = [output_name] + [f"d_{n}" for n in diff_set]
    return extended_ops, extended_input_names, extended_output_names