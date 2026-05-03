from ops import *
import numpy as np
import subprocess
import tempfile, os, subprocess, ctypes

def _eigen_include() -> str:
    prefix = subprocess.run(
        ["brew", "--prefix", "eigen"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return f"{prefix}/include/eigen3"

EIGEN_INCLUDE = _eigen_include()

def ptr_of(name, intermediates):
      return f"{name}.data()" if name in intermediates else name

def compile_cpp(ops, input_names, output_names, input_specs):
    c_str = '#include <Eigen/Dense>\n#include <cstddef>\n#include <vector>\n#include <cmath>\n\nextern "C" void f(\n'
    in_names = input_names if isinstance(input_names, (list, tuple)) else [input_names]
    out_names = output_names if isinstance(output_names, (list, tuple)) else [output_names]
    single_output = isinstance(output_names, str) or len(out_names) == 1

    shapes = {n: tuple(np.asarray(input_specs[n]).shape) for n in in_names}
    for op in ops:
        if isinstance(op, Add):
            assert shapes[op.a] == shapes[op.b], f"Add shape mismatch {shapes[op.a]} vs {shapes[op.b]}"
            shapes[op.out] = shapes[op.a]
        elif isinstance(op, Matmul):
            sa, sb = shapes[op.a], shapes[op.b]
            assert len(sa) == 2 and len(sb) == 2, "v2-cpp Matmul: 2D inputs only"
            M, K1 = sa
            K2, N = sb
            assert K1 == K2, f"Matmul K mismatch: {K1} vs {K2}"
            shapes[op.out] = (M, N)
        elif isinstance(op, Sign):
            shapes[op.out] = shapes[op.a]

        elif isinstance(op, BinaryMatmul):
            sa, sb = shapes[op.a], shapes[op.b]
            assert len(sa) == 2 and len(sb) == 2, "BinaryMatmul: 2D inputs only"
            M, K1 = sa
            K2, N = sb
            assert K1 == K2
            shapes[op.out] = (M, N)

        elif isinstance(op, Const):
            shapes[op.out] = tuple(np.asarray(op.value).shape)
        elif isinstance(op, Identity):
            shapes[op.out] = shapes[op.a]
        elif isinstance(op, Transpose):
            s = shapes[op.a]
            assert len(s) == 2, "Transpose: 2D only"
            shapes[op.out] = (s[1], s[0])
        elif isinstance(op, ReLU):
            shapes[op.out] = shapes[op.a]
        elif isinstance(op, ReLUBackward):
            shapes[op.out] = shapes[op.a]
        elif isinstance(op, Broadcast):
            shapes[op.out] = op.out_shape
        elif isinstance(op, ReduceSumToShape):
            shapes[op.out] = op.target_shape
        elif isinstance(op, MSELoss):
            assert shapes[op.pred] == shapes[op.target]
            shapes[op.out] = ()
        elif isinstance(op, MSELossBackward):
            shapes[op.out] = shapes[op.pred]
        elif isinstance(op, SoftmaxCrossEntropy):
            assert shapes[op.pred] == shapes[op.target]
            assert len(shapes[op.pred]) == 2, "SoftmaxCE: 2D inputs only"
            shapes[op.out] = ()
        elif isinstance(op, SoftmaxCrossEntropyBackward):
            shapes[op.out] = shapes[op.pred]

        else:
            raise NotImplementedError(f"{type(op).__name__} not yet implemented")

    const_names = []
    const_values = []
    for op in ops:
        if isinstance(op, Const):
            const_names.append(op.out)
            const_values.append(np.ascontiguousarray(op.value, dtype=np.float64))
    
    params = []
    for n in in_names:
        params.append(f"\tconst double* {n}")
    for n in const_names:
        params.append(f"\tconst double* {n}")
    for n in out_names:
        params.append(f"\tdouble* {n}")

    c_str += ",\n".join(params) + "\n) {\n"
    c_str += "\tusing RowMatXd = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;\n"

    intermediates = []
    for op in ops:
        if isinstance(op, Const):
            continue
        if op.out not in in_names and op.out not in out_names and op.out not in const_names:
            if op.out not in intermediates:
                intermediates.append(op.out)

    for name in intermediates:
        n = int(np.prod(shapes[name]))
        c_str += f"    std::vector<double> {name}({n});\n"

    for op in ops:
        if isinstance(op, Const):
            continue
        if isinstance(op, Add):
            n = int(np.prod(shapes[op.out]))
            c_str += f"\tfor (std::size_t i = 0; i < {n}; i++) {op.out}[i] = {op.a}[i] + {op.b}[i];\n"
        elif isinstance(op, Matmul) or isinstance(op, BinaryMatmul):
            M, K = shapes[op.a]
            _, N = shapes[op.b]
            a_ptr = ptr_of(op.a, intermediates)
            b_ptr = ptr_of(op.b, intermediates)
            out_ptr = ptr_of(op.out, intermediates)

            c_str += f"""\t{{Eigen::Map<const RowMatXd> a_m({a_ptr}, {M}, {K});\nEigen::Map<const RowMatXd> b_m({b_ptr}, {K}, {N});\nEigen::Map<RowMatXd> y_m({out_ptr}, {M}, {N});
            y_m.noalias() = a_m * b_m;\n}}\n"""
        elif isinstance(op, Identity):
            n = int(np.prod(shapes[op.out]))
            c_str += f"    for (std::size_t i = 0; i < {n}; i++) {op.out}[i] = {op.a}[i];\n"
        elif isinstance(op, Transpose):
            M, N = shapes[op.a]
            a_ptr   = ptr_of(op.a, intermediates)
            out_ptr = ptr_of(op.out, intermediates)
            c_str += f"""\t{{Eigen::Map<const RowMatXd> a_m({a_ptr}, {M}, {N});\nEigen::Map<RowMatXd> y_m({out_ptr}, {N}, {M});\ny_m.noalias() = a_m.transpose();}}\n"""
        elif isinstance(op, ReLU):
            n = int(np.prod(shapes[op.out]))
            c_str += f"\tfor (std::size_t i = 0; i < {n}; i++) {op.out}[i] = {op.a}[i] > 0.0 ? {op.a}[i] : 0.0;\n"
        elif isinstance(op, ReLUBackward):
            n = int(np.prod(shapes[op.out]))
            c_str += f"    for (std::size_t i = 0; i < {n}; i++) {op.out}[i] = {op.a}[i] > 0.0 ? {op.dy}[i] : 0.0;\n"
        elif isinstance(op, Sign):
            n = int(np.prod(shapes[op.out]))
            c_str += f"\tfor (std::size_t i = 0; i < {n}; i++) {op.out}[i] = {op.a}[i] > 0.0 ? 1.0 : -1.0;\n"
        elif isinstance(op, Broadcast):
            a_shape = op.a_shape
            out_shape = op.out_shape
            n_out = int(np.prod(out_shape))
            pad = len(out_shape) - len(a_shape)
            a_strides_padded = [0]*pad + [0 if d == 1 else int(np.prod(a_shape[i+1:])) for i, d in enumerate(a_shape)]
            out_strides = [int(np.prod(out_shape[i+1:])) for i in range(len(out_shape))]
            body = "            std::size_t src_off = 0; std::size_t rem = i;\n"
            for d in range(len(out_shape)):
                body += f"            {{ std::size_t c = rem / {out_strides[d]}; rem -= c * {out_strides[d]}; src_off += c *{a_strides_padded[d]}; }}\n"
            body += f"            {op.out}[i] = {op.a}[src_off];\n"
            c_str += f"\t{{\n        for (std::size_t i = 0; i < {n_out}; i++) {{\n{body}        }}\n    }}\n"

        elif isinstance(op, ReduceSumToShape):
            a_shape = shapes[op.a]
            target = op.target_shape
            pad = len(a_shape) - len(target)
            n_a = int(np.prod(a_shape))
            n_target = int(np.prod(target))
            a_strides = [int(np.prod(a_shape[i+1:])) for i in range(len(a_shape))]
            target_strides_padded = [0]*pad + [
                0 if d == 1 else int(np.prod(target[i+1:]))
                for i, d in enumerate(target)
            ]
            c_str += f"    for (std::size_t i = 0; i < {n_target}; i++) {op.out}[i] = 0.0;\n"
            body  = "            std::size_t dst_off = 0; std::size_t rem = i;\n"
            for d in range(len(a_shape)):
                body += f"            {{ std::size_t c = rem / {a_strides[d]}; rem -= c * {a_strides[d]}; dst_off += c *{target_strides_padded[d]}; }}\n"
            body += f"            {op.out}[dst_off] += {op.a}[i];\n"
            c_str += f"    {{\n        for (std::size_t i = 0; i < {n_a}; i++) {{\n{body}        }}\n    }}\n"
        elif isinstance(op, MSELoss):
            n = int(np.prod(shapes[op.pred]))
            c_str += f"""\t{{
                double sum = 0.0;
                for (std::size_t i = 0; i < {n}; i++) {{
                    double d = {op.pred}[i] - {op.target}[i];
                    sum += d * d;
                }}
                {op.out}[0] = sum / {n}.0;
            }}\n"""
        elif isinstance(op, MSELossBackward):
            n = int(np.prod(shapes[op.pred]))
            c_str += f"""    {{
                double dy_val = {op.dy}[0];
                double scale = 2.0 / {n}.0 * dy_val;
                for (std::size_t i = 0; i < {n}; i++) {{
                    {op.out}[i] = scale * ({op.pred}[i] - {op.target}[i]);
                }}
            }}\n"""
        elif isinstance(op, SoftmaxCrossEntropy):
            B, C = shapes[op.pred]
            c_str += f"""    {{
                double total = 0.0;
                for (std::size_t i = 0; i < {B}; i++) {{
                    // numerically-stable log-sum-exp
                    double row_max = {op.pred}[i * {C}];
                    for (std::size_t j = 1; j < {C}; j++) {{
                        double v = {op.pred}[i * {C} + j];
                        if (v > row_max) row_max = v;
                    }}
                    double sum_exp = 0.0;
                    for (std::size_t j = 0; j < {C}; j++) {{
                        sum_exp += std::exp({op.pred}[i * {C} + j] - row_max);
                    }}
                    double log_sum_exp = std::log(sum_exp) + row_max;
                    // -sum_j target[j] * log p[j]  =  -sum_j target[j] * (pred[j] - log_sum_exp)
                    double row_loss = 0.0;
                    for (std::size_t j = 0; j < {C}; j++) {{
                        row_loss -= {op.target}[i * {C} + j]
                                    * ({op.pred}[i * {C} + j] - log_sum_exp);
                    }}
                    total += row_loss;
                }}
                {op.out}[0] = total / {B}.0;
            }}\n"""
        elif isinstance(op, SoftmaxCrossEntropyBackward):
            B, C = shapes[op.pred]
            c_str += f"""    {{
                double dy_val = {op.dy}[0];
                double scale = dy_val / {B}.0;
                for (std::size_t i = 0; i < {B}; i++) {{
                    double row_max = {op.pred}[i * {C}];
                    for (std::size_t j = 1; j < {C}; j++) {{
                        double v = {op.pred}[i * {C} + j];
                        if (v > row_max) row_max = v;
                    }}
                    double sum_exp = 0.0;
                    for (std::size_t j = 0; j < {C}; j++) {{
                        sum_exp += std::exp({op.pred}[i * {C} + j] - row_max);
                    }}
                    for (std::size_t j = 0; j < {C}; j++) {{
                        double p = std::exp({op.pred}[i * {C} + j] - row_max) / sum_exp;
                        {op.out}[i * {C} + j] = (p - {op.target}[i * {C} + j]) * scale;
                    }}
                }}
            }}\n"""

            
    c_str += "}\n"

    tmp = tempfile.mkdtemp()
    src_path = os.path.join(tmp, "f.cpp")
    lib_path = os.path.join(tmp, "libf.so")

    with open(src_path, "w") as fh:
        fh.write(c_str)

    subprocess.run(["g++", "-O3", "-march=native", "-ffast-math", "-Wno-nan-infinity-disabled", "-shared", "-fPIC", "-std=c++17", "-I", EIGEN_INCLUDE, "-o", lib_path, src_path], check=True)

    lib = ctypes.CDLL(lib_path)
    ptr_t = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.f.restype = None

    output_shapes = [shapes[n] for n in out_names]

    n_total = len(in_names) + len(const_names) + len(out_names)
    lib.f.argtypes = [ptr_t] * n_total

    def f(*user_args):
        if len(user_args) != len(in_names):
            raise TypeError(f"expected {len(in_names)} inputs, got {len(user_args)}")
        user_args = [np.ascontiguousarray(a, dtype=np.float64) for a in user_args]
        outputs = [np.empty(s, dtype=np.float64) for s in output_shapes]
        lib.f(*user_args, *const_values, *outputs)
        return outputs[0] if single_output else tuple(outputs)

    return f