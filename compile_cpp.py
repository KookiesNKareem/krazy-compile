from ops import *
import numpy as np
import subprocess

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
    c_str = '#include <Eigen/Dense>\n#include <cstddef>\n#include <vector>\n\nextern "C" void f(\n'
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
        else:
            raise NotImplementedError(f"v1 only handles Add, got {type(op).__name__}")

    const_names = []
    const_values = []
    for op in ops:
        if isinstance(op, Const):
            const_names.append(op.out)
            const_values.append(np.ascontiguousarray(op.value, dtype=np.float64))
    
    params = []
    for n in in_names:
        params.append(f"  const double* {n}")
    for n in const_names:
        params.append(f"  const double* {n}")
    for n in out_names:
        params.append(f"  double* {n}")

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
            y_m.noalias() = a_m * b_m;\n
            }}"""
        elif isinstance(op, Sign):
            n = int(np.prod(shapes[op.out]))
            out_ref = ptr_of(op.out, intermediates)
            a_ref = ptr_of(op.a,   intermediates)
            c_str += f"\tfor (std::size_t i = 0; i < {n}; i++) {op.out}[i] = {op.a}[i] > 0.0 ? 1.0 : -1.0;\n"
    c_str += "}\n"

    import tempfile, os, subprocess, ctypes

    tmp = tempfile.mkdtemp()
    src_path = os.path.join(tmp, "f.cpp")
    lib_path = os.path.join(tmp, "libf.so")

    with open(src_path, "w") as fh:
        fh.write(c_str)

    subprocess.run(["g++", "-O3", "-march=native", "-ffast-math", "-Wno-nan-infinity-disabled", "-shared", "-fPIC", "-std=c++17", "-I", EIGEN_INCLUDE, "-o", lib_path, src_path], check=True)

    lib = ctypes.CDLL(lib_path)
    ptr_t = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.f.argtypes = [ptr_t] * (len(in_names) + len(out_names))
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