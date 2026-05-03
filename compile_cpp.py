from ops import *
import subprocess

def _eigen_include() -> str:
    prefix = subprocess.run(
        ["brew", "--prefix", "eigen"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return f"{prefix}/include/eigen3"

EIGEN_INCLUDE = _eigen_include()

def compile_cpp(ops, input_names, output_names, input_specs):
    c_str = '#include <Eigen/Dense>\n#include <cstddef>\n#include <vector>\n\nextern "C" void f(\n'
    in_names = input_names if isinstance(input_names, (list, tuple)) else [input_names]
    out_names = output_names if isinstance(output_names, (list, tuple)) else [output_names]
    single_output = isinstance(output_names, str) or len(out_names) == 1

    params = []
    for n in in_names:
        params.append(f"  const double* {n}")
    for n in out_names:
        params.append(f"  double* {n}")

    c_str += ",\n".join(params) + "\n) {\n"

    shapes = {n: tuple(np.asarray(input_specs[n]).shape) for n in in_names}
    for op in ops:
        if isinstance(op, Add):
            assert shapes[op.a] == shapes[op.b], f"Add shape mismatch {shapes[op.a]} vs {shapes[op.b]}"
            shapes[op.out] = shapes[op.a]
        else:
            raise NotImplementedError(f"v1 only handles Add, got {type(op).__name__}")

    intermediates = []
    for op in ops:
        if op.out not in in_names and op.out not in out_names:
            if op.out not in intermediates:
                intermediates.append(op.out)

    for name in intermediates:
        n = int(np.prod(shapes[name]))
        c_str += f"    std::vector<double> {name}({n});\n"

    for op in ops:
        if isinstance(op, Add):
            n = int(np.prod(shapes[op.out]))
            # If op.out is a passed-in pointer, no .data() needed; std::vector needs [] which works on both
            c_str += f"    for (std::size_t i = 0; i < {n}; i++) {op.out}[i] = {op.a}[i] + {op.b}[i];\n"

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

    def f(*args):
        if len(args) != len(in_names):
            raise TypeError(f"expected {len(in_names)} inputs, got {len(args)}")
        args = [np.ascontiguousarray(a, dtype=np.float64) for a in args]
        outputs = [np.empty(s, dtype=np.float64) for s in output_shapes]
        lib.f(*args, *outputs)
        return outputs[0] if single_output else tuple(outputs)

    return f