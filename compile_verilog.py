from ops import Add, ReLU, Matmul, Const
from dataclasses import dataclass
import math
import os
import subprocess
import tempfile

import numpy as np


@dataclass
class TensorInfo:
    shape: tuple
    width: int

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def _dtype_width(dt) -> int:
    return np.dtype(dt).itemsize * 8


def _ctype(width: int) -> str:
    if width <= 8:
        return "int8_t"
    if width <= 16:
        return "int16_t"
    if width <= 32:
        return "int32_t"
    if width <= 64:
        return "int64_t"
    raise ValueError(f"width {width} > 64 not supported (would need WData[] marshaling)")


def _flat_names(name: str, shape: tuple) -> list:
    if len(shape) == 0:
        return [name]
    return [name + "_" + "_".join(str(i) for i in idx) for idx in np.ndindex(shape)]


def _signed(width: int) -> str:
    return f"signed [{width - 1}:0]"


def _np_out_dtype(width: int):
    if width <= 8:
        return np.int8
    if width <= 16:
        return np.int16
    if width <= 32:
        return np.int32
    return np.int64


def _matmul_module() -> str:
    return """\
module matmul #(
    parameter M = 1, K = 1, N = 1, WA = 8, WB = 8, WY = 16
) (
    input  signed [M*K*WA-1:0] a,
    input  signed [K*N*WB-1:0] b,
    output signed [M*N*WY-1:0] y
);
    genvar i, j;
    generate
        for (i = 0; i < M; i = i + 1) begin : row
            for (j = 0; j < N; j = j + 1) begin : col
                reg signed [WY-1:0] acc;
                integer k;
                always @* begin
                    acc = 0;
                    for (k = 0; k < K; k = k + 1) begin
                        acc = acc
                            + $signed(a[(i*K + k) * WA +: WA])
                            * $signed(b[(k*N + j) * WB +: WB]);
                    end
                end
                assign y[(i*N + j) * WY +: WY] = acc;
            end
        end
    endgenerate
endmodule
"""


def compile_verilog(ops, input_names, output_name, input_specs):
    info: dict = {}
    for name in input_names:
        arr = np.asarray(input_specs[name])
        info[name] = TensorInfo(shape=tuple(arr.shape), width=_dtype_width(arr.dtype))

    consts: list = []
    for op in ops:
        if isinstance(op, Add):
            a, b = info[op.a], info[op.b]
            assert a.shape == b.shape, f"Add shape mismatch {a.shape} vs {b.shape}"
            info[op.out] = TensorInfo(shape=a.shape, width=max(a.width, b.width) + 1)
        elif isinstance(op, ReLU):
            a = info[op.a]
            info[op.out] = TensorInfo(shape=a.shape, width=a.width)
        elif isinstance(op, Matmul):
            a, b = info[op.a], info[op.b]
            assert len(a.shape) == 2 and len(b.shape) == 2, "Matmul: 2D only"
            M, K = a.shape
            K2, N = b.shape
            assert K == K2, f"Matmul K mismatch {K} vs {K2}"
            grow = math.ceil(math.log2(K)) if K > 1 else 0
            info[op.out] = TensorInfo(shape=(M, N), width=a.width + b.width + grow)
        elif isinstance(op, Const):
            arr = np.asarray(op.value)
            info[op.out] = TensorInfo(shape=tuple(arr.shape), width=_dtype_width(arr.dtype))
            consts.append((op.out, arr))
        else:
            raise TypeError(f"unknown op: {op!r}")

    assert output_name in info, f"output {output_name!r} not produced by any op or input"
    out_info = info[output_name]

    sv: list = []
    if any(isinstance(op, Matmul) for op in ops):
        sv.append(_matmul_module())

    ports: list = []
    for name in input_names:
        ti = info[name]
        for elt in _flat_names(name, ti.shape):
            ports.append(f"    input  {_signed(ti.width)} {elt}")
    for elt in _flat_names(output_name, out_info.shape):
        ports.append(f"    output {_signed(out_info.width)} {elt}")
    sv.append("module f (\n" + ",\n".join(ports) + "\n);")

    sv.append("    genvar __i;")
    for name, ti in info.items():
        sv.append(f"    wire {_signed(ti.n_elements * ti.width)} {name}__packed;")

    for name in input_names:
        ti = info[name]
        for k, elt in enumerate(_flat_names(name, ti.shape)):
            sv.append(f"    assign {name}__packed[{k}*{ti.width} +: {ti.width}] = {elt};")

    for name, arr in consts:
        ti = info[name]
        flat = arr.flatten().tolist()
        parts = []
        for v in reversed(flat):
            iv = int(v)
            parts.append(f"-{ti.width}'sd{-iv}" if iv < 0 else f"{ti.width}'sd{iv}")
        sv.append(f"    assign {name}__packed = {{{', '.join(parts)}}};")

    add_n = relu_n = mm_n = 0
    for op in ops:
        if isinstance(op, Const):
            continue
        if isinstance(op, Add):
            a, b, o = info[op.a], info[op.b], info[op.out]
            tag = f"add_{add_n}"; add_n += 1
            sv.append(
                f"    generate\n"
                f"        for (__i = 0; __i < {a.n_elements}; __i = __i + 1) begin : {tag}\n"
                f"            assign {op.out}__packed[__i*{o.width} +: {o.width}] =\n"
                f"                $signed({op.a}__packed[__i*{a.width} +: {a.width}])\n"
                f"              + $signed({op.b}__packed[__i*{b.width} +: {b.width}]);\n"
                f"        end\n"
                f"    endgenerate"
            )
        elif isinstance(op, ReLU):
            a, o = info[op.a], info[op.out]
            tag = f"relu_{relu_n}"; relu_n += 1
            sv.append(
                f"    generate\n"
                f"        for (__i = 0; __i < {a.n_elements}; __i = __i + 1) begin : {tag}\n"
                f"            assign {op.out}__packed[__i*{o.width} +: {o.width}] =\n"
                f"                ($signed({op.a}__packed[__i*{a.width} +: {a.width}]) >= 0)\n"
                f"                ? {op.a}__packed[__i*{a.width} +: {a.width}]\n"
                f"                : {o.width}'sd0;\n"
                f"        end\n"
                f"    endgenerate"
            )
        elif isinstance(op, Matmul):
            a, b, o = info[op.a], info[op.b], info[op.out]
            M, K = a.shape
            _, N = b.shape
            inst = f"mm_{mm_n}"; mm_n += 1
            sv.append(
                f"    matmul #(.M({M}), .K({K}), .N({N}), "
                f".WA({a.width}), .WB({b.width}), .WY({o.width})) {inst} (\n"
                f"        .a({op.a}__packed),\n"
                f"        .b({op.b}__packed),\n"
                f"        .y({op.out}__packed)\n"
                f"    );"
            )

    for k, elt in enumerate(_flat_names(output_name, out_info.shape)):
        sv.append(f"    assign {elt} = {output_name}__packed[{k}*{out_info.width} +: {out_info.width}];")
    sv.append("endmodule")
    v_str = "\n".join(sv) + "\n"

    out_w = out_info.width
    c: list = [
        '#include "Vf.h"',
        '#include "verilated.h"',
        '#include <cstdlib>',
        '#include <cstdint>',
        '#include <cstdio>',
        "",
        "int main(int argc, char** argv) {",
        "    Vf* dut = new Vf;",
    ]
    arg_idx = 1
    for name in input_names:
        ti = info[name]
        for elt in _flat_names(name, ti.shape):
            c.append(f"    dut->{elt} = ({_ctype(ti.width)})atoll(argv[{arg_idx}]);")
            arg_idx += 1
    c.append("    dut->eval();")
    shift = 64 - out_w
    for elt in _flat_names(output_name, out_info.shape):
        c.append(
            f'    printf("%lld ", (long long)((int64_t)((uint64_t)dut->{elt} << {shift}) >> {shift}));'
        )
    c.append('    printf("\\n");')
    c.append("    delete dut;")
    c.append("    return 0;")
    c.append("}")
    c_str = "\n".join(c) + "\n"

    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "f.sv"), "w") as fh:
        fh.write(v_str)
    with open(os.path.join(tmp, "sim.cpp"), "w") as fh:
        fh.write(c_str)
    subprocess.run(
        [
            "verilator", "--cc", "--exe", "--build",
            "--top-module", "f",
            "-Wno-WIDTH", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC",
            "f.sv", "sim.cpp",
        ],
        cwd=tmp, check=True,
    )
    binary = os.path.join(tmp, "obj_dir", "Vf")

    out_dtype = _np_out_dtype(out_w)
    out_shape = out_info.shape

    def f(*args):
        flat: list = []
        for a in args:
            flat.extend(np.asarray(a).flatten().tolist())
        str_args = [str(int(v)) for v in flat]
        result = subprocess.run([binary, *str_args], capture_output=True, text=True, check=True)
        vals = [int(t) for t in result.stdout.split()]
        arr = np.array(vals, dtype=out_dtype)
        if out_shape == ():
            return out_dtype(arr[0])
        return arr.reshape(out_shape)

    return f
