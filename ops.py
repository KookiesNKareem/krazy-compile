from dataclasses import dataclass
import numpy as np

@dataclass
class Matmul:
    a: str
    b: str
    out: str

@dataclass
class BinaryMatmul:
    a: str
    b: str
    out: str

@dataclass
class Add:
    a: str
    b: str
    out: str

@dataclass
class ReLU:
    a: str
    out: str

@dataclass
class Const:
    value: np.ndarray
    out: str

@dataclass
class Sign:
    a: str
    out: str