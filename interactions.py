from dataclasses import dataclass
import sympy as sp

@dataclass
class Field:
    ampl: object
    freq: sp.Expr
    pol: str

@dataclass
class BField(Field):
    pass

@dataclass
class EField(Field):
    pass
    