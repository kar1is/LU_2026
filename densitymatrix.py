from dataclasses import dataclass
import sympy as sp

@dataclass
class State:
    pass

@dataclass
class HyperfineState(State):
    n: object
    L: object
    F: object
    mF: sp.Expr

class DensityMatrix:
    def __init__(self, states):
        self.states = states
        self.len = len(states)
        self.reps = [sp.symbols(f'\u03C8{i}', commutative=False)
                     for i in range(1, self.len+1)]
        self._create_mtx()
        self._to_list()
     
    def _create_mtx(self):
        self.mtx = sp.zeros(self.len, self.len)
        for i in range(self.len):
            for j in range(self.len):
                self.mtx[i,j] = sp.symbols(f'{self.reps[i]}{self.reps[j]}')
    
    def _to_list(self):
        tmp_list = []
        for i in range(self.mtx.shape[0]):
            for j in range(self.mtx.shape[1]):
                tmp_list.append(self.mtx[j,i])
        self.list = sp.Matrix(tmp_list)