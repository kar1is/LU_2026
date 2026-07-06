from dataclasses import dataclass
import numpy as np
import sympy

g_lvl = 1
e_lvl = 0

@dataclass
class State:
    n: object
    L: object
    F: object
    mF: object
    

def sympy_range(start: object, stop: object, step=1: object):
    while start < stop:
        yield start
        start += step
        
def create_states(n: object, L: object, F: object) -> object:
    return [State(n, L, F, sympy.Rational(f'{mF}'))
            for mF in sympy_range((-1)*F, F+1)]

class DensityMatrix:
    def __init__(self, states):
        self.states = states
        self.num = len(states)
        self.create_mtx()
     
    def create_mtx(self):
        st_num = len(self.states)
        self.mtx = np.zeros((st_num, st_num), dtype='object')
        for i in range(st_num):
            for j in range(st_num):
                self.mtx[i,j] = sympy.Mul(self.states[i],self.states[j])
    
    def to_list(self):
        return self.mtx.ravel()


def new_default_Hamiltonian(density):
    for i in range(density.num):
        if i > 0:
            density.mtx[i,i] = sympy.Mul(density.mtx[i,i],
                               sympy.Symbol(f'\u0394_{density.states[i]}'))
        if i < density.num-1:
            density.mtx[i,i+1] = sympy.Mul(density.mtx[i,i],
                               sympy.Symbol(f'\u03A9_{density.states[i]}'))
            density.mtx[i+1,i] = sympy.Mul(density.mtx[i+1,i],
                               sympy.Symbol(f'\u03A9_{density.states[i]}'))
    return density

ground = create_states(sympy.Integer(6), sympy.Integer(0), sympy.Rational(3,2))
print(ground)
print(ground[0])

