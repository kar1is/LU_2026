from densitymatrix import State, HyperfineState
import sympy as sp

### Basic ###
    
def com(A: sp.MatrixBase, B: sp.MatrixBase) -> sp.MatrixBase:
    '''Matrix commutator.'''
    return A*B - B*A

def acom(A, B):
    '''Matrix anticommutator.'''
    return A*B + B*A
         
def create_states(n: sp.Expr, L: sp.Expr, F: sp.Expr) -> list[State]:
    '''Creates hyperfine states (for now).'''
    def _sympy_range(start: object, stop: object, step: object = 1):
        while start < stop:
            yield start
            start += step
    return [HyperfineState(n, L, F, sp.Rational(f'{mF}'))
            for mF in _sympy_range((-1)*F, F+1)]


### Interaction ###

def to_list(mtx):
    tmp_list = []
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            tmp_list.append(mtx[i,j])
    return sp.Matrix(tmp_list)