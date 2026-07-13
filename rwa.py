from density_matrix import DensityMatrix
from operators import Operator
from symbols import t
import sympy as sp

def z_rotor(rho: DensityMatrix, angle: sp.Expr) -> sp.MatrixBase:
    z_rotor = sp.zeros(rho.len, rho.len)
    for i in range(rho.len):
        z_rotor[i,i] = sp.srepr(sp.exp(-sp.I * rho.states[i].mF * angle))
    return z_rotor

def RWA(operator, rotation: sp.MatrixBase, omega):
    def keep_harmonics(expr, omega):
        target = sp.I * omega * t
        result = 0
        for term in sp.Add.make_args(sp.expand(expr)):
            keep = True
            for e in term.atoms(sp.exp):
                exponent = sp.expand(e.args[0])
                n = exponent.coeff(target)
                if abs(n) > 1:
                    keep = False
                    break
            if keep:
                result += term
        return sp.simplify(result)
    
    if isinstance(operator, Operator):
        superop = operator
        superop.mtx = rotation.H * superop.mtx * rotation - \
            sp.I * rotation.H * sp.diff(rotation, t)
        for i in range(superop.len):
            for j in range(superop.len):
                superop.mtx[i,j] = keep_harmonics(superop.mtx[i,j].rewrite(sp.exp), omega)
        return superop
    else:
        print('WRONG ENTRY!')
        superop = rotation.H * operator * rotation - \
            sp.I * rotation.H * sp.diff(rotation, t)
        
        for i in range(superop.shape[0]):
            for j in range(superop.shape[1]):
                superop[i,j] = keep_harmonics(superop[i,j], omega)
        return superop
