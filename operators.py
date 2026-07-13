from densitymatrix import DensityMatrix
from interactions import BField, EField
from symbols import t
from utilities import com, acom
import sympy as sp

class Operator:
    len: int
    symbols: list[object]
    mtx: list[object]
    
    def add_sym(self, symbols: object):
        if not isinstance(symbols, list) and \
            symbols not in self.symbols:
            symbols = [symbols]
        for symbol in symbols:
            if symbol in self.symbols:
                self.symbols.append(symbol)

class Hamiltonian(Operator):
    def __init__(self, rho: DensityMatrix):
        self.len = rho.len
        self.rho = rho
        self.states = rho.states
        self.symbols = []
        self._create_mtx()
    
    def _create_mtx(self):
        self.mtx = sp.zeros(self.len, self.len)
        for i in range(self.len):
            self.add_sym(sp.symbols(f'E{i+1}'))
            self.mtx[i,i] += sp.symbols(f'E{i+1}')
    
    def add_field(self, field):
        for i in range(self.len):
            for j in range(self.len):
                    
                if isinstance(field, BField) and \
                    field.pol == 'pi' and \
                        i == j:
                        self.add_sym(sp.symbols('B0'))
                        self.mtx[i,i] += self.states[i].mF * sp.symbols('B0')
                
                # Creates matrix elements for dipole interaction with B field
                if isinstance(field, BField) and \
                    field.pol != 'pi' and \
                    abs(self.states[i].L - self.states[j].L) == 0 and \
                    abs(self.states[i].mF - self.states[j].mF) == 1:
                        self.add_sym(sp.symbols('\u03A9RF', positive=True))
                        self.add_sym(field.freq)
                        self.mtx[i,j] += sp.Rational(1,2) * \
                            (sp.exp(sp.I * field.freq * t) + \
                             sp.exp(-sp.I * field.freq * t)) * \
                            sp.symbols('\u03A9RF', positive=True)
                
                if isinstance(field, EField) and \
                    field.pol == 'pi' and \
                    abs(self.states[i].L - self.states[j].L) == 1 and \
                    abs(self.states[i].mF - self.states[j].mF) == 0:
                        self.add_sym(sp.symbols('\u03A9', positive=True))
                        self.add_sym(field.freq)
                        self.mtx[i,j] += sp.Rational(1,2) * \
                            (sp.exp(sp.I * field.freq * t) + \
                             sp.exp(-sp.I * field.freq * t)) * \
                            sp.symbols('\u03A9', positive=True)
                
                if isinstance(field, EField) and \
                    field.pol != 'pi' and \
                    abs(self.states[i].L - self.states[j].L) == 1 and \
                    abs(self.states[i].mF - self.states[j].mF) == 1:
                        self.add_sym(sp.symbols('\u03A9', positive=True))
                        self.add_sym(field.freq)
                        self.mtx[i,j] += sp.Rational(1,2) * \
                            (sp.exp(sp.I * field.freq * t) + \
                             sp.exp(-sp.I * field.freq * t)) * \
                            sp.symbols('\u03A9', positive=True)


class Lindblad(Operator):
    coeff = None
    
    def __init__(self, *args):
        if isinstance(args[0], DensityMatrix):
            self.rho = args[0]
            self.len = args[0].len
        elif isinstance(args[0], Hamiltonian):
            self.rho = args[0].rho
            self.len = args[0].len
            self._add_hamiltonian(args[0])
        self.jump = []
        self.lindblad = sp.zeros(self.len, self.len)
    
    def _add_hamiltonian(self, ham):
        self.H = ham
        self.symbols = ham.symbols
        self.mtx = -sp.I * com(ham.mtx, self.rho.mtx)
    
    def add_jump(self, C):
        self.jump.append(C)
        self.add_sym(C.symbols)
        self.mtx += C.mtx*self.rho.mtx*C.mtx.H - \
            sp.Rational(1,2)*acom(C.mtx.H*C.mtx, self.rho.mtx)
    
    def get_coeff(self):
        if self.coeff == None:
            for i in range(self.mtx.shape[0]):
                for j in range(self.mtx.shape[1]):
                    self.mtx[i,j] = sp.collect(self.mtx[i,j], self.rho.list)

            mtx2 = sp.Matrix(self.mtx).reshape(self.rho.len**2,1)

            self.coeff = mtx2.jacobian(self.rho.list)
            return self.coeff
        else:
            return self.coeff

class Project(Operator):
    def __init__(self, rho, ket, bra, coeff):
        self.len = rho.len
        self.mtx = sp.zeros(self.len, self.len)
        if isinstance(ket, int) and isinstance(bra, int):
            self.mtx[ket, bra] = sp.sqrt(coeff)
            self.symbols = coeff