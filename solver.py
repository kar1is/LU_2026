import utilities as ut
from operators import Hamiltonian, Lindblad, Project
from interactions import BField, EField
from rwa import RWA
from densitymatrix import DensityMatrix
from symbols import t

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True)

omegaB_RF = sp.symbols('\u03A9RF', positive=True)
omegaB_RF_small = sp.symbols('\u03C9RF', positive=True)
omega_laser_small = sp.symbols('\u03C9L', positive=True)
B0 = sp.symbols('B0')
rate_trans_rlx = sp.symbols('\u03B3', positive=True)
rate_spont_rlx = sp.symbols('\u0393', positive=True)

num_values = {
    rate_trans_rlx: 2*np.pi*8e-3, #gamma
    rate_spont_rlx: 2*np.pi*6.1, #Gamma
    sp.symbols('E1'): 0,
    sp.symbols('E2'): 0,
    sp.symbols('E3'): 0,
    sp.symbols('E4'): 0,
    omegaB_RF_small: 2*np.pi*1,
    omegaB_RF: 2*np.pi*5,
    omega_laser_small: 0,
    sp.symbols('\u03A9', positive=True): 2*np.pi*10,
}

# Create density matrix
ground_states = ut.create_states(n=6, L=0, F=1)
excited_states = ut.create_states(n=6, L=1, F=0)
total_states = ground_states + excited_states
rho = DensityMatrix(total_states)

# Create interaction fields
efield = EField(1, omega_laser_small, 'pi')
# efield = EField(1, omega_laser_small, 'x') # for E \perp B conf
bfield_rf = BField(1, omegaB_RF_small, 'x')
bfield_z = BField(1, 0, 'pi')
bfield_z_ampl = np.linspace(-1e2, 1e2, 100)

H = Hamiltonian(rho)
H.add_field(efield)
H.add_field(bfield_z)
H.add_field(bfield_rf)

rotor1 = sp.eye(rho.len)
rotor1[0,0] = sp.srepr(sp.exp(sp.I * omegaB_RF_small * t))
H = RWA(H, rotor1, omegaB_RF_small)

rotor2 = sp.eye(rho.len)
rotor2[2,2] = sp.srepr(sp.exp(-sp.I * omegaB_RF_small * t))
H = RWA(H, rotor2, omegaB_RF_small)

rotor3 = sp.eye(rho.len)
rotor3[3,3] = sp.srepr(sp.exp(-sp.I * omega_laser_small * t))
H = RWA(H, rotor3, omega_laser_small)

lindbladian = Lindblad(H)

relax = [] # Symbolic jump operators
for i in range(rho.len-1):
    tmp_op = Project(rho, i, -1, rate_spont_rlx)
    lindbladian.add_jump(tmp_op)
    tmp_op = Project(rho, i, i, rate_trans_rlx)
    relax.append(tmp_op)
    lindbladian.add_jump(tmp_op)

coeff = lindbladian.get_coeff()
for i in range(rho.len**2):
    coeff[-1,i] = 0
for i in range(1, rho.len+1):
    coeff[-1,4*(i-1)+i-1] = 1

b = np.zeros(rho.len**2)
b[-1] = 1

exc_pop = []
for strength in bfield_z_ampl:
    tmp_coeff = np.array(coeff.subs(num_values).subs(B0, strength).evalf(), 
                      dtype=np.complex128)
    
    solution = np.linalg.solve(tmp_coeff, b)
    exc_pop.append(solution[-1])

plt.plot(bfield_z_ampl, exc_pop / np.max(exc_pop))
plt.show()