"""All settings and symbols in one place. Change numbers here, nowhere else."""
import numpy as np
import sympy as sp

# ---- Run settings ----
BATCH_SIZE = 64             # for computing on the GPU
CACHE_FILE = "lindblad_cache.pkl"

BRF_values = np.linspace(0.1, 3, 100)
E0_values = [2]
k = 9                                   # number of Floquet orders on each side

zeeman_field = {
    'min_amplitude': 0.1,
    'max_amplitude': 5,
    'no_points': 600,
}

angles = {
    'E_theta': sp.pi/2,
    'E_phi': 0,
    'B_RF_theta': sp.pi/4,
    'B_RF_phi': 0,
}

states = {
    'ground_level': '6S1/2',
    'ground_manifold': 4,
    'excited_level': '6P1/2',
    'excited_manifold': 4,
}

B0_values = np.linspace(zeeman_field['min_amplitude'],
                        zeeman_field['max_amplitude'],
                        zeeman_field['no_points'])

# ---- Apparatus parameters (used when building numerical values) ----
ATOM_FILE = "data/cs.json"
TRANSITION = "D1"
GAMMA_TRANSIT_MHZ = 0.015     # transit-time relaxation
OMEGA_RF_MHZ = 0.508          # RF drive frequency
OMEGA_R_MHZ = 0               # optical detuning reference

# ---- Symbols ----
ΩRF = sp.symbols('\u03A9RF', real=True)         # Rabi
ωRF = sp.symbols('\u03C9RF', positive=True)     # oscillation
ΩR = sp.symbols('\u03A9E', positive=True)       # Rabi
ωR = sp.symbols('\u03C9E', positive=True)       # oscillation
B0 = sp.symbols('B0', real=True)                # static field ampl
γ = sp.symbols('\u03B3', positive=True)         # gamma
Γ = sp.symbols('\u0393', positive=True)         # Gamma
BRF = sp.symbols('BRF', positive=True)
E0 = sp.symbols('E0', positive=True)

symbols_map = {"Gamma": Γ, "gamma": γ, "omega_RF": ωRF, "omega_R": ωR}
