"""Builds everything that does NOT depend on B0 / BRF sweeps:
atom, states, Lindbladian (or loads it from cache), RHS vector, lambdified blocks.
The result is bundled into one `System` object that the other files receive."""
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import sympy as sp

from utilities import branching_rate, split_frequencies
from operators import Hamiltonian, Lindblad
from interactions import BField, EField
from rwa import RWA, optical_rotor
from densitymatrix import DensityMatrix
from atom import Atom
from parameters import build_num_values
from matrix_elements import electric_dipole_element, \
                            hyperfine_self_interaction

from config import (
    CACHE_FILE, E0_values, k, angles, states, symbols_map,
    ATOM_FILE, TRANSITION, GAMMA_TRANSIT_MHZ, OMEGA_RF_MHZ, OMEGA_R_MHZ,
    ωRF, ωR, B0, γ, Γ, BRF, E0,
)


@dataclass
class System:
    """Everything the solvers and the sweep need."""
    rho: DensityMatrix
    num_values: dict
    dipole_table: np.ndarray
    r: np.ndarray
    k: int
    n_dim: int
    block_dim: int
    state_dim: int
    order_set: set
    nonzero_positions: dict
    nonzero_idx_arrays: dict
    A_func_batched: dict


def _compute_dipole_table(rho, num_values):
    probe_field = EField(amplitude=E0_values[0], frequency=0,
                         theta=sp.pi/2, phi=0)
    table = np.zeros((len(rho.excited_states), len(rho.ground_states)),
                     dtype=np.complex128)
    for i, exc in enumerate(rho.excited_states):
        for j, gnd in enumerate(rho.ground_states):
            val = electric_dipole_element(probe_field, exc, gnd)
            table[i, j] = complex(sp.N(val))
    return table


def _load_or_build_lindbladian(rho, num_values):
    """Returns {Floquet order n: sympy matrix with numbers substituted}."""
    if os.path.exists(CACHE_FILE):
        print("Loading cached Lindbladian setup...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    H = Hamiltonian(rho)

    e_field = EField(amplitude=E0, frequency=ωR,
                     theta=angles['E_theta'], phi=angles['E_phi'])
    b_rf_field = BField(amplitude=BRF, frequency=ωRF,
                        theta=angles['B_RF_theta'], phi=angles['B_RF_phi'])
    b_zeeman_field = BField(amplitude=B0, frequency=0)

    H.add_field(e_field)
    H.add_field(b_rf_field)
    H.add_field(b_zeeman_field)

    # Optical RWA
    rotor = optical_rotor(rho, ωR)
    H = RWA(H, rotor, ωR)

    print(f"Hamiltonian creation took {time.time()-t0:.1f}s")
    t0 = time.time()

    L = Lindblad(rho, H)
    for exc in rho.excited_states:
        total = sum(branching_rate(exc, gnd) for gnd in rho.ground_states)
        for gnd in rho.ground_states:
            rate = branching_rate(exc, gnd) / total
            L.spontaneous_decay(exc, gnd, rate, Γ)

    L.add_transit(γ)           # transit loss
    A = L.get_coeff()

    print(f"Lindblad coefficients took {time.time()-t0:.1f}s")

    freq_mtx = split_frequencies(A, ωRF)
    frequency_dict = {n: mtx.subs(num_values) for n, mtx in freq_mtx.items()}

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(frequency_dict, f)

    print('Done building.')
    return frequency_dict


def _build_rhs(rho, num_values):
    """Right-hand side vector: transit repopulation of the ground states,
    placed in the central (order 0) Floquet block."""
    r = sp.zeros((2*k + 1) * rho.len**2, 1)
    b = sp.zeros(rho.len**2, 1)

    for state in rho.ground_states:
        i = rho.states.index(state)
        b[i*rho.len + i] -= num_values[γ] / len(rho.ground_states)

    r[k*rho.len**2:(k+1)*rho.len**2, 0] = b
    return np.array(r, dtype=np.complex128).flatten()


def _lambdify_orders(frequency_dict):
    """Find nonzero entries symbolically (ONCE), then lambdify each order."""
    nonzero_positions = {}
    A_func_batched = {}

    for n, mtx in frequency_dict.items():
        rows, cols = mtx.shape
        positions = [(i, j) for i in range(rows) for j in range(cols)
                     if mtx[i, j] != 0]
        nonzero_positions[n] = positions
        exprs = [mtx[i, j] for (i, j) in positions]
        A_func_batched[n] = sp.lambdify((B0, BRF, E0), exprs,
                                        modules="numpy", cse=True)

    nonzero_idx_arrays = {
        order: (np.array([p[0] for p in positions]),
                np.array([p[1] for p in positions]))
        for order, positions in nonzero_positions.items()}

    return nonzero_positions, nonzero_idx_arrays, A_func_batched

def _hyperfine_offset_MHz(ground, excited):
    """Shift of the excited F manifold minus the ground F manifold, in MHz."""
    g = ground.sublevels()[0]
    e = excited.sublevels()[0]
    shift_g = hyperfine_self_interaction(g, g) / (2 * sp.pi)
    shift_e = hyperfine_self_interaction(e, e) / (2 * sp.pi)
    return float(shift_e - shift_g)

def build_system():
    t0 = time.time()
    print('Starting..')

    cs_atom = Atom(ATOM_FILE)

    ground = cs_atom.atomic_level(
        states['ground_level']).manifold(states['ground_manifold'])
    excited = cs_atom.atomic_level(
        states['excited_level']).manifold(states['excited_manifold'])
    
    num_values = build_num_values(
        cs_atom,
        transition_name=TRANSITION,
        symbols=symbols_map,
        gamma_transit_MHz=GAMMA_TRANSIT_MHZ,
        omega_RF_MHz=OMEGA_RF_MHZ,
        omega_R_MHz=OMEGA_R_MHZ + _hyperfine_offset_MHz(ground, excited),
        E1=0,
        E2=0,
    )

    rho = DensityMatrix(ground=ground, excited=excited)

    dipole_table = _compute_dipole_table(rho, num_values)
    frequency_dict = _load_or_build_lindbladian(rho, num_values)
    r = _build_rhs(rho, num_values)

    nonzero_positions, nonzero_idx_arrays, A_func_batched = \
        _lambdify_orders(frequency_dict)

    print(f"System setup took {time.time()-t0:.1f}s")

    return System(
        rho=rho,
        num_values=num_values,
        dipole_table=dipole_table,
        r=r,
        k=k,
        n_dim=2*k + 1,
        block_dim=rho.len**2,
        state_dim=rho.len,
        order_set=set(A_func_batched.keys()),
        nonzero_positions=nonzero_positions,
        nonzero_idx_arrays=nonzero_idx_arrays,
        A_func_batched=A_func_batched,
    )
