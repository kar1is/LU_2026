"""Runs the solvers over the B0 grid for a given BRF (CPU-parallel or GPU)."""
import numpy as np
import cupy as cp
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix

from absorption import absorption_fast
from config import BATCH_SIZE, Γ, ωRF
from floquet import dense_from_nonzero, get_block_standalone
from solvers import solve_block_tridiagonal, solve_block_tridiagonal_gpu


def _evaluate_orders(system, BRF_value, E0_value, B0_values):
    """vals_by_order[order] has shape (n_nonzero_for_order, N)."""
    N = len(B0_values)
    vals_by_order = {}
    for order in system.order_set:
        raw = system.A_func_batched[order](B0_values, BRF_value, E0_value)
        vals_by_order[order] = np.array(
            [np.broadcast_to(v, (N,)) for v in raw], dtype=np.complex128)
    return vals_by_order


# ------------------------------- CPU ---------------------------------------
def solve_one_B0(idx, D0_values, idx0, L_sparse, U_block, omega_rf_num, r,
                 n_dim, block_dim, k, state_dim, dipole_table, gamma_big):
    """Solve a single B0 point. Takes only plain arrays so joblib can ship it
    to worker processes cheaply."""
    D0_base = dense_from_nonzero(block_dim, idx0, D0_values[:, idx])
    solution = solve_block_tridiagonal(D0_base, L_sparse, U_block,
                                       omega_rf_num, r, n_dim, block_dim, k)
    rho_tmp = solution[k*state_dim**2:(k+1)*state_dim**2]
    rho0 = rho_tmp.reshape(state_dim, state_dim)
    return absorption_fast(dipole_table, rho0, gamma_big)


def compute_result_rf(system, BRF_value, E0_value, B0_values, n_jobs=20):
    omega_rf_num = system.num_values[ωRF]
    N = len(B0_values)
    vals_by_order = _evaluate_orders(system, BRF_value, E0_value, B0_values)

    # Any column works for L and U (they are B0-independent).
    L_block = get_block_standalone(system, -1, {-1: vals_by_order[-1][:, 0]})
    U_block = get_block_standalone(system, 1, {1: vals_by_order[1][:, 0]})
    L_sparse = csr_matrix(L_block)

    idx0 = system.nonzero_idx_arrays[0]
    gamma_big = system.num_values[Γ]

    exc_populations = Parallel(n_jobs=n_jobs)(
        delayed(solve_one_B0)(idx, vals_by_order[0], idx0, L_sparse, U_block,
                              omega_rf_num, system.r, system.n_dim,
                              system.block_dim, system.k, system.state_dim,
                              system.dipole_table, gamma_big)
        for idx in range(N)
    )
    return np.array(exc_populations), BRF_value


# ------------------------------- GPU ---------------------------------------
def compute_result_rf_gpu(system, BRF_value, E0_value, B0_values,
                          batch_size=BATCH_SIZE):
    N = len(B0_values)
    omega_rf_num = system.num_values[ωRF]
    block_dim, n_dim, k = system.block_dim, system.n_dim, system.k
    state_dim = system.state_dim

    # Symbolic expressions are evaluated on the CPU.
    vals_by_order = _evaluate_orders(system, BRF_value, E0_value, B0_values)

    # These coupling blocks are assumed independent of B0.
    L_block = get_block_standalone(system, -1, {-1: vals_by_order[-1][:, 0]})
    U_block = get_block_standalone(system, 1, {1: vals_by_order[1][:, 0]})

    L_gpu = cp.asarray(L_block)
    U_gpu = cp.asarray(U_block)

    r_gpu = cp.asarray(system.r)
    r_gpu_blocks = [r_gpu[i * block_dim:(i + 1) * block_dim]
                    for i in range(n_dim)]

    ii, jj = system.nonzero_idx_arrays[0]
    ii_gpu = cp.asarray(ii)
    jj_gpu = cp.asarray(jj)

    exc_populations = np.empty(N, dtype=np.float64)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        current_batch = end - start

        # Construct only this batch of dense D0 blocks.
        D0_batch = cp.zeros((current_batch, block_dim, block_dim),
                            dtype=cp.complex128)
        values = cp.asarray(vals_by_order[0][:, start:end].T)
        D0_batch[:, ii_gpu, jj_gpu] = values

        solutions = solve_block_tridiagonal_gpu(
            D0_batch, L_gpu, U_gpu, omega_rf_num,
            r_gpu_blocks, n_dim, block_dim, k)

        # Keep only the central Floquet order and move it back to the CPU.
        central = solutions[:, k * block_dim:(k + 1) * block_dim]
        rho_batch = cp.asnumpy(central).reshape(
            current_batch, state_dim, state_dim)

        for j in range(current_batch):
            exc_populations[start + j] = absorption_fast(
                system.dipole_table, rho_batch[j], system.num_values[Γ])

        del D0_batch, values, solutions, central, rho_batch

    return exc_populations, BRF_value
