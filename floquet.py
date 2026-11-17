"""Helpers for turning the nonzero entries of each Floquet order into
dense blocks, or into one big sparse Floquet matrix."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix


def dense_from_nonzero(block_dim, idx_arrays, values):
    """Dense (block_dim, block_dim) array with `values` placed at idx_arrays."""
    dense = np.zeros((block_dim, block_dim), dtype=np.complex128)
    ii, jj = idx_arrays
    dense[ii, jj] = values
    return dense


def get_block_standalone(system, order, block_data_dict):
    """Dense block for a single Floquet order.
    block_data_dict = {order: flat array of nonzero values}."""
    if order not in block_data_dict:
        return np.zeros((system.block_dim, system.block_dim),
                        dtype=np.complex128)
    return dense_from_nonzero(system.block_dim,
                              system.nonzero_idx_arrays[order],
                              block_data_dict[order])


def eval_block_for_all_B0(system, order, B0_values, BRF_value, E0_value):
    """Array of shape (block_dim, block_dim, N). Only nonzero entries are
    computed; everything else stays zero."""
    N = len(B0_values)
    out = np.zeros((system.block_dim, system.block_dim, N),
                   dtype=np.complex128)
    vals = system.A_func_batched[order](B0_values, BRF_value, E0_value)
    for (i, j), v in zip(system.nonzero_positions[order], vals):
        out[i, j, :] = np.broadcast_to(v, (N,))
    return out


@dataclass
class FloquetStructure:
    rows_idx: np.ndarray
    cols_idx: np.ndarray
    block_offsets: list          # (start, end, order) into the flat data array
    total_nnz: int
    diag_shift_pos: np.ndarray
    diag_shift_m: np.ndarray
    shape: tuple


def build_floquet_structure(system):
    """Precompute the sparsity structure using ONLY the nonzero entries."""
    n_dim, block_dim, k = system.n_dim, system.block_dim, system.k
    order_set = system.order_set

    block_positions = [(row, col, row - col)
                       for row in range(n_dim) for col in range(n_dim)
                       if (row - col) in order_set]

    rows_idx_list, cols_idx_list = [], []
    block_offsets = []
    diag_shift_pos_list, diag_shift_m_list = [], []

    cursor = 0
    for (row, col, order) in block_positions:
        positions = system.nonzero_positions[order]
        n_entries = len(positions)
        local_i = np.array([p[0] for p in positions])
        local_j = np.array([p[1] for p in positions])

        rows_idx_list.append(row*block_dim + local_i)
        cols_idx_list.append(col*block_dim + local_j)
        block_offsets.append((cursor, cursor + n_entries, order))

        if order == 0:
            diag_mask = local_i == local_j
            diag_shift_pos_list.append(cursor + np.where(diag_mask)[0])
            diag_shift_m_list.append(np.full(diag_mask.sum(), row - k))

        cursor += n_entries

    return FloquetStructure(
        rows_idx=np.concatenate(rows_idx_list),
        cols_idx=np.concatenate(cols_idx_list),
        block_offsets=block_offsets,
        total_nnz=cursor,
        diag_shift_pos=(np.concatenate(diag_shift_pos_list)
                        if diag_shift_pos_list else np.array([], dtype=int)),
        diag_shift_m=(np.concatenate(diag_shift_m_list)
                      if diag_shift_m_list else np.array([])),
        shape=(n_dim*block_dim, n_dim*block_dim),
    )


def build_floquet_matrix_fast(structure, block_data_dict, omega_rf_num):
    """block_data_dict: {order: flat array of nonzero values, matching
    system.nonzero_positions[order] ordering}"""
    data = np.empty(structure.total_nnz, dtype=np.complex128)
    for start, end, order in structure.block_offsets:
        data[start:end] = block_data_dict[order]
    if len(structure.diag_shift_pos):
        data[structure.diag_shift_pos] -= \
            1j * structure.diag_shift_m * omega_rf_num
    return coo_matrix((data, (structure.rows_idx, structure.cols_idx)),
                      shape=structure.shape).tocsr()
