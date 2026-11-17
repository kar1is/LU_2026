"""Block-tridiagonal solvers for the Floquet system (CPU and GPU versions).
They only take plain arrays/numbers, so they don't depend on any other file."""
import numpy as np
import cupy as cp


def solve_block_tridiagonal(D0_base, L_sparse, U_block, omega_rf_num,
                            r, n_dim, block_dim, k):
    diag_idx = np.arange(block_dim)
    D_blocks = []
    for row in range(n_dim):
        m = row - k
        Dm = D0_base.copy()
        Dm[diag_idx, diag_idx] -= 1j * m * omega_rf_num
        D_blocks.append(Dm)

    c_prime = [None]*n_dim
    d_prime = [None]*n_dim
    r_blocks = [r[i*block_dim:(i+1)*block_dim] for i in range(n_dim)]

    rhs0 = np.hstack([U_block, r_blocks[0][:, None]])
    sol0 = np.linalg.solve(D_blocks[0], rhs0)
    c_prime[0] = sol0[:, :-1]
    d_prime[0] = sol0[:, -1]

    for i in range(1, n_dim):
        Lc = L_sparse @ c_prime[i-1]       # sparse @ dense — fast
        Ld = L_sparse @ d_prime[i-1]       # sparse @ dense (vector) — fast

        denom = D_blocks[i] - Lc
        rhs_vec = r_blocks[i] - Ld

        if i < n_dim - 1:
            rhs = np.hstack([U_block, rhs_vec[:, None]])
            sol = np.linalg.solve(denom, rhs)
            c_prime[i] = sol[:, :-1]
            d_prime[i] = sol[:, -1]
        else:
            d_prime[i] = np.linalg.solve(denom, rhs_vec)

    x_blocks = [None]*n_dim
    x_blocks[-1] = d_prime[-1]
    for i in range(n_dim-2, -1, -1):
        x_blocks[i] = d_prime[i] - c_prime[i] @ x_blocks[i+1]

    return np.concatenate(x_blocks)


def solve_block_tridiagonal_gpu(D0_batch, L_gpu, U_gpu, omega_rf_num,
                                r_gpu_blocks, n_dim, block_dim, k):
    """
    Solve a batch of independent Floquet block-tridiagonal systems.

    D0_batch: (batch, block_dim, block_dim)
    L_gpu:    (block_dim, block_dim)
    U_gpu:    (block_dim, block_dim)

    Returns:
        (batch, n_dim * block_dim) GPU array
    """
    batch = D0_batch.shape[0]
    diag_idx = cp.arange(block_dim)

    c_prime = [None] * n_dim
    d_prime = [None] * n_dim

    for i in range(n_dim):
        m = i - k

        # Each Floquet diagonal block is D0 - i*m*omega*I.
        Dm = D0_batch.copy()
        Dm[:, diag_idx, diag_idx] -= 1j * m * omega_rf_num

        if i == 0:
            denom = Dm
            rhs_vec = cp.broadcast_to(r_gpu_blocks[0], (batch, block_dim))
        else:
            Lc = cp.matmul(L_gpu[None, :, :], c_prime[i - 1])
            Ld = cp.matmul(L_gpu[None, :, :],
                           d_prime[i - 1][:, :, None])[:, :, 0]

            denom = Dm - Lc
            rhs_vec = r_gpu_blocks[i][None, :] - Ld

        # Solve for the next coupling matrix and the RHS in one batched solve.
        rhs = cp.concatenate(
            (cp.broadcast_to(U_gpu, (batch, block_dim, block_dim)),
             rhs_vec[:, :, None]),
            axis=2,
        )

        sol = cp.linalg.solve(denom, rhs)

        if i < n_dim - 1:
            c_prime[i] = sol[:, :, :block_dim]
            d_prime[i] = sol[:, :, block_dim]
        else:
            d_prime[i] = sol[:, :, block_dim]

    # Back substitution.
    x_blocks = [None] * n_dim
    x_blocks[-1] = d_prime[-1]

    for i in range(n_dim - 2, -1, -1):
        x_blocks[i] = (
            d_prime[i]
            - cp.matmul(c_prime[i], x_blocks[i + 1][:, :, None])[:, :, 0]
        )

    return cp.concatenate(x_blocks, axis=1)
