"""
Splitting of the 1.45 G (one-photon) and 2.9 G (two-photon) features, measured on the
steady-state absorption signal of YOUR Floquet-Lindblad model, computed with YOUR solver
(compute_result_rf_gpu / compute_result_rf).  Nothing of yours is modified.

For every B_RF:
  1. one batched solve on a B0 grid around the resonance (1 call),
  2. the two most prominent features are located,
  3. one more batched solve on a fine grid around each of them, parabola vertex -> position
     with sub-grid precision (1 call).
Their separation (in gauss) is the splitting.  Output: log-log splitting vs B_RF plus the
local exponent d ln S / d ln B_RF  (1 = linear, 2 = quadratic).

Usage:   python splitting_from_lindbladian.py          # both resonances
         python splitting_from_lindbladian.py 2        # only the 2.9 G one
Results are cached after every B_RF (cache_splitting.pkl); a stopped run resumes.
"""
import importlib
import os
import pickle
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ============================ EDIT THESE ============================
SETUP_MODULE = "setup_system"        # file (without .py) that defines build_system()  (the one you pasted first)
SWEEP_MODULE = "sweep"         # file (without .py) that defines compute_result_rf_gpu  (the one you just pasted)
USE_GPU = True                 # False -> compute_result_rf (joblib, CPU)
B_CENTER = {1: 1.45, 2: 2.90}  # nominal resonance fields [G]
KIND = "peak"                  # track "peak" (maxima of the signal) or "dip" (minima)
B_RF_GRID = np.geomspace(0.05, 3.0, 32)   # [G]
N_SCAN = 161                   # B0 points in the coarse scan (one batched call)
N_FINE = 41                    # B0 points around each feature in the refinement call
HALF0, HALF1 = 0.20, 0.40      # scan half-width = HALF0 + HALF1*B_RF  [G]
PROM_FRAC = 0.02               # a feature needs prominence > this * (max-min of the scan)
E0_VALUE = 0.04                     # None -> E0_values[0] from config
CACHE = "cache_splitting.pkl"
FIG = "splitting_from_lindbladian.png"
# ====================================================================


def find_two(Bs, y, kind):
    """Indices of the two most prominent extrema of a scan (None if fewer than two)."""
    sgn = 1 if kind == "peak" else -1
    idx, pr = find_peaks(sgn * y, prominence=PROM_FRAC * np.ptp(y))
    if len(idx) < 2:
        return None
    return np.sort(idx[np.argsort(pr["prominences"])[::-1][:2]])


def vertex(x, y, sgn, half=3):
    """Sub-grid extremum: parabola through the points around the best sample."""
    i = int(np.argmax(sgn * y))
    lo, hi = max(i - half, 0), min(i + half + 1, len(x))
    c = np.polyfit(x[lo:hi] - x[i], y[lo:hi], 2)
    if c[0] == 0:
        return x[i]
    v = x[i] - c[1] / (2 * c[0])
    return v if x[lo] <= v <= x[hi - 1] else x[i]


def measure(signal, q, BRF, kind=KIND):
    """signal(B0_array) -> array of absorption signals (one batched solve)."""
    Bc, half = B_CENTER[q], HALF0 + HALF1 * BRF
    Bs = np.linspace(max(Bc - half, 0.12), min(Bc + half, 4.95), N_SCAN)
    y = signal(Bs)
    sel = find_two(Bs, y, kind)
    if sel is None:
        return dict(B1=np.nan, B2=np.nan, S=np.nan, Bs=Bs, y=y)
    h, sgn = Bs[1] - Bs[0], (1 if kind == "peak" else -1)
    fine = [np.linspace(Bs[i] - 1.5 * h, Bs[i] + 1.5 * h, N_FINE) for i in sel]
    yf = np.asarray(signal(np.concatenate(fine))).reshape(2, N_FINE)
    pos = [vertex(fine[j], yf[j], sgn) for j in range(2)]
    return dict(B1=pos[0], B2=pos[1], S=pos[1] - pos[0], Bs=Bs, y=y)


def local_exponent(x, S, w=5):
    p = np.full(len(x), np.nan)
    for i in range(len(x) - w + 1):
        sl = slice(i, i + w)
        if np.all(np.isfinite(S[sl])) and np.all(S[sl] > 0):
            p[i + w // 2] = np.polyfit(np.log(x[sl]), np.log(S[sl]), 1)[0]
    return p


def make_plot(res, orders):
    fig, ax = plt.subplots(len(orders), 3, figsize=(16, 4.4 * len(orders)), squeeze=False)
    for r, q in enumerate(orders):
        x = np.array(sorted(k[1] for k in res if k[0] == q))
        R = [res[(q, b)] for b in x]
        S = np.array([d["S"] for d in R])
        grid = np.linspace(B_CENTER[q] - 1, B_CENTER[q] + 1, 200)
        Y = np.array([np.interp(grid, d["Bs"], d["y"]) for d in R])
        a = ax[r, 0]
        a.pcolormesh(grid, x, Y, shading="auto")
        a.plot([d["B1"] for d in R], x, "w.", ms=4)
        a.plot([d["B2"] for d in R], x, "r.", ms=4)
        a.set_yscale("log"); a.set_xlabel("B0 [G]"); a.set_ylabel("B_RF [G]")
        a.set_title(f"signal and tracked features, resonance ~{B_CENTER[q]} G\n"
                    f"(CHECK that the dots sit on the two features you mean)", fontsize=9)
        ok = np.isfinite(S)
        b = ax[r, 1]
        b.loglog(x[ok], S[ok], "o-", ms=3.5)
        if ok.any():
            x0, S0 = x[ok][0], S[ok][0]
            for pw, ls in ((1, "--"), (2, ":")):
                b.loglog(x, S0 * (x / x0) ** pw, "k", ls=ls, lw=0.8, label=f"slope {pw}")
            b.legend()
        b.set_xlabel("B_RF [G]"); b.set_ylabel("splitting of the two features [G]")
        b.set_title(f"{q}-photon: splitting (unresolved points omitted)")
        b.grid(alpha=0.3, which="both")
        c = ax[r, 2]
        c.semilogx(x, local_exponent(x, S), "o-", ms=3.5)
        c.axhline(1, c="k", ls="--", lw=0.8); c.axhline(2, c="k", ls=":", lw=0.8)
        c.set_xlabel("B_RF [G]"); c.set_ylabel("local exponent d ln S / d ln B_RF")
        c.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG, dpi=140); print("saved", FIG)


def main():
    orders = tuple(int(a) for a in sys.argv[1:]) or (1, 2)
    build_system = importlib.import_module(SETUP_MODULE).build_system
    sweep = importlib.import_module(SWEEP_MODULE)
    compute = sweep.compute_result_rf_gpu if USE_GPU else sweep.compute_result_rf
    from config import E0_values

    S = build_system()
    E0 = E0_values[0] if E0_VALUE is None else E0_VALUE
    print(f"E0 = {E0}, Floquet order k = {S.k}, block_dim = {S.block_dim}")

    def make_signal(BRF):
        return lambda Bs: np.asarray(compute(S, BRF, E0, np.asarray(Bs, dtype=float))[0])

    res = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    todo = [(q, float(b)) for q in orders for b in B_RF_GRID if (q, float(b)) not in res]
    print(f"{len(todo)} (resonance, B_RF) points to do; "
          f"~{N_SCAN + 2 * N_FINE} solves each")
    t_start = time.time()
    for n, (q, b) in enumerate(todo, 1):
        t0 = time.time()
        res[(q, b)] = measure(make_signal(b), q, b)
        pickle.dump(res, open(CACHE, "wb"))
        d = res[(q, b)]
        eta = (time.time() - t_start) / n * (len(todo) - n)
        print(f"order {q}  B_RF={b:6.3f}: "
              + (f"S = {d['S']:.4f} G" if np.isfinite(d["S"]) else "unresolved")
              + f"   ({time.time() - t0:.1f} s, ETA {eta / 60:.1f} min)", flush=True)
    make_plot(res, orders)


if __name__ == "__main__":
    main()
