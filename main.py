"""Entry point: build the system, sweep BRF x B0 on the GPU, plot.

Run with:  python main.py
(The __main__ guard matters: joblib worker processes would otherwise
re-run this whole file.)"""
import time

from config import BRF_values, E0_values, B0_values, BATCH_SIZE
from setup_system import build_system
from sweep import compute_result_rf_gpu   # or compute_result_rf for the CPU version
from plotting import make_animation_and_splittings, plot_splitting, \
                        plot_spectra_3d


def main():
    system = build_system()
 
    t0 = time.time()
    results = [compute_result_rf_gpu(system, BRF_value, E0_values[0],
                                     B0_values, batch_size=BATCH_SIZE)
               for BRF_value in BRF_values]
    print(f"GPU solve loop ({len(B0_values)} B0 points, "
          f"{len(BRF_values)} RF points) took {time.time() - t0:.1f}s")

    peak_splitting_145, peak_splitting_290 = \
        make_animation_and_splittings(results)
    
    plot_spectra_3d(results)
    print(results)
    print('splittings')
    print(peak_splitting_290) 
    plot_splitting(peak_splitting_290, '2.90 šķelšanās')
    # plot_splitting(peak_splitting_145, '1.45 šķelšanās')


if __name__ == "__main__":
    main()
