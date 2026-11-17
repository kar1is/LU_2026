"""Plots, the GIF animation, and the peak-splitting analysis."""
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.signal import argrelextrema

from config import BRF_values, B0_values, angles, states


def make_animation_and_splittings(results, gif_name="animation_mixed.gif"):
    """Draws one frame per BRF value, saves them as a GIF, and collects how far
    neighbouring peaks are apart in two B0 windows.

    Returns (peak_splitting_145, peak_splitting_290) as arrays of
    [BRF_value, splitting]."""
    colors = plt.colormaps['Reds'](np.linspace(0.2, 1, len(BRF_values)))

    peak_splitting_145 = []
    peak_splitting_290 = []
    frames = []
    
    plot_spectra(results)

    for (exc_pop, BRF_value), color in zip(results, colors):
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.set_title(
            rf'$E \angle {{{angles["E_theta"]/np.pi*90:0.1f}}}$, '
            rf'$F={{{states["ground_manifold"]}}} \Leftrightarrow '
            rf'F={{{states["excited_manifold"]}}}$ '
            rf'$B_{{RF}}={{{BRF_value:0.1f}}}$'
        )
        ax.label_outer()
        ax.set_box_aspect(0.5)

        handles, labels = ax.get_legend_handles_labels()
        fig.subplots_adjust(right=0.75)
        fig.legend(handles, labels,
                   loc="center left",
                   bbox_to_anchor=(0.75, 0.5),
                   ncol=1)
        ax.set_xlabel('Magnētiskā lauka intensitāte [G]')
        ax.set_ylabel('Absorbcijas signāls [a.u.]')
        #ax.set_ylim(4.7, 5.65)

        peak_indeces = argrelextrema(exc_pop, np.greater)[0]
        for i in range(len(peak_indeces)):
            this_B = B0_values[peak_indeces[i]]
            prev_B = B0_values[peak_indeces[i-1]]

            if 1 <= this_B <= 2 and 1 <= prev_B <= 2:
                peak_splitting_145.append([BRF_value, abs(this_B - prev_B)])

            if 2 <= this_B <= 3.5 and 2 <= prev_B <= 3.5:
                peak_splitting_290.append([BRF_value, abs(this_B - prev_B)])

        ax.plot(B0_values, exc_pop, color=color)

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()))
        plt.close(fig)

    imageio.mimsave(gif_name, frames, duration=0.3, loop=0)

    return np.array(peak_splitting_145), np.array(peak_splitting_290)

def plot_spectra(results):
    colors = plt.colormaps['Reds'](np.linspace(0.2, 1, len(BRF_values)))
    plt.figure(figsize=(8, 4))
    for (exc_pop, BRF_value), color in zip(results, colors):
        plt.plot(B0_values, exc_pop, label=f'BRF = {BRF_value:.1f}', color=color)
    plt.xlabel('Magnētiskā lauka intensitāte [G]')
    plt.ylabel('Absorbcijas signāls [a.u.]')
    # plt.legend()
    plt.show()
    
def plot_spectra_3d(results, kind='surface'):
    BRFs = np.array([r[1] for r in results])
    Z = np.array([r[0] for r in results])          # shape (n_BRF, n_B0)
    B, R = np.meshgrid(B0_values, BRFs)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(projection='3d')

    if kind == 'surface':
        ax.plot_surface(B, R, Z, cmap='Reds', linewidth=0, antialiased=True)
    else:   # waterfall of lines
        colors = plt.colormaps['Reds'](np.linspace(0.2, 1, len(BRFs)))
        for (exc_pop, BRF_value), color in zip(results, colors):
            ax.plot(B0_values, np.full_like(B0_values, BRF_value),
                    exc_pop, color=color)

    ax.set_xlabel('Magnētiskā lauka intensitāte [G]')
    ax.set_ylabel('$B_{RF}$')
    ax.set_zlabel('Absorbcijas signāls [a.u.]')
    ax.view_init(elev=30, azim=-60)     # change these to rotate the view
    plt.show()

def plot_splitting(peak_splitting, title='2.90 šķelšanās'):
    if len(peak_splitting) == 0:
        print(f"No peak pairs found for '{title}', nothing to plot.")
        return
    plt.figure()
    plt.scatter(peak_splitting[:, 0], peak_splitting[:, 1])
    plt.title(title)
    plt.show()
