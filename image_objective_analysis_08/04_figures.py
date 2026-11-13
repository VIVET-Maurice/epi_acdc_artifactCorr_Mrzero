"""Write the figures of README.md as PNG files in figures/.

The drawing code is in plots.py, shared with image_objective_results.ipynb.
Run after 02_mechanisms.py, 03_optimise_variants.py and 05_compare_variants.py:
    python 04_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plots
from acdc08_twin import DEFAULT_RUN

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)


def main():
    arrays, scores = plots.load_results()
    notebook_history = np.load(DEFAULT_RUN / "mrzero08_optimization.npz")["relative_mse"]
    figures = {
        "fig1_images.png": plots.figure_images(arrays, scores),
        "fig2_time_structure.png": plots.figure_time(arrays),
        "fig3_local_echo.png": plots.figure_echo(arrays),
        "fig4_currents.png": plots.figure_currents(arrays),
        "fig5_variants.png": plots.figure_variants(arrays),
        "fig6_convergence.png": plots.figure_convergence(notebook_history),
    }
    for name, figure in figures.items():
        if figure is not None:
            figure.savefig(FIG / name)
            plt.close(figure)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
