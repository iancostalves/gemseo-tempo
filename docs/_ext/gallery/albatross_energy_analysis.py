from tempo_docdata import load

from gemseo_tempo.problems.albatross.plots import plot_energy_analysis

plot_energy_analysis(load("albatross/cycle_codesign_dense.npz"))
