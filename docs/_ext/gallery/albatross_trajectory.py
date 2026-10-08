from tempo_docdata import load

from gemseo_tempo.problems.albatross.plots import plot_trajectory

plot_trajectory(load("albatross/cycle_codesign_dense.npz"))
