from tempo_docdata import load

from gemseo_tempo.problems.albatross.plots import plot_frames

plot_frames(load("albatross/cycle_codesign_dense.npz"))
