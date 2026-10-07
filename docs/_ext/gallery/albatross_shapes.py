from tempo_docdata import load

from gemseo_tempo.problems.albatross.plots import plot_maneuver_pattern

plot_maneuver_pattern(
    {
        "O, optimized": load("albatross/cycle_loop_dense.npz"),
        "S, optimized": load("albatross/cycle_weave_dense.npz"),
    }
)
