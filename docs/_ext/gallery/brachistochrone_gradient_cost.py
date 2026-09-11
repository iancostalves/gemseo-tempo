from tempo_docdata import load

from gemseo_tempo.problems.brachistochrone.plots import plot_gradient_cost

records = load("brachistochrone/ad_vs_fd.json")
plot_gradient_cost(
    [record["n_design"] for record in records],
    [record["ad_seconds"] for record in records],
    [record["fd_seconds"] for record in records],
)
