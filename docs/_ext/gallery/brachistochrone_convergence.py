from tempo_docdata import load

from gemseo_tempo.problems.brachistochrone.plots import plot_convergence

study = load("brachistochrone/convergence.json")
plot_convergence(study["sizes"], study["errors"])
