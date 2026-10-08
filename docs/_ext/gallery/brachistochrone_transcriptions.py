from tempo_docdata import load

from gemseo_tempo.benchmarking.plots import plot_transcriptions

plot_transcriptions(load("brachistochrone/transcriptions.json")["rows"])
