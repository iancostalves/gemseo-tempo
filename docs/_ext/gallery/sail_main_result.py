from tempo_docdata import load

from gemseo_tempo.problems.sail.plots import plot_tracks

downwind = load("sail/downwind.json")
track = load("sail/track_downwind.npz")
plot_tracks({"single shooting": (track["x"], track["y"])}, mark=downwind["mark"])
