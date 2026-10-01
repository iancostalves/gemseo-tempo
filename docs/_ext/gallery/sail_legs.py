from matplotlib import pyplot
from tempo_docdata import load

from gemseo_tempo.problems.sail.plots import plot_tracks
from gemseo_tempo.problems.sail.problem import CORRIDOR_HALF_WIDTH

_, axes = pyplot.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
for axis, leg in zip(axes, ("downwind", "upwind"), strict=True):
    track = load(f"sail/track_{leg}.npz")
    plot_tracks(
        {leg: (track["x"], track["y"])},
        axes=axis,
        mark=load(f"sail/{leg}.json")["mark"],
        channel=CORRIDOR_HALF_WIDTH,
    )
    axis.set_title(f"{leg.capitalize()} leg, inside the race area")
