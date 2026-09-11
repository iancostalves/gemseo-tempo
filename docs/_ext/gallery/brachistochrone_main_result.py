from tempo_docdata import load

from gemseo_tempo.problems.brachistochrone.plots import plot_descent

cycloid = load("brachistochrone/cycloid.npz")
paths = {
    variant: load(f"brachistochrone/variant_{variant}.npz")
    for variant in ("ideal", "friction", "drag")
}
plot_descent(
    {variant: (arrays["x"], arrays["y"]) for variant, arrays in paths.items()},
    cycloid=(cycloid["x"], cycloid["y"]),
)
