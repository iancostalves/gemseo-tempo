import matplotlib.pyplot as plt
from tempo_docdata import load

from gemseo_tempo.benchmarking.differentiation import plot_compile_times

study = load("albatross/differentiation.json")
panels = [key for key in ("inputs", "outputs") if key in study]
_, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 3.8), squeeze=False)
for axis, key in zip(axes[0], panels, strict=True):
    plot_compile_times(study[key], f"n_{key}", axes=axis)
plt.tight_layout()
