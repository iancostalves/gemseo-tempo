from tempo_docdata import load

from gemseo_tempo.benchmarking.differentiation import plot_mode_radar

records = load("brachistochrone/differentiation.json")["inputs"]
plot_mode_radar([r for r in records if r["jit"] and r["n_inputs"] == 22])
