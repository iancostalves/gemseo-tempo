from tempo_docdata import load

from gemseo_tempo.benchmarking.differentiation import plot_stage_modes

study = load("albatross/chain_modes.json")
plot_stage_modes(study["chains"], study["fd"])
