from tempo_docdata import load

from gemseo_tempo.problems.sail.plots import plot_channel_trade

channel = load("sail/channel.json")
widths = channel["widths"]
plot_channel_trade(
    [record["half_width"] for record in widths],
    [record["time"] for record in widths],
    [record["across"] for record in widths],
)
