from matplotlib import pyplot

from gemseo_tempo.problems.albatross.problem import soaring_cycle

times, states, controls = soaring_cycle()
figure, axes = pyplot.subplots(2, 1, figsize=(6.4, 5.4), sharex=True)
axes[0].plot(times, states["h"], color="tab:blue")
axes[0].set_ylabel("height h (m)")
axes[0].set_title("The guessed cycle: height and heading over one period")
axes[1].plot(times, states["psi"], color="tab:orange")
axes[1].set_ylabel("heading psi (rad)")
axes[1].set_xlabel("time (s)")
for axis in axes:
    axis.grid(alpha=0.3)
