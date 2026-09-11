from gemseo_tempo.problems.brachistochrone import Guess
from gemseo_tempo.problems.brachistochrone import variant_constants
from gemseo_tempo.problems.brachistochrone.plots import plot_descent
from gemseo_tempo.problems.brachistochrone.problem import simulate_guess

guess = Guess()
trajectory = simulate_guess(guess, variant_constants("ideal"))
plot_descent({"guessed": (trajectory.states["x"], trajectory.states["y"])})
