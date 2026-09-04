# Copyright 2026 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Run the albatross problem and draw its figures.

Two runs. The first fixes the wind and asks how much energy a cycle can
take out of it, sweeping the wind strength to find where the cycle breaks
even. The second lets everything go, including the wing, and asks for the
weakest wind the bird could live in.

Usage::

    python examples/albatross/run_albatross.py --output figures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from matplotlib import pyplot

from gemseo_tempo.problems.albatross import create_codesign_optimization
from gemseo_tempo.problems.albatross import create_energy_optimization
from gemseo_tempo.problems.albatross import energy_budget
from gemseo_tempo.problems.albatross import optimal_trajectory
from gemseo_tempo.problems.albatross import plot_controls
from gemseo_tempo.problems.albatross import plot_energy
from gemseo_tempo.problems.albatross import plot_energy_analysis
from gemseo_tempo.problems.albatross import plot_polar
from gemseo_tempo.problems.albatross import plot_section_trade
from gemseo_tempo.problems.albatross import plot_trajectory
from gemseo_tempo.problems.albatross import plot_wind_profile
from gemseo_tempo.problems.albatross.sequential import sequential_codesign
from gemseo_tempo.scenario.description import describe_problem

WIND_SWEEP = (0.0, 8.0, 12.0, 16.0, 18.0, 22.0, 26.0)
"""The reference wind speeds the energy formulation is swept over, in m/s."""


def _save(figure, path: Path) -> Path:
    """Write a figure and close it.

    Parameters
    ----------
    figure
        The figure to write.
    path
        The destination file.

    Returns
    -------
    Path
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    pyplot.close(figure)
    return path


def run_energy_sweep(output: Path, size: int = 20) -> float:
    """Sweep the wind and report where the cycle breaks even.

    Parameters
    ----------
    output
        The directory to write figures into.
    size
        The discretization.

    Returns
    -------
    float
        The reference wind at which the cycle first gains energy, in m/s.
    """
    print(f"{'w_ref':>7} {'energy':>9} {'net':>9} {'drag':>9} {'gradient':>9}")
    neutral = float("nan")
    previous = None
    for wind in WIND_SWEEP:
        scenario, gemseo_scenario = create_energy_optimization(size=size, w_ref=wind)
        gemseo_scenario.execute(algo_name="SLSQP", max_iter=300)
        trajectory = optimal_trajectory(scenario, gemseo_scenario)
        budget = energy_budget(trajectory)
        net = budget["air_change"]
        print(
            f"{wind:7.1f} {-float(gemseo_scenario.optimization_result.f_opt):9.3f}"
            f" {net:+9.3f} {budget['drag_work']:+9.3f}"
            f" {budget['gradient_work']:+9.3f}"
        )
        if previous is not None and previous[1] < 0.0 <= net:
            # Linear crossing between the last two winds.
            span = net - previous[1]
            neutral = previous[0] + (wind - previous[0]) * (-previous[1]) / span
        previous = (wind, net)
        if wind == 22.0:
            _save(plot_trajectory(trajectory).figure, output / "trajectory.png")
            _save(plot_controls(trajectory).figure, output / "controls.png")
            _save(plot_energy(trajectory).figure, output / "energy.png")
            _save(
                plot_energy_analysis(trajectory)[0].figure,
                output / "energy_analysis.png",
            )
    print(f"\nthe cycle breaks even near w_ref = {neutral:.2f} m/s")
    return neutral


def run_codesign(output: Path, size: int = 20) -> float:
    """Find the weakest wind the bird can fly in, wing included.

    Parameters
    ----------
    output
        The directory to write figures into.
    size
        The discretization.

    Returns
    -------
    float
        The minimum reference wind, in m/s.
    """
    scenario, gemseo_scenario = create_codesign_optimization(size=size)
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=500)
    result = gemseo_scenario.optimization_result
    optimum = result.x_opt_as_dict
    area = float(optimum["constant.wing_area"][0])
    span = float(optimum["constant.wing_span"][0])
    print(
        f"\nminimum wind {float(result.f_opt):.3f} m/s"
        f" with S={area:.3f} m2, b={span:.3f} m, AR={span**2 / area:.1f},"
        f" cycle {float(optimum['t_f'][0]):.2f} s"
    )
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    _save(
        plot_energy_analysis(trajectory)[0].figure,
        output / "codesign_energy_analysis.png",
    )
    _save(plot_trajectory(trajectory).figure, output / "codesign_trajectory.png")
    return float(result.f_opt)


def draw_the_model(output: Path) -> None:
    """Write the figures of the bird and the wind it flies in.

    Parameters
    ----------
    output
        The directory to write into.
    """
    _save(plot_wind_profile()[0].figure, output / "wind_profile.png")
    _save(plot_polar()[0].figure, output / "polar.png")
    _save(plot_section_trade().figure, output / "section_trade.png")


def report_the_problems(size: int = 20) -> None:
    """Print the problem statement of each formulation.

    Parameters
    ----------
    size
        The discretization.
    """
    from gemseo_tempo.problems.albatross.problem import _bounds
    from gemseo_tempo.problems.albatross.problem import create_codesign_scenario
    from gemseo_tempo.problems.albatross.problem import create_energy_scenario
    from gemseo_tempo.scenario import ControlAndDesign
    from gemseo_tempo.scenario import OptimalControl

    energy = create_energy_scenario(size=size, w_ref=22.0)
    print("")
    print("=== A: maximum energy in a fixed wind ===")
    print(
        describe_problem(
            energy, OptimalControl(energy, maximize=True, bounds=_bounds(energy))
        )
    )
    codesign = create_codesign_scenario(size=size)
    print("")
    print("=== C: minimum wind, everything free ===")
    print(
        describe_problem(
            codesign,
            ControlAndDesign(
                codesign,
                ["w_ref", "wing_area", "wing_span", "section"],
                bounds=_bounds(codesign),
            ),
        )
    )


def run_sequential(size: int = 20) -> None:
    """Size the bird first, then fly it, and report both.

    Parameters
    ----------
    size
        The discretization.
    """
    result = sequential_codesign(size=size)
    print("")
    print("=== B: size the bird, then fly it ===")
    print(f"{'figure of merit':>16} {'baseline':>10} {'elasticity':>12}")
    for name, sensitivity in result.sensitivities.items():
        print(
            f"{name:>16} {sensitivity.baseline:10.3f} {sensitivity.elasticity:+12.4f}"
        )
    wing = ", ".join(f"{k}={v:.3f}" for k, v in result.wing.items())
    print(f"sized wing: {wing}")
    print(f"sequential minimum wind: {result.minimum_wind:.3f} m/s")


def main() -> None:
    """Run every formulation and write the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures/albatross"))
    parser.add_argument("--size", type=int, default=20)
    arguments = parser.parse_args()
    logging.disable(logging.CRITICAL)

    draw_the_model(arguments.output)
    report_the_problems(arguments.size)
    run_energy_sweep(arguments.output, arguments.size)
    simultaneous = run_codesign(arguments.output, arguments.size)
    run_sequential(arguments.size)
    print(f"simultaneous minimum wind: {simultaneous:.3f} m/s")
    print(f"figures written to {arguments.output}")


if __name__ == "__main__":
    main()
