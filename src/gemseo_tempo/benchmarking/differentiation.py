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
"""Forward mode, reverse mode and finite differences, and what each costs.

A Jacobian of ``m`` outputs with respect to ``n`` inputs costs ``n``
forward passes, ``m`` reverse passes, or ``n + 1`` evaluations by forward
differences. Which is cheapest is a property of the problem's shape, so the
three are measured on the same function as that shape changes, with and
without compilation, and on four counts:

- **compilation**, the one-off cost of tracing and compiling the function
  and its Jacobian program;
- **execution**, the cost of one Jacobian once compiled;
- **precision**, the largest error against an exact Jacobian, scaled by the
  largest entry;
- **memory**, the bytes XLA's compiled program holds for arguments,
  outputs and temporaries.

Everything is measured on the bare JAX function, not through a GEMSEO
discipline, which caches a linearization until its inputs change, so that
timing a second call at the same point would time the cache.

Each mode may be given its own function, because each is used in its own
configuration: forward mode needs the Direct adjoint through an ODE solve,
while reverse mode normally runs with the checkpointing one, which forward
mode cannot use, and whose memory is the point of it.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

FD = "fd"
"""Forward finite differences, as GEMSEO computes them."""

FORWARD = "forward"
"""Forward-mode automatic differentiation."""

REVERSE = "reverse"
"""Reverse-mode automatic differentiation."""

CHECKPOINTED = "reverse_checkpointed"
"""Reverse mode through the checkpointing adjoint of an ODE solve.

The library's default adjoint, which trades recomputation for memory. It is
kept apart from :data:`REVERSE`, measured through the Direct adjoint,
because the two differ in memory by design and, on some problems, in
precision by accident.
"""

MODES = (FD, FORWARD, REVERSE, CHECKPOINTED)
"""The ways of obtaining a Jacobian, in the order they are drawn."""

MODE_COLORS = {
    FD: "tab:red",
    FORWARD: "tab:green",
    REVERSE: "tab:blue",
    CHECKPOINTED: "tab:purple",
}
"""The colour of each mode in every figure."""

MODE_LABELS = {
    FD: "finite differences",
    FORWARD: "forward",
    REVERSE: "reverse",
    CHECKPOINTED: "reverse, checkpointed",
}
"""The legend label of each mode."""

_REFERENCES = {FD: REVERSE, FORWARD: REVERSE, REVERSE: FORWARD, CHECKPOINTED: FORWARD}
"""The exact Jacobian each mode's error is taken against."""

_FLOOR = 1e-16
"""The smallest error a radar axis distinguishes, machine precision."""


@dataclass
class ModeMeasure:
    """What one way of obtaining a Jacobian cost, on one function."""

    mode: str
    """``"fd"``, ``"forward"`` or ``"reverse"``."""

    jit: bool
    """Whether the programs were compiled."""

    n_inputs: int
    """The number of inputs differentiated with respect to."""

    n_outputs: int
    """The number of outputs differentiated."""

    compile_function: float
    """Seconds to compile the function; zero when not compiled."""

    compile_jacobian: float
    """Seconds to compile the Jacobian program; for finite differences,
    which only evaluate the function, that of the function."""

    seconds: float
    """Seconds for one Jacobian, the fastest of the repeats."""

    error: float
    """The largest error against an exact Jacobian, over its largest entry."""

    memory: int
    """Bytes of XLA's compiled program: arguments, outputs, temporaries."""

    step: float = numpy.nan
    """For finite differences, the relative step, the most precise tried."""

    variant: str = ""
    """For finite differences, ``"batched"`` or ``"sequential"``, the faster."""


def flat_function(problem, inputs: Sequence[str], outputs: Sequence[str]):
    """Return a transcribed problem as a function of one flat vector.

    Parameters
    ----------
    problem
        The transcribed problem of a :class:`.DynamicScenario`.
    inputs
        The inputs to differentiate with respect to, in order.
    outputs
        The outputs to differentiate, in order.

    Returns
    -------
    tuple
        The function, from the flat inputs to the flat outputs, and the
        flat point the problem's defaults make.
    """
    defaults = dict(problem.input_defaults)
    shapes = {name: numpy.shape(defaults[name]) for name in inputs}
    sizes = [int(numpy.prod(shapes[name])) or 1 for name in inputs]
    point = jnp.concatenate([jnp.ravel(jnp.asarray(defaults[name])) for name in inputs])

    def function(flat):
        values, start = dict(defaults), 0
        for name, size in zip(inputs, sizes, strict=True):
            values[name] = jnp.reshape(flat[start : start + size], shapes[name])
            start += size
        computed = problem.jax_function(values)
        return jnp.concatenate([jnp.ravel(computed[name]) for name in outputs])

    return function, point


def _memory(compiled) -> int:
    """Return the bytes a compiled program holds."""
    stats = compiled.memory_analysis()
    return int(
        stats.argument_size_in_bytes
        + stats.output_size_in_bytes
        + stats.temp_size_in_bytes
    )


def _fastest(call, repeats: int) -> float:
    """Return the fastest of a few timed calls."""
    best = numpy.inf
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(call())
        best = min(best, time.perf_counter() - started)
    return best


def _compiled(function, point) -> tuple:
    """Compile a function ahead of time, and time the compilation."""
    started = time.perf_counter()
    compiled = jax.jit(function).lower(point).compile()
    return compiled, time.perf_counter() - started


FD_STEPS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
"""The relative steps tried for finite differences; the most precise is kept."""


def _differences(evaluate, point, step: float):
    """Return a Jacobian by forward differences, one evaluation per input."""
    steps = step * jnp.maximum(1.0, jnp.abs(point))
    base = evaluate(point)
    columns = [
        (evaluate(point.at[index].add(steps[index])) - base) / steps[index]
        for index in range(point.size)
    ]
    return jnp.stack(columns, axis=1)


def batched_differences(function):
    """Return forward differences as one vectorized evaluation.

    Every perturbed point is evaluated in the same call, through ``vmap``,
    which is the most a JAX model allows finite differences: one dispatch,
    and whatever the batch lets XLA share between evaluations.

    Parameters
    ----------
    function
        The function of one flat vector.

    Returns
    -------
    Callable
        ``(point, step) -> jacobian``, the step relative to each input.
    """

    def jacobian(point, step):
        steps = step * jnp.maximum(1.0, jnp.abs(point))
        points = jnp.vstack([point, point + jnp.diag(steps)])
        values = jax.vmap(function)(points)
        return ((values[1:] - values[0]) / steps[:, None]).T

    return jacobian


def best_step(jacobian, point, reference, steps=FD_STEPS) -> tuple[float, float]:
    """Return the finite-difference step that comes closest to an exact Jacobian.

    Too large a step measures the curvature, too small a one the rounding,
    and where the two meet depends on the function, so the step is chosen
    against an exact Jacobian rather than assumed.

    Parameters
    ----------
    jacobian
        ``(point, step) -> jacobian``, as :func:`batched_differences`
        returns.
    point
        Where to differentiate.
    reference
        The exact Jacobian there.
    steps
        The relative steps tried.

    Returns
    -------
    tuple
        The best step and its error, over the largest entry.
    """
    scale = max(float(numpy.max(numpy.abs(reference))), _FLOOR)
    errors = {
        step: float(
            numpy.max(numpy.abs(numpy.asarray(jacobian(point, step)) - reference))
        )
        / scale
        for step in steps
    }
    step = min(errors, key=errors.get)
    return step, errors[step]


def measure_modes(
    functions: Mapping[str, Callable],
    point,
    jit: bool = True,
    repeats: int = 3,
    step: float = 1e-7,
) -> list[ModeMeasure]:
    """Measure each way of obtaining the Jacobian of a function, at its best.

    Every mode is given its best configuration. The exact modes are
    compiled, or not, as asked. Finite differences take the step that comes
    closest to an exact Jacobian, and are timed both batched into one
    vectorized call and one evaluation at a time, the faster counting.

    The errors are taken against reverse mode, and those of reverse mode
    and of its checkpointed variant against forward mode: two exact methods
    that agree to rounding are the best reference available without a
    closed form.

    Parameters
    ----------
    functions
        The function each mode differentiates, by mode, all computing the
        same thing; a mode left out is not measured.
    point
        The flat point to differentiate at.
    jit
        Whether to compile the programs.
    repeats
        How many times each Jacobian is timed; the fastest counts.
    step
        The finite-difference step, relative to each input, when no exact
        mode is measured to choose it against.

    Returns
    -------
    list
        One measure per mode given.
    """
    point = jnp.asarray(point, dtype=float)
    programs = {
        FORWARD: jax.jacfwd,
        REVERSE: jax.jacrev,
        CHECKPOINTED: jax.jacrev,
    }
    jacobians, measures = {}, {}
    # The exact modes first, so that the finite-difference step can be
    # chosen against one of them.
    for mode in (mode for mode in MODES if mode in functions and mode != FD):
        function = functions[mode]
        _, compile_function = _compiled(function, point)
        transform = programs[mode]
        compiled_jacobian, compile_jacobian = _compiled(transform(function), point)
        if jit:

            def jacobian(compiled=compiled_jacobian):
                return compiled(point)
        else:

            def jacobian(transform=transform, function=function):
                return transform(function)(point)

        value = numpy.atleast_2d(numpy.asarray(jax.block_until_ready(jacobian())))
        jacobians[mode] = value
        measures[mode] = {
            "mode": mode,
            "jit": jit,
            "n_inputs": int(point.size),
            "n_outputs": int(value.shape[0]),
            "compile_function": compile_function if jit else 0.0,
            "compile_jacobian": compile_jacobian if jit else 0.0,
            "seconds": _fastest(jacobian, repeats),
            "memory": _memory(compiled_jacobian),
        }
    if FD in functions:
        measures[FD] = _measure_differences(
            functions[FD], point, jit, repeats, step, jacobians
        )
    result = []
    for mode in (mode for mode in MODES if mode in measures):
        measure = measures[mode]
        value = jacobians[mode]
        reference = jacobians.get(_REFERENCES[mode])
        if reference is None:
            error = numpy.nan
        else:
            scale = max(float(numpy.max(numpy.abs(reference))), _FLOOR)
            error = float(numpy.max(numpy.abs(value - reference))) / scale
        result.append(ModeMeasure(error=error, **measure))
    return result


def _measure_differences(function, point, jit, repeats, step, jacobians):
    """Measure finite differences at their best, and add their Jacobian.

    Parameters
    ----------
    function
        The function.
    point
        Where to differentiate.
    jit
        Whether to compile.
    repeats
        How many times each variant is timed.
    step
        The step to fall back on when there is no exact Jacobian.
    jacobians
        The exact Jacobians measured so far, by mode, which this completes.

    Returns
    -------
    dict
        The measure of the faster variant.
    """
    compiled_function, compile_function = _compiled(function, point)
    batched = batched_differences(function)
    started = time.perf_counter()
    compiled_batched = jax.jit(batched).lower(point, 1.0).compile()
    compile_batched = time.perf_counter() - started
    run_batched = compiled_batched if jit else batched
    reference = jacobians.get(REVERSE, jacobians.get(FORWARD))
    if reference is not None:
        step, _ = best_step(compiled_batched, point, reference)
    evaluate = compiled_function if jit else function
    variants = {
        "batched": (
            lambda: run_batched(point, step),
            compile_batched,
            _memory(compiled_batched),
        ),
        "sequential": (
            lambda: _differences(evaluate, point, step),
            compile_function,
            None,
        ),
    }
    timings = {name: _fastest(call, repeats) for name, (call, _, _) in variants.items()}
    variant = min(timings, key=timings.get)
    call, compile_jacobian, memory = variants[variant]
    value = numpy.atleast_2d(numpy.asarray(jax.block_until_ready(call())))
    jacobians[FD] = value
    if memory is None:
        memory = _memory(compiled_function) + value.nbytes
    return {
        "mode": FD,
        "jit": jit,
        "n_inputs": int(point.size),
        "n_outputs": int(value.shape[0]),
        "compile_function": compile_function if jit else 0.0,
        "compile_jacobian": compile_jacobian if jit else 0.0,
        "seconds": timings[variant],
        "memory": memory,
        "step": step,
        "variant": variant,
    }


def as_records(measures: Sequence[ModeMeasure]) -> list[dict]:
    """Return measures as plain dictionaries, ready to store.

    Parameters
    ----------
    measures
        The measures.

    Returns
    -------
    list
        One dictionary per measure.
    """
    return [asdict(measure) for measure in measures]


def plot_mode_sweep(records: Sequence[Mapping], dimension: str, axes=None):
    """Draw the cost of a Jacobian by each mode against one dimension.

    Parameters
    ----------
    records
        Stored measures, compiled and not, over a range of the dimension.
    dimension
        ``"n_inputs"`` or ``"n_outputs"``, the abscissa.
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = plt.subplots(figsize=(5.0, 3.6))
    for mode in MODES:
        for jit, style in ((True, "-"), (False, "--")):
            chosen = sorted(
                (r for r in records if r["mode"] == mode and r["jit"] == jit),
                key=lambda r: r[dimension],
            )
            if not chosen:
                continue
            axes.plot(
                [r[dimension] for r in chosen],
                [r["seconds"] for r in chosen],
                style,
                marker="o" if jit else None,
                color=MODE_COLORS[mode],
                label=f"{MODE_LABELS[mode]}{'' if jit else ', not compiled'}",
            )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("inputs" if dimension == "n_inputs" else "outputs")
    axes.set_ylabel("seconds per Jacobian")
    axes.grid(alpha=0.3, which="both")
    axes.legend(fontsize="small")
    return axes


def plot_compile_times(records: Sequence[Mapping], dimension: str, axes=None):
    """Draw what compiling the function and each Jacobian program costs.

    Parameters
    ----------
    records
        Stored compiled measures over a range of the dimension.
    dimension
        ``"n_inputs"`` or ``"n_outputs"``, the abscissa.
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = plt.subplots(figsize=(5.0, 3.6))
    compiled = [r for r in records if r["jit"]]
    function = {}
    for record in compiled:
        function.setdefault(record[dimension], record["compile_function"])
    sizes = sorted(function)
    axes.plot(
        sizes,
        [function[size] for size in sizes],
        "k-",
        marker="s",
        label="the function",
    )
    for mode in (FORWARD, REVERSE, CHECKPOINTED):
        chosen = sorted(
            (r for r in compiled if r["mode"] == mode), key=lambda r: r[dimension]
        )
        if chosen:
            axes.plot(
                [r[dimension] for r in chosen],
                [r["compile_jacobian"] for r in chosen],
                "-",
                marker="o",
                color=MODE_COLORS[mode],
                label=f"its {MODE_LABELS[mode]} Jacobian",
            )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("inputs" if dimension == "n_inputs" else "outputs")
    axes.set_ylabel("seconds to compile")
    axes.grid(alpha=0.3, which="both")
    axes.legend(fontsize="small")
    return axes


RADAR_AXES = (
    ("compilation", lambda r: r["compile_function"] + r["compile_jacobian"]),
    ("execution", lambda r: r["seconds"]),
    ("precision", lambda r: r["error"]),
    ("memory", lambda r: r["memory"]),
)
"""The four axes of the radar, each a cost to be kept small."""


def radar_scores(records: Sequence[Mapping]) -> dict[str, list[float]]:
    """Score each mode on each radar axis, the best at one.

    Each cost is compared on a logarithmic scale, since the modes differ by
    orders of magnitude, and mapped so that the best mode scores one and
    the worst a fifth: the worst is still drawn, so that its shape shows.

    Parameters
    ----------
    records
        One compiled measure per mode, on the same function.

    Returns
    -------
    dict
        The scores of each mode, in the order of :data:`RADAR_AXES`.
    """
    scores = {record["mode"]: [] for record in records}
    for _, cost in RADAR_AXES:
        logs = {
            record["mode"]: numpy.log10(max(float(cost(record)), _FLOOR))
            for record in records
        }
        best, worst = min(logs.values()), max(logs.values())
        for mode, value in logs.items():
            share = 1.0 if worst == best else (worst - value) / (worst - best)
            scores[mode].append(0.2 + 0.8 * share)
    return scores


def plot_mode_radar(records: Sequence[Mapping], axes=None):
    """Draw the modes on the four costs, the best at the rim.

    Parameters
    ----------
    records
        One compiled measure per mode, on the same function.
    axes
        A polar axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = plt.subplots(
            figsize=(5.0, 5.0),
            subplot_kw={"projection": "polar"},
            layout="constrained",
        )
    labels = [label for label, _ in RADAR_AXES]
    angles = numpy.linspace(0.0, 2.0 * numpy.pi, len(labels), endpoint=False)
    closed = numpy.concatenate([angles, angles[:1]])
    for mode, scores in radar_scores(records).items():
        values = numpy.concatenate([scores, scores[:1]])
        axes.plot(closed, values, color=MODE_COLORS[mode], label=MODE_LABELS[mode])
        axes.fill(closed, values, color=MODE_COLORS[mode], alpha=0.12)
    axes.set_xticks(angles)
    axes.set_xticklabels(labels)
    axes.set_yticks([0.2, 0.6, 1.0])
    axes.set_yticklabels(["worst", "", "best"], fontsize="small")
    # Between memory and compilation, clear of the axis labels.
    axes.set_rlabel_position(315.0)
    axes.set_ylim(0.0, 1.05)
    axes.figure.legend(loc="outside lower center", ncols=2, fontsize="small")
    return axes


STAGE_INITIALS = {FORWARD: "F", REVERSE: "R"}
"""The letter each mode is abbreviated to in a chain's label."""


def chain_label(modes: Mapping[str, str]) -> str:
    """Return the short label of a chain's modes, stage by stage.

    Parameters
    ----------
    modes
        The mode of each stage, in execution order.

    Returns
    -------
    str
        One letter per stage, such as ``"R·F·R"``.
    """
    return "·".join(STAGE_INITIALS[mode] for mode in modes.values())


def plot_stage_modes(chains: Sequence[Mapping], fd: Mapping | None = None, axes=None):
    """Draw what each choice of modes along a chain costs, in time and memory.

    Parameters
    ----------
    chains
        One record per combination of modes: ``modes``, ``seconds`` and
        ``memory``.
    fd
        The same for finite differences of the whole chain, if measured.
    axes
        Two axes, for the time and for the memory; a new figure when
        omitted.

    Returns
    -------
    list
        The axes drawn on.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    rows = [(chain_label(chain["modes"]), chain) for chain in chains]
    if fd is not None:
        rows.append(("FD", fd))
    positions = numpy.arange(len(rows))
    for axis, key, label in (
        (axes[0], "seconds", "seconds per chain Jacobian"),
        (axes[1], "memory", "bytes, the largest stage"),
    ):
        values = [row[key] for _, row in rows]
        # The cheapest chain in green, finite differences in their own red.
        smallest = min(
            value for (name, _), value in zip(rows, values, strict=True) if name != "FD"
        )
        colors = [
            MODE_COLORS[FD]
            if name == "FD"
            else ("tab:green" if value == smallest else "tab:gray")
            for (name, _), value in zip(rows, values, strict=True)
        ]
        axis.barh(positions, values, color=colors)
        axis.set_xscale("log")
        # Every bar from the decade below the smallest, so that their lengths
        # compare; left to itself the axis starts just short of the smallest.
        axis.set_xlim(left=10.0 ** numpy.floor(numpy.log10(min(values))))
        axis.set_xlabel(label)
        axis.grid(alpha=0.3, axis="x", which="both")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([name for name, _ in rows])
    axes[0].set_ylabel("statics · dynamics · outputs")
    axes[0].invert_yaxis()
    return axes
