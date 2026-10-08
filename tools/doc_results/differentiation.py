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
"""Differentiating each problem every way, at its best."""

from __future__ import annotations

import itertools
import time

import jax.numpy as jnp
import numpy

from .common import _write_json


def _measure_configuration(build, adjoints, jit, repeats):
    """Measure the three modes on one configuration of one problem.

    Parameters
    ----------
    build
        ``adjoint -> scenario``, the problem at this configuration.
    adjoints
        The adjoint each mode differentiates the integration through.
    jit
        Whether to compile.
    repeats
        How many times each Jacobian is timed.

    Returns
    -------
    list
        One record per mode.
    """
    from gemseo_tempo.benchmarking.differentiation import as_records
    from gemseo_tempo.benchmarking.differentiation import flat_function
    from gemseo_tempo.benchmarking.differentiation import measure_modes

    if not jit:
        # Without compiling, the checkpointed variant only repeats what
        # reverse mode shows, at the cost the sweep can least afford.
        adjoints = {
            mode: a for mode, a in adjoints.items() if mode != "reverse_checkpointed"
        }
    problems = {adjoint: build(adjoint).problem for adjoint in set(adjoints.values())}
    functions, point = {}, None
    for mode, adjoint in adjoints.items():
        problem = problems[adjoint]
        inputs = list(problem.design_variables)
        outputs = [problem.objective_name] + [
            name for name, _ in problem.constraint_names
        ]
        functions[mode], point = flat_function(problem, inputs, outputs)
    # A generic point, a thousandth away from the guess: the sail's guess
    # heads exactly along a breakpoint of its polar table, where a one-sided
    # difference straddles a kink and an exact derivative picks one side.
    offset = numpy.random.default_rng(0).standard_normal(point.size)
    point = point * (1.0 + 1e-3 * offset)
    return as_records(measure_modes(functions, point, jit=jit, repeats=repeats))


def _differentiation_study(name, builder, adjoints, sweeps, eager, smoke) -> dict:
    """Sweep the inputs and the outputs, by every mode, compiled and not.

    Parameters
    ----------
    name
        The problem, which names the stored file.
    builder
        ``(size, n_save, adjoint) -> scenario``: the control intervals set
        the inputs, the save points the outputs of any path constraint.
    adjoints
        The adjoint each mode differentiates the integration through.
    sweeps
        ``{"inputs": (sizes, n_save), "outputs": (size, saves)}``; either
        may be left out.
    eager
        The largest size and save count also measured without compiling,
        whose cost grows quickly, as ``(size, n_save)``.
    smoke
        Whether to run the tiny version, for the tests.

    Returns
    -------
    dict
        What is stored.
    """
    repeats = 1 if smoke else 3
    study = {}
    if "inputs" in sweeps:
        sizes, n_save = sweeps["inputs"]
        records = []
        for size in sizes[:2] if smoke else sizes:
            for jit in (True, False) if size <= eager[0] else (True,):
                records += _measure_configuration(
                    lambda adjoint, size=size: builder(size, n_save, adjoint),
                    adjoints,
                    jit,
                    repeats,
                )
        study["inputs"] = records
    if "outputs" in sweeps:
        size, saves = sweeps["outputs"]
        records = []
        for n_save in saves[:2] if smoke else saves:
            for jit in (True, False) if n_save <= eager[1] else (True,):
                records += _measure_configuration(
                    lambda adjoint, n_save=n_save: builder(size, n_save, adjoint),
                    adjoints,
                    jit,
                    repeats,
                )
        study["outputs"] = records
    _write_json(f"{name}/differentiation.json", study)
    return study


def _chain_rule(products, seeds, data) -> dict:
    """Carry total derivatives through a chain of stage Jacobians.

    Parameters
    ----------
    products
        ``(jacobian, wrt, of)`` per stage, in execution order: the Jacobian
        of the concatenated ``of`` outputs with respect to the concatenated
        ``wrt`` inputs.
    seeds
        The total derivative of each design variable, its identity block.
    data
        The values, which give each output's size.

    Returns
    -------
    dict
        The total derivative of every quantity, by name.
    """
    derivatives = dict(seeds)
    for jacobian, wrt, of in products:
        rows = jacobian @ numpy.vstack([derivatives[name] for name in wrt])
        start = 0
        for name in of:
            count = int(numpy.size(data[name]))
            derivatives[name] = rows[start : start + count]
            start += count
    return derivatives


def _chain_modes_study(smoke: bool) -> dict:
    """Differentiate the albatross's design chain with a mode per stage.

    The split assembly makes three stages of the energy cycle: the statics,
    which turn the wing into the aerodynamic constants; the dynamics, which
    fly the cycle; and the outputs, which read its energy and constraints.
    Each stage is differentiated in its own mode, every combination of the
    eight, on the bare JAX function of the stage, and the chain Jacobian is
    their product. Finite differences of the whole chain stand alongside,
    at their best configuration.

    Parameters
    ----------
    smoke
        Whether to run the tiny version, for the tests.

    Returns
    -------
    dict
        What is stored.
    """
    import jax

    from gemseo_tempo.benchmarking.differentiation import FD
    from gemseo_tempo.benchmarking.differentiation import REVERSE
    from gemseo_tempo.benchmarking.differentiation import _fastest
    from gemseo_tempo.benchmarking.differentiation import _measure_differences
    from gemseo_tempo.benchmarking.differentiation import _memory
    from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
    from gemseo_tempo.problems.albatross import create_energy_scenario
    from gemseo_tempo.problems.albatross.problem import WING_BOUNDS
    from gemseo_tempo.transcriptions import SingleShooting

    size, n_save = (5, 11) if smoke else (20, 21)
    repeats = 1 if smoke else 3
    wing = ["constant." + key for key in WING_BOUNDS]
    modes = ("forward", "reverse")

    def flatten(values, names):
        return jnp.concatenate([jnp.ravel(values[name]) for name in names])

    def stage(discipline, wrt, of, data):
        """Return the stage as a function of one vector, and its point."""
        shapes = {name: jnp.shape(data[name]) for name in wrt}

        def function(flat):
            values, start = dict(data), 0
            for name in wrt:
                count = int(numpy.prod(shapes[name])) or 1
                values[name] = jnp.reshape(flat[start : start + count], shapes[name])
                start += count
            return flatten(discipline.jax_out_func(values), of)

        return function, flatten(data, wrt)

    records, reference = [], None
    for combination in itertools.product(modes, repeat=3):
        methods = dict(
            zip(("statics", "dynamics", "outputs"), combination, strict=True)
        )
        # The Direct adjoint whatever the mode, so that only the mode varies:
        # it is exact both ways, where the checkpointing one is not always.
        scenario = create_energy_scenario(
            transcription=SingleShooting(),
            size=size,
            w_ref=22.0,
            integrator=DiffraxIntegrator(
                dt0=1.0 / 400, n_save=n_save, adjoint="Direct"
            ),
            assembly="split",
            differentiation_method=methods,
        )
        problem = scenario.problem
        disciplines = dict(
            zip(("statics", "dynamics", "outputs"), scenario.disciplines, strict=True)
        )
        data = dict(problem.input_defaults)
        for discipline in disciplines.values():
            inputs = {
                name: data.get(name, discipline.default_input_data.get(name))
                for name in discipline.input_grammar.names
            }
            data.update(discipline.jax_out_func(inputs))
        controls = [
            name for name in problem.design_variables if name.startswith("control.")
        ]
        design = controls + wing
        final = [problem.objective_name] + [n for n, _ in problem.constraint_names]
        consumers = {
            "statics": set(disciplines["dynamics"].input_grammar.names)
            | set(disciplines["outputs"].input_grammar.names),
            "dynamics": set(disciplines["outputs"].input_grammar.names),
            "outputs": set(final),
        }
        # The total derivative of every quantity with respect to the design,
        # seeded with the identity on the design itself.
        sizes = {name: int(numpy.size(data[name])) for name in design}
        identity = numpy.eye(sum(sizes.values()))
        seeds, start = {}, 0
        for name in design:
            seeds[name] = identity[start : start + sizes[name]]
            start += sizes[name]
        record = {"modes": methods}
        products, known = [], set(design)
        for label, discipline in disciplines.items():
            wrt = [n for n in discipline.input_grammar.names if n in known]
            of = [n for n in discipline.output_grammar.names if n in consumers[label]]
            known.update(of)
            function, point = stage(discipline, wrt, of, data)
            transform = jax.jacfwd if methods[label] == "forward" else jax.jacrev
            started = time.perf_counter()
            compiled = jax.jit(transform(function)).lower(point).compile()
            record[f"{label}_compile"] = time.perf_counter() - started
            record[f"{label}_seconds"] = _fastest(
                lambda c=compiled, p=point: c(p), repeats
            )
            record[f"{label}_memory"] = _memory(compiled)
            jacobian = numpy.atleast_2d(numpy.asarray(compiled(point)))
            record[f"{label}_shape"] = list(jacobian.shape)
            products.append((jacobian, wrt, of))
        total = _chain_rule(products, seeds, data)
        chain_jacobian = numpy.vstack([total[name] for name in final])
        record["product_seconds"] = _fastest(
            lambda products=products, seeds=seeds, data=data: _chain_rule(
                products, seeds, data
            ),
            repeats,
        )
        record["seconds"] = record["product_seconds"] + sum(
            record[f"{label}_seconds"] for label in disciplines
        )
        record["memory"] = max(record[f"{label}_memory"] for label in disciplines)
        if reference is None:
            # The whole chain as one function of the design, differenced, and
            # differentiated in reverse through the Direct adjoint: the
            # reference the products are checked against.
            def whole(flat, problem=problem, design=design, final=final):
                values = dict(problem.input_defaults)
                start = 0
                for name in design:
                    count = int(numpy.size(values[name]))
                    values[name] = jnp.reshape(
                        flat[start : start + count], jnp.shape(values[name])
                    )
                    start += count
                return flatten(problem.jax_function(values), final)

            point = flatten(problem.input_defaults, design)
            reference = numpy.asarray(jax.jit(jax.jacrev(whole))(point))
            # At their best: batched or one evaluation at a time, whichever
            # is faster, at the step closest to the exact Jacobian.
            jacobians = {REVERSE: reference}
            measured = _measure_differences(
                whole, point, True, repeats, 1e-6, jacobians
            )
            scale = float(numpy.max(numpy.abs(reference)))
            fd = {
                "modes": {"chain": "fd"},
                "seconds": measured["seconds"],
                "memory": measured["memory"],
                "step": measured["step"],
                "variant": measured["variant"],
                "error": float(numpy.max(numpy.abs(jacobians[FD] - reference))) / scale,
            }
        scale = float(numpy.max(numpy.abs(reference)))
        record["error"] = (
            float(numpy.max(numpy.abs(chain_jacobian - reference))) / scale
        )
        records.append(record)
    study = {"size": size, "n_save": n_save, "chains": records, "fd": fd}
    _write_json("albatross/chain_modes.json", study)
    return study


def generate_differentiation(name: str, smoke: bool) -> None:
    """Run one problem's differentiation studies.

    Parameters
    ----------
    name
        The problem.
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
    from gemseo_tempo.transcriptions import SingleShooting

    # Every mode through the Direct adjoint, which is exact in both
    # directions, and reverse mode once more through the checkpointing one,
    # the library's default, whose saving is memory and whose risk is
    # precision.
    checkpoint = dict.fromkeys(("fd", "forward", "reverse"), "Direct")
    checkpoint["reverse_checkpointed"] = "RecursiveCheckpoint"
    if name == "brachistochrone":
        from gemseo_tempo.problems.brachistochrone import create_scenario

        def builder(size, n_save, adjoint):
            return create_scenario(
                transcription=SingleShooting(),
                variant="ideal",
                size=size,
                integrator=DiffraxIntegrator(
                    dt0=1.0 / 400, n_save=n_save, adjoint=adjoint
                ),
            )

        _differentiation_study(
            name,
            builder,
            checkpoint,
            {"inputs": ((5, 10, 20, 40, 80, 160), 21)},
            (80, 0),
            smoke,
        )
    elif name == "sail":
        from gemseo_tempo.problems.sail import create_scenario
        from gemseo_tempo.problems.sail.problem import sail_integrator

        def builder(size, n_save, adjoint):
            return create_scenario(
                transcription=SingleShooting(),
                size=size,
                integrator=sail_integrator(dt0=1.0 / 300, n_save=n_save),
            )

        # The sail's derivatives go through the Direct adjoint either way.
        _differentiation_study(
            name,
            builder,
            dict.fromkeys(("fd", "forward", "reverse"), "Direct"),
            {
                "inputs": ((5, 10, 25, 50, 100, 200), 26),
                "outputs": (25, (11, 41, 161, 641)),
            },
            (50, 161),
            smoke,
        )
    else:
        from gemseo_tempo.problems.albatross import create_energy_scenario

        def builder(size, n_save, adjoint):
            return create_energy_scenario(
                transcription=SingleShooting(),
                size=size,
                w_ref=22.0,
                integrator=DiffraxIntegrator(
                    dt0=1.0 / 400, n_save=n_save, adjoint=adjoint
                ),
            )

        _differentiation_study(
            name,
            builder,
            checkpoint,
            {
                "inputs": ((5, 10, 20, 40, 80), 21),
                "outputs": (20, (11, 41, 161, 641)),
            },
            (20, 41),
            smoke,
        )
        _chain_modes_study(smoke)
