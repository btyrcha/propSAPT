"""Calculations whose CPSCF solves are batched.

A builder is a generator that calculates something, e.g. a density component or the
terms of a property. Where it needs CPSCF amplitudes it yields a list of requests,
`(monomer, perturbation)` pairs with an (ov) perturbation as for `Dimer.cpscf`, and is
sent back the (vo) amplitudes, in the same order. Its return value is its result.

`solve_batched` runs several builders side by side and solves the requests they have
pending at the same time with one `Dimer.cpscf_batch` call per monomer, which costs much
less than a solve per request. `solve_now` runs one builder alone.
"""

import inspect
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import nullcontext

import numpy as np
import psi4

from .molecule import Dimer
from .utils import CalcTimer

CPSCFRequest = tuple[str, np.ndarray]
DensityBuilder = Generator[list[CPSCFRequest], list[np.ndarray], np.ndarray]


def solve_batched(
    mol: Dimer, builders: Mapping[str, Generator], report: bool = True
) -> dict:
    """Runs builders side by side, batching their CPSCF solves.

    Every builder runs up to its next request. The requests then pending are solved
    with one `Dimer.cpscf_batch` call per monomer, and the amplitudes are sent back to
    the builders, which run on to their next requests or finish. The time each builder
    takes, without the solves, is printed under its name.

    Args:
        mol (Dimer): The dimer.
        builders (Mapping[str, Generator]): Builders keyed by the name of what they
            calculate, e.g. "Second-order Induction density calculation".
        report (bool, optional): Whether to print the timings to the Psi4 output.
            Defaults to True.

    Returns:
        dict: The return value of each builder, keyed as `builders`.
    """

    results = {}
    pending = {}
    timings = dict.fromkeys(builders, 0.0)

    def advance(name, amplitudes):
        start = time.perf_counter()
        try:
            requests = builders[name].send(amplitudes)
        except StopIteration as finished:
            results[name] = finished.value
        else:
            pending[name] = list(requests)
        timings[name] += time.perf_counter() - start

    for name in builders:
        advance(name, None)

    while pending:
        amplitudes = {
            name: [None] * len(requests) for name, requests in pending.items()
        }
        monomers = sorted(
            {monomer for requests in pending.values() for monomer, _ in requests}
        )
        for monomer in monomers:
            slots = [
                (name, i)
                for name, requests in pending.items()
                for i, (requested, _) in enumerate(requests)
                if requested == monomer
            ]
            timer = CalcTimer(
                f"CPSCF of monomer {monomer} for {len(slots)} perturbations"
            )
            with timer if report else nullcontext():
                solved = mol.cpscf_batch(
                    monomer, [pending[name][i][1] for name, i in slots]
                )
            for (name, i), x in zip(slots, solved):
                amplitudes[name][i] = x

        waiting, pending = pending, {}
        for name in waiting:
            advance(name, amplitudes[name])

    for name, seconds in timings.items() if report else ():
        psi4.core.print_out(
            f"...{name} took a total of {seconds: .2f} seconds, besides CPSCF.\n"
        )

    return results


def solve_now(mol: Dimer, builder: Generator):
    """Runs one builder, solving its CPSCF requests as they come.

    Returns:
        The builder's return value.
    """

    return solve_batched(mol, {"builder": builder}, report=False)["builder"]


def theta_amplitudes(mol: Dimer, key: str, build: Callable):
    """Builder step that returns the theta amplitudes `key` kept on `mol`, see
    `Dimer.get_theta_amplitudes`. If they are missing, they are calculated with `build`,
    a function of the dimer or a builder of them, whose requests are passed on.

    Use as `theta = yield from theta_amplitudes(mol, key, build)` in a builder.
    """

    if key not in mol.theta_amplitudes:
        amplitudes = build(mol)
        if inspect.isgenerator(amplitudes):
            amplitudes = yield from amplitudes
        mol.store_theta_amplitudes(key, amplitudes)

    return mol.theta_amplitudes[key]
