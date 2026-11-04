"""
Invariant checks: cheap checks of properties that propSAPT results must have.

A failed check warns with `InvariantWarning`, attributed to the first caller outside
propSAPT, and writes the same message into the Psi4 output file; it never stops a
calculation. Turn the warnings into errors with
``warnings.simplefilter("error", InvariantWarning)``. Each check takes its tolerance from
a propSAPT option, and setting the option to None turns the check off.

Classes
-------
InvariantWarning
    Warned when a result violates an invariant by more than its tolerance.

Functions
---------
check_symmetric
    Density matrices are symmetric, ``max|D - D^T|`` within the `check_symmetry_atol`
    option. `calc_density_matrix` checks every component, `save_densities` every matrix
    it saves.
check_partition_of_unity
    Fragment weights sum to 1 wherever the basis functions reach, within the
    `check_partition_atol` option. `FragmentOperators.from_grid` checks the weights
    unless given `partition_of_unity=False`.
check_charge_closure
    Fragment populations add up to the total, within the `check_closure_atol` option.
    `FragmentOperators.population` and `charge` check them against Tr(D S) when the
    fragment weights are a partition of unity.
"""

import os
import sys
import warnings
from collections.abc import Mapping
import numpy as np
import psi4

from .evaluate import (
    _check_basisset,
    _get_chunk_size,
    _iter_phi_chunks,
    _prepare_points,
)
from .options import PROPSAPT_OPTIONS

# number of failing matrices listed by index in a closure warning
_LISTED_INDICES = 10
# warnings are attributed to the first frame outside this directory
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


class InvariantWarning(RuntimeWarning):
    """Warned when a result violates an invariant by more than its tolerance."""


def check_symmetric(
    matrices: np.ndarray | Mapping[str, np.ndarray],
    atol: float | None = None,
    where: str = "",
) -> dict[str, float]:
    """
    Check that density matrices are symmetric, ``max|D - D^T| <= atol``, and warn about
    those that are not.

    The density matrix of a real wavefunction is symmetric. An antisymmetric part does not
    enter the density or the trace with a symmetric operator, but it corrupts
    eigendecompositions (`numpy.linalg.eigh` reads one triangle) and formats that store one
    triangle, such as fchk files.

    Args:
        matrices (np.ndarray | Mapping[str, np.ndarray]): A square matrix, or square
            matrices by name.
        atol (float | None, optional): Largest element of ``|D - D^T|`` allowed.
            Defaults to the `check_symmetry_atol` option; the check is off if that is None.
        where (str, optional): What the matrices belong to, for the message,
            e.g. "monomer A". Defaults to "".

    Raises:
        ValueError: If a matrix is not square, or the tolerance is not a number >= 0.

    Returns:
        dict[str, float]: ``max|D - D^T|`` of each matrix above the tolerance, by name
            ("matrix" for a single matrix); empty if all pass or the check is off.
    """

    atol = _tolerance(atol, "check_symmetry_atol")
    if atol is None:
        return {}

    if not isinstance(matrices, Mapping):
        matrices = {"matrix": matrices}

    deviations = {}
    for name, matrix in matrices.items():
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Matrix '{name}' should be square to check its symmetry, "
                f"has shape {matrix.shape}!"
            )
        deviation = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
        if not deviation <= atol:  # also catches NaN
            deviations[name] = deviation

    if deviations:
        of = f" of {where}" if where else ""
        listed = ", ".join(
            f"{dev:.1e} for '{name}'" for name, dev in deviations.items()
        )
        _warn(
            f"Density matrices{of} are not symmetric: max|D - D^T| = {listed}, above "
            f"{atol:.1e} (option `check_symmetry_atol`). Only their symmetric part enters "
            "densities and traces with symmetric operators; symmetrize them with "
            "0.5 * (D + D.T) before an eigendecomposition or saving one triangle.",
        )

    return deviations


def check_partition_of_unity(
    points: np.ndarray,
    weights: np.ndarray,
    fragment_weights: Mapping[str, np.ndarray],
    basisset: psi4.core.BasisSet,
    atol: float | None = None,
    where: str = "",
    chunk_size: int | None = None,
) -> float | None:
    """
    Check that fragment weights sum to 1 wherever the basis functions reach, and warn if
    they do not.

    The defect is the largest element of

        Delta[mu,nu] = sum_q w_q (sum_X w_X(r_q) - 1) phi_mu(r_q) phi_nu(r_q),

    the difference between the sum of the fragments' weighted-overlap matrices and the
    overlap matrix on the same grid. Points that no basis function reaches do not count,
    so weights that stop summing to 1 far from the molecule (LISA weights do, tens of bohr
    away) pass. With nonnegative quadrature weights the defect is at most
    ``max|sum_X w_X - 1|`` times the largest diagonal element of the overlap matrix on
    the grid (about 1 for normalized basis functions), so weights within the tolerance at
    every point pass without evaluating any basis function.

    Args:
        points (np.ndarray): Points of the quadrature grid in bohr, shape (..., 3).
        weights (np.ndarray): Quadrature weights, with the shape of `points` without its
            last axis.
        fragment_weights (Mapping[str, np.ndarray]): Weight w_X of each fragment at the
            points, with the same shape as `weights`.
        basisset (psi4.core.BasisSet): Basis set of the functions phi.
        atol (float | None, optional): Largest defect allowed. Defaults to the
            `check_partition_atol` option; the check is off if that is None.
        where (str, optional): What the weights belong to, for the message.
            Defaults to "".
        chunk_size (int | None, optional): Number of points evaluated at once.
            Defaults to the `point_eval_chunk_size` option.

    Raises:
        ValueError: If the shapes do not match, or the tolerance is not a number >= 0.

    Returns:
        float | None: The defect if it is above the tolerance, else None.
    """

    atol = _tolerance(atol, "check_partition_atol")
    if atol is None:
        return None

    _check_basisset(basisset)
    points = _prepare_points(points)
    chunk_size = _get_chunk_size(chunk_size)
    points_shape = points.shape[:-1]

    weights = np.asarray(weights, dtype=float)
    if weights.shape != points_shape:
        raise ValueError(
            f"`weights` should have the shape {points_shape} of the points, "
            f"was {weights.shape}!"
        )
    if not isinstance(fragment_weights, Mapping) or not fragment_weights:
        raise ValueError("Fragment weights should be a nonempty mapping of arrays!")

    deviation = np.full(points_shape, -1.0)
    for name, fragment in fragment_weights.items():
        fragment = np.asarray(fragment, dtype=float)
        if fragment.shape != points_shape:
            raise ValueError(
                f"Weights of fragment `{name}` should have the shape {points_shape} of "
                f"the points, was {fragment.shape}!"
            )
        deviation += fragment
    deviation = deviation.reshape(-1)
    weights = weights.reshape(-1)

    pointwise = float(np.max(np.abs(deviation), initial=0.0))
    if pointwise <= atol and np.all(weights >= 0.0):
        return None  # the defect is at most `pointwise`

    deviating = deviation != 0.0
    scaled = (weights * deviation)[deviating]
    nbf = basisset.nbf()
    defect_matrix = np.zeros((nbf, nbf))
    for chunk, phi in _iter_phi_chunks(
        points.reshape(-1, 3)[deviating], basisset, chunk_size
    ):
        defect_matrix += phi.T @ (scaled[chunk, None] * phi)
    defect = float(np.max(np.abs(defect_matrix), initial=0.0))

    if defect <= atol:
        return None

    of = f" of {where}" if where else ""
    _warn(
        f"The weights of fragments {', '.join(map(str, fragment_weights))}{of} do not "
        "sum to 1 where the basis functions reach: "
        f"max|sum_q w_q (sum_X w_X - 1) phi_mu phi_nu| = {defect:.1e} (at the points, "
        f"max|sum_X w_X - 1| = {pointwise:.1e}), above {atol:.1e} (option "
        "`check_partition_atol`). Their populations will not add up to the electrons of "
        "a density; for fragments that do not cover the whole system use "
        "`partition_of_unity=False`.",
    )

    return defect


def check_charge_closure(
    populations: Mapping[str, float | np.ndarray],
    total: float | np.ndarray,
    atol: float | None = None,
    where: str = "",
) -> float | np.ndarray | None:
    """
    Check that the populations of fragments add up to the total, sum_X N_X = N, and warn
    for the densities whose populations do not.

    For a density matrix D and `FragmentOperators` the populations are Tr(D W_X) and the
    total is the electron count Tr(D S); a closure error then means that the quadrature
    grid or the fragment weights are not accurate enough for that density. Fragment
    charges and the total charge, e.g. integrated from cube files, work the same way.

    Args:
        populations (Mapping[str, float | np.ndarray]): Population of each fragment,
            a float or an array with one value per density.
        total (float | np.ndarray): Total population, with the shape of each population.
        atol (float | None, optional): Largest ``|sum_X N_X - N|`` allowed, in electrons.
            Defaults to the `check_closure_atol` option; the check is off if that is None.
        where (str, optional): What the populations belong to, for the message.
            Defaults to "".

    Raises:
        ValueError: If the shapes do not match, or the tolerance is not a number >= 0.

    Returns:
        float | np.ndarray | None: The closure error ``sum_X N_X - N``, a float or an
            array with one value per density; None if the check is off.
    """

    atol = _tolerance(atol, "check_closure_atol")
    if atol is None:
        return None

    if not isinstance(populations, Mapping) or not populations:
        raise ValueError("Populations should be a nonempty mapping of fragments!")

    total = np.asarray(total, dtype=float)
    closure = -total
    for name, population in populations.items():
        population = np.asarray(population, dtype=float)
        if population.shape != total.shape:
            raise ValueError(
                f"Population of fragment `{name}` has shape {population.shape}, "
                f"the total has {total.shape}!"
            )
        closure = closure + population

    failed = ~(np.abs(closure) <= atol)  # NaN fails
    if np.any(failed):
        if closure.ndim == 0:
            which, worst = "", f"{float(closure):+.1e}"
        else:
            indices = np.flatnonzero(failed)
            listed = ", ".join(str(i) for i in indices[:_LISTED_INDICES])
            if len(indices) > _LISTED_INDICES:
                listed += ", ..."
            label = "index" if len(indices) == 1 else "indices"
            which = (
                f" for {len(indices)} of {closure.size} densities ({label} {listed})"
            )
            largest = closure.reshape(-1)[indices[np.argmax(np.abs(closure[failed]))]]
            worst = f"{'' if len(indices) == 1 else 'up to '}{largest:+.1e}"

        of = f" of {where}" if where else ""
        _warn(
            f"The populations of fragments {', '.join(map(str, populations))}{of} do "
            f"not add up to the total{which}: sum - total = {worst} e, above {atol:.1e} "
            "(option `check_closure_atol`). For a density matrix D the total is "
            "Tr(D S): the quadrature grid or the fragment weights are not accurate "
            "enough for it.",
        )

    return float(closure) if closure.ndim == 0 else closure


def _tolerance(atol, option_key: str) -> float | None:
    """
    Returns the tolerance an invariant check uses: `atol` if given, else the propSAPT
    option `option_key`, validated as a finite number >= 0.

    Every check of this module resolves its tolerance here, so that all of them treat it
    alike. The package runs the checks without `atol` (e.g. in `calc_density_matrix` and
    `FragmentOperators`), so the option is how a user tightens, loosens or turns off a
    check inside a calculation; `atol` serves direct calls of the checks. Options are
    stored unvalidated, so their values are checked here, when a check runs: otherwise
    True would act as 1.0, a negative tolerance would always warn and NaN never.

    Args:
        atol (float | None): Tolerance passed to the check; None takes the option. A
            None here cannot turn the check off, only the option can.
        option_key (str): Name of the option that holds the check's tolerance,
            e.g. "check_closure_atol".

    Raises:
        ValueError: If the tolerance is not a finite real number >= 0 (a bool is not).

    Returns:
        float | None: The tolerance as a float; None if `atol` and the option are both
            None, i.e. the check is off.
    """

    if atol is None:
        atol = PROPSAPT_OPTIONS[option_key]
        if atol is None:
            return None

    if (
        isinstance(atol, bool)
        or not isinstance(atol, (int, float, np.integer, np.floating))
        or not 0.0 <= atol < np.inf
    ):
        raise ValueError(
            f"The tolerance should be a finite number >= 0 (option `{option_key}`), "
            f"was {atol!r}!"
        )

    return float(atol)


def _warn(message: str):
    """Warns `InvariantWarning`, attributed to the first caller outside propSAPT, and
    writes `message` into the Psi4 output file."""

    # stack level 1 is this function; count the frames inside the package above it
    stacklevel = 1
    frame = sys._getframe(0)
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(
        _PACKAGE_DIR
    ):
        frame = frame.f_back
        stacklevel += 1

    warnings.warn(message, InvariantWarning, stacklevel=stacklevel)
    psi4.core.print_out(f"\nWarning: {message}\n")
