"""
Evaluation of densities, orbitals and the electrostatic potential at arbitrary points.

All functions take `points` in bohr, as an array of shape (..., 3), and the Psi4 basis
set in which the matrices are expressed (Psi4 AO order, `basisset.nbf()` functions).
The values are returned with the shape of `points` without its last axis, preceded by
one axis over the matrices or orbitals when several are given.

Basis function values are computed for blocks of `chunk_size` points at a time, which
bounds the memory to `chunk_size * nbf` values; the default is the
`point_eval_chunk_size` option.

Functions
---------
evaluate_basis_functions
    Values of all basis functions at the points.
evaluate_density
    Density of one or more AO density matrices at the points.
evaluate_orbital
    Values of one or more orbitals (columns of MO coefficients) at the points.
evaluate_esp
    Electrostatic potential of one or more AO density matrices at the points.
"""

from collections.abc import Iterator
from time import time
from typing import Sequence
import numpy as np
import psi4

from .options import PROPSAPT_OPTIONS


def evaluate_basis_functions(
    points: np.ndarray,
    basisset: psi4.core.BasisSet,
    chunk_size: int | None = None,
) -> np.ndarray:
    """
    Evaluate all basis functions of `basisset` at `points`.

    Args:
        points (np.ndarray): Points in bohr, shape (..., 3).
        basisset (psi4.core.BasisSet): Basis set to evaluate.
        chunk_size (int | None, optional): Number of points evaluated at once.
            Defaults to the `point_eval_chunk_size` option.

    Returns:
        np.ndarray: Basis function values, shape (..., nbf), in Psi4 AO order.
    """

    _check_basisset(basisset)
    points = _prepare_points(points)
    chunk_size = _get_chunk_size(chunk_size)

    flat_points = points.reshape(-1, 3)
    phi = np.empty((len(flat_points), basisset.nbf()))
    for chunk, phi_chunk in _iter_phi_chunks(flat_points, basisset, chunk_size):
        phi[chunk] = phi_chunk

    return phi.reshape(points.shape[:-1] + (basisset.nbf(),))


def evaluate_density(
    dm: np.ndarray | Sequence[np.ndarray],
    points: np.ndarray,
    basisset: psi4.core.BasisSet,
    chunk_size: int | None = None,
) -> np.ndarray:
    """
    Evaluate the density of AO density matrix `dm` at `points`,
    rho(r) = sum_{mu,nu} phi_mu(r) dm[mu,nu] phi_nu(r).

    No spin factor is applied: for the total density of an RHF wavefunction pass `2 * Da`.

    Args:
        dm (np.ndarray | Sequence[np.ndarray]): AO density matrix, shape (nbf, nbf),
            or several of them, as a sequence or an array of shape (M, nbf, nbf).
        points (np.ndarray): Points in bohr, shape (..., 3).
        basisset (psi4.core.BasisSet): Basis set of `dm`.
        chunk_size (int | None, optional): Number of points evaluated at once.
            Defaults to the `point_eval_chunk_size` option.

    Returns:
        np.ndarray: Density at the points, shape (...) for a single matrix
            and (M, ...) for several.
    """

    _check_basisset(basisset)
    dm_stack, single = _stack_matrices(dm, basisset.nbf())
    points = _prepare_points(points)
    chunk_size = _get_chunk_size(chunk_size)

    densities, _ = _evaluate_on_points(
        points.reshape(-1, 3), basisset, chunk_size, density_stack=dm_stack
    )

    return _shape_output(densities, single, points.shape[:-1])


def evaluate_orbital(
    coeffs: np.ndarray,
    points: np.ndarray,
    basisset: psi4.core.BasisSet,
    chunk_size: int | None = None,
) -> np.ndarray:
    """
    Evaluate orbitals given by MO coefficients `coeffs` at `points`,
    psi_i(r) = sum_mu phi_mu(r) coeffs[mu,i].

    Orbitals are the columns of `coeffs`, as in Psi4's `Ca()`:
    pass `C[:, i]` for orbital `i`, or `C[:, i:j]` for several.

    Args:
        coeffs (np.ndarray): MO coefficients, shape (nbf,) for one orbital
            or (nbf, k) for k orbitals.
        points (np.ndarray): Points in bohr, shape (..., 3).
        basisset (psi4.core.BasisSet): Basis set of `coeffs`.
        chunk_size (int | None, optional): Number of points evaluated at once.
            Defaults to the `point_eval_chunk_size` option.

    Returns:
        np.ndarray: Orbital values at the points, shape (...) for `coeffs` of shape (nbf,)
            and (k, ...) for `coeffs` of shape (nbf, k).
    """

    _check_basisset(basisset)
    orbital_columns, single = _stack_orbitals(coeffs, basisset.nbf())
    points = _prepare_points(points)
    chunk_size = _get_chunk_size(chunk_size)

    _, orbitals = _evaluate_on_points(
        points.reshape(-1, 3), basisset, chunk_size, orbital_columns=orbital_columns
    )

    return _shape_output(orbitals, single, points.shape[:-1])


def evaluate_esp(
    dm: np.ndarray | Sequence[np.ndarray],
    points: np.ndarray,
    basisset: psi4.core.BasisSet,
    nuclei: bool = False,
) -> np.ndarray:
    """
    Evaluate the electrostatic potential of AO density matrix `dm` at `points`,
    from analytic potential integrals. Electrons contribute negatively.

    As for `evaluate_density`, no spin factor is applied to `dm`.

    Args:
        dm (np.ndarray | Sequence[np.ndarray]): AO density matrix, shape (nbf, nbf),
            or several of them, as a sequence or an array of shape (M, nbf, nbf).
        points (np.ndarray): Points in bohr, shape (..., 3).
        basisset (psi4.core.BasisSet): Basis set of `dm`.
        nuclei (bool, optional): Add the potential of the nuclei of `basisset.molecule()`,
            use it for total (not interaction-induced) densities. Ghost atoms carry no
            charge, and a nucleus lying exactly on a point is skipped at that point.
            Defaults to False.

    Returns:
        np.ndarray: Electrostatic potential at the points, shape (...) for a single matrix
            and (M, ...) for several.
    """

    _check_basisset(basisset)
    dm_stack, single = _stack_matrices(dm, basisset.nbf())
    points = _prepare_points(points)

    flat_points = points.reshape(-1, 3)
    mints = psi4.core.MintsHelper(basisset)
    esp = _electronic_esp(flat_points, mints, dm_stack)
    if nuclei:
        esp += _nuclear_potential(flat_points, basisset.molecule())

    return _shape_output(esp, single, points.shape[:-1])


def _check_basisset(basisset):
    """Raises `TypeError` if `basisset` is not a Psi4 basis set."""

    if not isinstance(basisset, psi4.core.BasisSet):
        raise TypeError(
            f"`basisset` should be of type `psi4.core.BasisSet` was `{type(basisset)}`!"
        )


def _prepare_points(points) -> np.ndarray:
    """Returns `points` as a float array, checking that its last axis has length 3."""

    points = np.asarray(points, dtype=float)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError(f"`points` should have shape (..., 3), was {points.shape}!")

    return points


def _get_chunk_size(chunk_size) -> int:
    """Returns `chunk_size`, or the `point_eval_chunk_size` option if None."""

    if chunk_size is None:
        chunk_size = PROPSAPT_OPTIONS["point_eval_chunk_size"]

    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError(
            f"`chunk_size` should be a positive integer, was {chunk_size}!"
        )

    return int(chunk_size)


def _stack_matrices(dm, nbf: int) -> tuple[np.ndarray, bool]:
    """
    Stacks AO matrices into an array of shape (M, nbf, nbf).

    Returns:
        tuple[np.ndarray, bool]: The stack, and whether a single matrix was given.
    """

    if isinstance(dm, (np.ndarray, psi4.core.Matrix)):
        stack = np.asarray(dm, dtype=float)
        single = stack.ndim == 2
        if single:
            stack = stack[None]
    else:
        matrices = [np.asarray(m, dtype=float) for m in dm]
        if len(matrices) == 0:
            raise ValueError("No matrices given! `dm` must be nonempty.")
        for m in matrices:
            if m.shape != (nbf, nbf):
                raise ValueError(
                    f"Every matrix should have shape ({nbf}, {nbf}) of the basis set, "
                    f"was {m.shape}!"
                )
        stack = np.stack(matrices)
        single = False

    if stack.ndim != 3 or stack.shape[1:] != (nbf, nbf):
        raise ValueError(
            f"`dm` should have shape ({nbf}, {nbf}) or (M, {nbf}, {nbf}) of the basis set, "
            f"was {stack.shape}!"
        )

    return stack, single


def _stack_orbitals(coeffs, nbf: int) -> tuple[np.ndarray, bool]:
    """
    Returns MO coefficients as columns, an array of shape (nbf, k).

    Returns:
        tuple[np.ndarray, bool]: The columns, and whether a single vector was given.
    """

    columns = np.asarray(coeffs, dtype=float)
    single = columns.ndim == 1
    if single:
        columns = columns[:, None]

    if columns.ndim != 2 or columns.shape[0] != nbf:
        raise ValueError(
            f"`coeffs` should have shape ({nbf},) or ({nbf}, k) of the basis set, "
            f"orbitals being columns as in Psi4's `Ca()`, was {np.shape(coeffs)}!"
        )

    return columns, single


def _shape_output(values: np.ndarray, single: bool, points_shape: tuple) -> np.ndarray:
    """Reshapes values of shape (M, n_points) to (M, *points_shape), dropping M if single."""

    values = values.reshape((len(values),) + points_shape)

    return values[0] if single else values


def _iter_phi_chunks(
    points: np.ndarray, basisset: psi4.core.BasisSet, chunk_size: int
) -> Iterator[tuple[slice, np.ndarray]]:
    """
    Yields basis function values for consecutive chunks of `points`.

    Args:
        points (np.ndarray): Points in bohr, shape (n, 3).
        basisset (psi4.core.BasisSet): Basis set to evaluate.
        chunk_size (int): Number of points per chunk.

    Yields:
        tuple[slice, np.ndarray]: Slice of `points` in the chunk and the basis function
            values there, shape (n_chunk, nbf).
    """

    n_points = len(points)
    if n_points == 0:
        return

    nbf = basisset.nbf()
    # delta = 0 switches the screening off, any positive cutoff zeroes distant functions
    extents = psi4.core.BasisExtents(basisset, 0.0)
    functions = psi4.core.BasisFunctions(basisset, min(chunk_size, n_points), nbf)

    for start in range(0, n_points, chunk_size):
        chunk_points = points[start : start + chunk_size]
        n_chunk = len(chunk_points)

        block = psi4.core.BlockOPoints(
            psi4.core.Vector.from_array(np.ascontiguousarray(chunk_points[:, 0])),
            psi4.core.Vector.from_array(np.ascontiguousarray(chunk_points[:, 1])),
            psi4.core.Vector.from_array(np.ascontiguousarray(chunk_points[:, 2])),
            psi4.core.Vector.from_array(
                np.zeros(n_chunk)
            ),  # quadrature weights, unused
            extents,
        )
        functions.compute_functions(block)

        # values come for the block's local functions, scatter them into Psi4 AO order
        local_to_global = np.asarray(block.functions_local_to_global(), dtype=int)
        phi = np.zeros((n_chunk, nbf))
        phi[:, local_to_global] = np.asarray(functions.basis_values()["PHI"])[
            :n_chunk, : local_to_global.size
        ]

        yield slice(start, start + n_chunk), phi


def _evaluate_on_points(
    points: np.ndarray,
    basisset: psi4.core.BasisSet,
    chunk_size: int,
    density_stack: np.ndarray | None = None,
    orbital_columns: np.ndarray | None = None,
    timings: dict | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Evaluates densities and orbitals at `points`, computing basis function values
    once per chunk for all of them.

    Args:
        points (np.ndarray): Points in bohr, shape (n, 3).
        basisset (psi4.core.BasisSet): Basis set of the matrices.
        chunk_size (int): Number of points evaluated at once.
        density_stack (np.ndarray | None, optional): AO density matrices,
            shape (M, nbf, nbf). Defaults to None.
        orbital_columns (np.ndarray | None, optional): MO coefficients as columns,
            shape (nbf, k). Defaults to None.
        timings (dict | None, optional): If given, the time spent on basis function
            values and on contractions is added to its "phi" and "contract" entries.
            Defaults to None.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: Densities, shape (M, n), and orbital
            values, shape (k, n), or None where no matrices were given.
    """

    n_points = len(points)
    densities = (
        np.empty((len(density_stack), n_points)) if density_stack is not None else None
    )
    orbitals = (
        np.empty((orbital_columns.shape[1], n_points))
        if orbital_columns is not None
        else None
    )

    t_phi = 0.0
    t_contract = 0.0
    _t0 = time()
    for chunk, phi in _iter_phi_chunks(points, basisset, chunk_size):
        _t1 = time()
        t_phi += _t1 - _t0

        if density_stack is not None:
            # one matrix at a time keeps the intermediate at (n_chunk, nbf)
            for m_idx, dm in enumerate(density_stack):
                densities[m_idx, chunk] = np.einsum("pi,pi->p", phi @ dm, phi)

        if orbital_columns is not None:
            orbitals[:, chunk] = (phi @ orbital_columns).T

        _t0 = time()
        t_contract += _t0 - _t1

    if timings is not None:
        timings["phi"] = timings.get("phi", 0.0) + t_phi
        timings["contract"] = timings.get("contract", 0.0) + t_contract

    return densities, orbitals


def _electronic_esp(
    points: np.ndarray, mints: psi4.core.MintsHelper, dm_stack: np.ndarray
) -> np.ndarray:
    """
    Electrostatic potential of the electrons of each AO density matrix at `points`.

    Args:
        points (np.ndarray): Points in bohr, shape (n, 3).
        mints (psi4.core.MintsHelper): Integral helper built on the matrices' basis set.
        dm_stack (np.ndarray): AO density matrices, shape (M, nbf, nbf).

    Returns:
        np.ndarray: Potential, shape (M, n).
    """

    esp = np.zeros((len(dm_stack), len(points)))
    if len(points) == 0:
        return esp

    psi4_points = psi4.core.Matrix.from_array(np.ascontiguousarray(points))
    # unit probe charge at every point, Psi4 returns the electronic potential only
    probe_charges = psi4.core.Vector.from_array(np.ones(len(points)))

    for m_idx, dm in enumerate(dm_stack):
        esp[m_idx] = np.asarray(
            mints.electrostatic_potential_value(
                probe_charges, psi4_points, psi4.core.Matrix.from_array(dm)
            )
        )

    return esp


def _nuclear_potential(points: np.ndarray, molecule: psi4.core.Molecule) -> np.ndarray:
    """
    Electrostatic potential of the nuclei of `molecule` at `points`,
    sum_A Z_A / |r - R_A|. A nucleus lying exactly on a point is skipped at that point.

    Args:
        points (np.ndarray): Points in bohr, shape (n, 3).
        molecule (psi4.core.Molecule): Molecule with the nuclei, ghost atoms have no charge.

    Returns:
        np.ndarray: Potential, shape (n,).
    """

    nuclei = np.asarray(molecule.geometry())  # in bohr
    potential = np.zeros(len(points))

    # one nucleus at a time keeps the memory at the number of points
    for atom_idx, nucleus in enumerate(nuclei):
        charge = molecule.Z(atom_idx)
        if charge == 0.0:
            continue

        dist = np.linalg.norm(points - nucleus, axis=1)
        dist[dist < 1.0e-10] = np.inf  # skip a nucleus on the point
        potential += charge / dist

    return potential
