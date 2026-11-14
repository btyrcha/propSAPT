"""
Natural orbitals of the interaction-induced density changes.

The density change ``dP_X`` of monomer X (AO basis, one spin) lies in the span of the
monomer's orbitals ``C_X``, which are orthonormal (``C_X^T S C_X = 1``). In that basis it
is the symmetric matrix ``D_X = C_X^T S dP_X S C_X``. The eigenvectors ``U`` of ``D_X``
give the natural orbitals ``C_X U``, and its eigenvalues ``n`` their occupations. The
density change is then exactly ``dP_X = (C_X U) diag(n) (C_X U)^T``.

- **Occupations:** they are for one spin, like the matrices of `calc_densities`, so the
  total density change is twice theirs. They sum to ``Tr(dP_X S) = 0``: a density change
  moves charge without changing the number of electrons.
- **Signs:** orbitals of positive occupation show where charge goes, and those of negative
  occupation where it comes from.
- **Reading them:** a few orbitals usually carry most of a density change, so they picture
  it in the spirit of natural orbitals for chemical valence (NOCV). First-order
  polarisation and exchange mix only occupied with virtual orbitals, so their occupations
  come in +/- pairs.
- **Dimer keys:** for the sum of both monomers (e.g. "pol"), the orbitals of A and of B
  are listed together. The two sets are not orthogonal to each other, but their density is
  exactly the sum.

Classes
-------
NaturalOrbitals
    Occupations and AO coefficients of natural orbitals, with the monomer of each.

Functions
---------
natural_orbitals
    Natural orbitals of one monomer's density change.
density_natural_orbitals
    Natural orbitals of the density changes of `calc_densities` or of a bundle.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .artifacts import DensitySet

#: Largest ``|dP - C D C^T|`` allowed by `natural_orbitals`, relative to ``max|dP|``: a
#: density change of a monomer lies in the span of its orbitals, up to round-off.
SPAN_RTOL = 1.0e-8

_MONOMERS = ("A", "B")


@dataclass(frozen=True, eq=False)
class NaturalOrbitals:
    """
    Natural orbitals of a density change, sorted by decreasing occupation.
    """

    #: Occupations for one spin, shape (k,): the total density change is twice theirs.
    occupations: np.ndarray
    #: AO coefficients, shape (nbf, k), one orbital per column as in Psi4's ``Ca()``.
    coefficients: np.ndarray
    #: Monomer of each orbital, "A" or "B", shape (k,).
    monomers: np.ndarray

    def __len__(self) -> int:
        return len(self.occupations)

    def density(self) -> np.ndarray:
        """
        The density change of the orbitals, ``sum_k n_k c_k c_k^T`` (AO basis, one spin).

        Returns:
            np.ndarray: AO matrix, shape (nbf, nbf).
        """

        return (self.coefficients * self.occupations) @ self.coefficients.T

    def select(self, cutoff: float) -> "NaturalOrbitals":
        """
        The orbitals whose occupation is larger than `cutoff` in magnitude.

        Args:
            cutoff (float): Smallest ``|n|`` kept (one spin); smaller ones are dropped.

        Returns:
            NaturalOrbitals: The orbitals kept, in the same order.
        """

        keep = np.abs(self.occupations) > cutoff
        return NaturalOrbitals(
            self.occupations[keep], self.coefficients[:, keep], self.monomers[keep]
        )


def natural_orbitals(
    density: np.ndarray, orbitals: np.ndarray, overlap: np.ndarray, monomer: str = "A"
) -> NaturalOrbitals:
    """
    Natural orbitals of a monomer's density change, in the basis of its orbitals.

    Args:
        density (np.ndarray): Density change, AO basis, shape (nbf, nbf).
        orbitals (np.ndarray): The monomer's orthonormal orbitals ``C`` (e.g. `Dimer.C_A`),
            shape (nbf, nmo).
        overlap (np.ndarray): AO overlap matrix ``S`` (e.g. `Dimer.S`), shape (nbf, nbf).
        monomer (str, optional): Monomer recorded for the orbitals. Defaults to "A".

    Raises:
        ValueError: If the shapes do not match, or if `density` does not lie in the span of
            `orbitals` (to `SPAN_RTOL`). That can only happen when the monomer has fewer
            orbitals than basis functions (linear dependencies removed): with as many, every
            matrix lies in their span, so the check cannot tell an MO-basis matrix from an AO
            one.

    Returns:
        NaturalOrbitals: As many orbitals as `orbitals` has columns, by decreasing
            occupation.
    """

    density = np.asarray(density, dtype=np.float64)
    orbitals = np.asarray(orbitals, dtype=np.float64)
    overlap = np.asarray(overlap, dtype=np.float64)
    nbf = overlap.shape[0]
    if (
        density.shape != (nbf, nbf)
        or overlap.shape != (nbf, nbf)
        or orbitals.ndim != 2
        or orbitals.shape[0] != nbf
    ):
        raise ValueError(
            f"Shapes do not match: density {density.shape}, orbitals {orbitals.shape}, "
            f"overlap {overlap.shape}; the density and the overlap are AO matrices "
            "(nbf, nbf) and the orbitals (nbf, nmo)!"
        )

    projector = overlap @ orbitals
    in_orbitals = projector.T @ density @ projector
    in_orbitals = 0.5 * (in_orbitals + in_orbitals.T)

    deviation = np.max(np.abs(orbitals @ in_orbitals @ orbitals.T - density))
    if not deviation <= SPAN_RTOL * np.max(np.abs(density)):
        raise ValueError(
            f"The density change does not lie in the span of the orbitals of monomer "
            f"{monomer} (max deviation {deviation:.1e}): pass an AO matrix of that "
            "monomer's density change, as `calc_densities` returns them in the AO basis "
            "(the default)."
        )

    occupations, vectors = np.linalg.eigh(in_orbitals)
    order = np.argsort(-occupations, kind="stable")

    return NaturalOrbitals(
        occupations[order],
        orbitals @ vectors[:, order],
        np.full(len(occupations), monomer),
    )


def density_natural_orbitals(
    densities: Mapping[str, np.ndarray] | DensitySet,
    dimer=None,
    keys: Iterable[str] | None = None,
    cutoff: float = 0.0,
) -> dict[str, NaturalOrbitals]:
    """
    Natural orbitals of interaction-induced density changes.

    Args:
        densities (Mapping[str, np.ndarray] | DensitySet): The AO matrices returned by
            `calc_densities` (the default ``orbital_basis="AO"``), or a bundle read by
            `prop_sapt.load_densities`, whose monomer wavefunctions give the orbitals.
        dimer (Dimer | None, optional): The dimer of the matrices, for its orbitals and
            overlap; required with a dict of matrices, unused with a bundle.
            Defaults to None.
        keys (Iterable[str] | None, optional): Keys of the matrices, e.g. "pol_A" or
            "pol" for the dimer, which needs "pol_A" and "pol_B". Defaults to every key of
            `densities`.
        cutoff (float, optional): Orbitals with ``|n|`` up to `cutoff` (one spin) are
            dropped. Defaults to 0.0: every orbital is kept.

    Raises:
        TypeError: If `densities` is a dict of matrices and no `dimer` is given.
        KeyError: If a key, or a monomer matrix of a dimer key, is missing.
        ValueError: If a bundle has no monomer wavefunctions, or a matrix is not a density
            change of its monomer (see `natural_orbitals`).

    Returns:
        dict[str, NaturalOrbitals]: Natural orbitals by key. For a dimer key, those of A
            and B together, by decreasing occupation.
    """

    if isinstance(densities, DensitySet):
        matrices = densities.matrices
        orbitals, overlap = _bundle_orbitals(densities)
    else:
        if dimer is None:
            raise TypeError(
                "`dimer` is needed for its orbitals with a dict of density matrices!"
            )
        matrices = densities
        orbitals = {"A": np.asarray(dimer.C_A), "B": np.asarray(dimer.C_B)}
        overlap = np.asarray(dimer.S)

    keys = list(matrices) if keys is None else list(keys)

    result = {}
    for key in keys:
        sets = [
            natural_orbitals(matrix, orbitals[monomer], overlap, monomer=monomer)
            for monomer, matrix in _monomer_matrices(key, matrices)
        ]
        orbitals_of_key = _concatenate(sets)
        result[key] = orbitals_of_key.select(cutoff) if cutoff else orbitals_of_key

    return result


def _monomer_matrices(key: str, matrices: Mapping[str, np.ndarray]) -> list:
    """The (monomer, matrix) pairs of `key`: one for "<component>_A" or "_B", both
    monomers' for a dimer key "<component>"."""

    for monomer in _MONOMERS:
        if key.endswith(f"_{monomer}") and key in matrices:
            return [(monomer, matrices[key])]

    missing = [f"{key}_{monomer}" for monomer in _MONOMERS]
    missing = [name for name in missing if name not in matrices]
    if missing:
        raise KeyError(
            f"No density matrix {key!r} of a monomer, and the dimer key needs "
            f"{', '.join(repr(name) for name in missing)}; the keys are {sorted(matrices)}."
        )

    return [(monomer, matrices[f"{key}_{monomer}"]) for monomer in _MONOMERS]


def _concatenate(sets: list[NaturalOrbitals]) -> NaturalOrbitals:
    """The orbitals of `sets` together, by decreasing occupation."""

    if len(sets) == 1:
        return sets[0]

    occupations = np.concatenate([s.occupations for s in sets])
    order = np.argsort(-occupations, kind="stable")
    return NaturalOrbitals(
        occupations[order],
        np.concatenate([s.coefficients for s in sets], axis=1)[:, order],
        np.concatenate([s.monomers for s in sets])[order],
    )


def _bundle_orbitals(densities: DensitySet) -> tuple[dict, np.ndarray]:
    """The monomer orbitals (``Ca``) and the AO overlap of a bundle's wavefunctions."""

    missing = [m for m in _MONOMERS if m not in densities.metadata["wavefunctions"]]
    if missing:
        raise ValueError(
            f"Bundle '{densities.path}' has no wavefunction of monomer(s) "
            f"{', '.join(missing)}: natural orbitals need the monomer orbitals, which "
            "`calc_densities` saves with the matrices."
        )

    data = {monomer: densities.wavefunction_data(monomer) for monomer in _MONOMERS}
    orbitals = {
        monomer: np.asarray(data[monomer]["matrix"]["Ca"], dtype=np.float64)
        for monomer in _MONOMERS
    }
    overlap = np.asarray(data["A"]["matrix"]["S"], dtype=np.float64)

    return orbitals, overlap
