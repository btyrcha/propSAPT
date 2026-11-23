"""The backend contract: what a quantum-chemistry code supplies to propSAPT.

A backend is a finished reference calculation of a dimer: the SCFs of both monomers in
the dimer-centred basis set (DCBS) and the integrals of the dimer, as NumPy arrays, with
the operations propSAPT calls while it calculates. `helper_SAPT` reads only these, so
the SAPT contractions work with any code that supplies them.

Conventions every backend follows, where the arrays enter or leave propSAPT:

- AO arrays are in the dimer basis set, monomer ghosts and extra fragments included,
  in the backend's own AO order; the contractions work in the MO basis, so the order
  cancels inside propSAPT and only matters for AO matrices passed in or handed out.
- Orbitals are the columns of the MO coefficients `C`, occupied first.
- `V` of a monomer is the attraction of an electron by that monomer's nuclei, -Z/r
  summed over its atoms.
- `Qpq` is a factor of the density-fitted two-electron integrals in the fitting basis
  of the SAPT terms (Psi4's `DF_BASIS_SAPT`): sum_Q Q_pq Q_rs = (pq|rs). Any factor
  works, e.g. J^(-1/2) (P|pq) or a Cholesky one.
"""

from abc import ABC
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..fragments import DimerFragments

#: Monomer labels of a dimer.
MONOMERS = ("A", "B")


def check_reference(reference: str) -> str:
    """
    Check that propSAPT supports a reference.

    Args:
        reference (str): "RHF" or "RKS"; "ROHF", "UHF" and "UKS" are not implemented yet.

    Raises:
        ValueError: If the reference is unknown or not implemented yet.

    Returns:
        str: The reference.
    """

    if reference not in ["RHF", "ROHF", "UHF", "RKS", "UKS"]:
        raise ValueError(f"Unsupported reference type '{reference}'.")

    if reference in ["ROHF", "UHF", "UKS"]:
        # NOTE: Future upgrades
        raise ValueError(f"Reference '{reference}' not implemented yet.")

    return reference


@dataclass(frozen=True, eq=False)
class MonomerReference:
    """
    The SCF of one monomer in the dimer-centred basis set.
    """

    #: MO coefficients, (nbf, nmo): doubly occupied, singly occupied, virtual orbitals.
    C: np.ndarray
    #: Orbital energies, (nmo,).
    eps: np.ndarray
    #: Number of doubly occupied orbitals.
    ndocc: int
    #: Attraction of an electron by the monomer's nuclei in the AO basis, (nbf, nbf).
    V: np.ndarray
    #: SCF energy.
    energy: float
    #: Repulsion of the monomer's nuclei.
    nuclear_repulsion: float
    #: Number of singly occupied orbitals, 0 for closed-shell references.
    nsocc: int = 0

    def __post_init__(self):
        nbf, nmo = _shape(self.C, 2, "C")
        if _shape(self.eps, 1, "eps") != (nmo,):
            raise ValueError(
                f"Orbital energies should have shape ({nmo},), had {self.eps.shape}!"
            )
        if _shape(self.V, 2, "V") != (nbf, nbf):
            raise ValueError(
                f"V should have shape ({nbf}, {nbf}), as the AOs of C, "
                f"had {self.V.shape}!"
            )
        if not 0 <= self.ndocc + self.nsocc <= nmo or min(self.ndocc, self.nsocc) < 0:
            raise ValueError(
                f"{self.ndocc} doubly and {self.nsocc} singly occupied orbitals do not "
                f"fit in {nmo} orbitals!"
            )

    @property
    def nbf(self) -> int:
        """Number of basis functions."""
        return self.C.shape[0]

    @property
    def nmo(self) -> int:
        """Number of molecular orbitals."""
        return self.C.shape[1]


class Backend(ABC):
    """
    A reference calculation of a dimer by a quantum-chemistry code, with the operations
    propSAPT needs from that code. Subclasses run the calculation and pass its results
    to this constructor, which checks them.

    Args:
        reference (str): "RHF" or "RKS", see `check_reference`.
        monomers (Mapping[str, MonomerReference]): SCFs of monomers "A" and "B".
        S (np.ndarray): AO overlap matrix, (nbf, nbf).
        Qpq (np.ndarray): Density-fitting factor of the two-electron integrals,
            (naux, nbf, nbf), see the module conventions.
        nuclear_repulsion (float): Repulsion of all nuclei of the dimer.
        fragments (DimerFragments): Atoms and fragments of the dimer, in the frame of
            the calculation.
        functional (str | None, optional): Functional of an RKS reference; None for
            other references, whatever is passed. Defaults to None.
        grac_A (float | None, optional): GRAC shift of monomer A. Defaults to None.
        grac_B (float | None, optional): GRAC shift of monomer B. Defaults to None.

    Raises:
        ValueError: If the reference is not supported, or the arrays are inconsistent.
    """

    def __init__(
        self,
        *,
        reference: str,
        monomers: Mapping[str, MonomerReference],
        S: np.ndarray,
        Qpq: np.ndarray,
        nuclear_repulsion: float,
        fragments: "DimerFragments",
        functional: str | None = None,
        grac_A: float | None = None,
        grac_B: float | None = None,
    ):

        self.reference = check_reference(reference)
        self.functional = functional if reference == "RKS" else None
        self.grac_A = grac_A
        self.grac_B = grac_B

        if set(monomers) != set(MONOMERS):
            raise ValueError(
                f"Monomers should be {list(MONOMERS)}, were {sorted(monomers)}!"
            )
        for label in MONOMERS:
            if not isinstance(monomers[label], MonomerReference):
                raise ValueError(
                    f"Monomer {label} should be a MonomerReference, "
                    f"was {type(monomers[label]).__name__}!"
                )

        nbf = _shape(S, 2, "S")[0]
        if S.shape != (nbf, nbf):
            raise ValueError(f"S should be square, had shape {S.shape}!")
        if _shape(Qpq, 3, "Qpq")[1:] != (nbf, nbf):
            raise ValueError(
                f"Qpq should have shape (naux, {nbf}, {nbf}), as S, had {Qpq.shape}!"
            )
        for label in MONOMERS:
            if monomers[label].nbf != nbf:
                raise ValueError(
                    f"Monomer {label} has {monomers[label].nbf} basis functions, "
                    f"the dimer {nbf}: monomers use the dimer basis set!"
                )

        self.monomers = dict(monomers)
        self.S = S
        self.Qpq = Qpq
        self.nuclear_repulsion = nuclear_repulsion
        self.fragments = fragments

    @property
    def nbf(self) -> int:
        """Number of basis functions of the dimer basis set."""
        return self.S.shape[0]

    def clean_scratch(self) -> None:
        """Free what the backend keeps on disk between calculations, at the end of a
        job such as `calc_sapt_energy`. Nothing by default."""

    def timer(self) -> AbstractContextManager[None]:
        """Context timed by the backend's own timers, if it has any; propSAPT logs its
        timings either way. None by default."""

        return nullcontext()


def _shape(array: np.ndarray, ndim: int, name: str) -> tuple[int, ...]:
    """Shape of `array`, checked to have `ndim` dimensions."""

    if not isinstance(array, np.ndarray) or array.ndim != ndim:
        found = array.shape if isinstance(array, np.ndarray) else type(array).__name__
        raise ValueError(f"{name} should be a {ndim}-dimensional array, was {found}!")

    return array.shape
