"""Hellmann-Feynman forces on nuclei from one-particle electron density matrices.

Functions
---------
calc_force_at_nuc
    Total force on a nucleus: electronic contribution via AO electric field integrals
    plus (optionally) nuclear-nuclear repulsion. Accepts a single density matrix or a
    sequence of matrices, computing the integrals only once.
calc_force_nuc_nuc
    Purely nuclear-nuclear repulsion force on a given nucleus (no electron density needed).
"""

from typing import Sequence
import numpy as np
import psi4


def calc_force_at_nuc(
    mol: psi4.core.Molecule,
    rho: np.ndarray | Sequence[np.ndarray],
    atidx: int,
    add_nuclear: bool = True,
    **kwargs,
) -> np.ndarray | list[np.ndarray]:
    """Computes the Hellmann-Feynman force on nucleus `atidx` from one or more electron density matrices.

    The electronic contribution is evaluated analytically via the electric field integrals
    at the nuclear position: ``F_i = Z_A * Tr[rho * <mu|(r_i - R_Ai)/|r - R_A|^3|nu>]``.
    The nuclear-nuclear repulsion is added by default.

    Accepts a single density matrix or a sequence of density matrices. When a sequence is
    provided, the electric field integrals are computed only once and reused for all matrices.

    Args:
        mol (psi4.core.Molecule): Psi4 molecule defining the geometry and nuclear charges.
        rho (np.ndarray | Sequence[np.ndarray]): One-particle density matrix in the AO basis,
            shape (nbf, nbf), or a sequence of such matrices.
        atidx (int): Zero-based index of the nucleus on which the force is computed.
        add_nuclear (bool, optional): If True, adds the nuclear-nuclear repulsion force.
            Set to False to obtain the electronic contribution only. Defaults to True.
        **kwargs: Accepts `basisset` (psi4.core.BasisSet) to reuse a pre-built basis set
            instead of constructing one from the molecule.

    Raises:
        ValueError: If `rho` is an empty sequence.
        TypeError: If `mol` is not a psi4.core.Molecule.

    Returns:
        np.ndarray | list[np.ndarray]: Force vector of shape (3,) in atomic units (Hartree/Bohr)
            when a single density matrix is given; a list of such vectors when a sequence is given.
    """

    # check and prepare `rho`
    return_single = False
    if isinstance(rho, np.ndarray):
        return_single = True
        rho = [rho]
    else:
        rho = list(rho)

    if len(rho) == 0:
        raise ValueError(
            "No density matrix given!\n"
            "`rho` must be either np.ndarray or Iterable of nonzero length."
        )

    # check mol and get the basisset
    if isinstance(mol, psi4.core.Molecule):
        if kwargs.get("basisset", None) is not None and isinstance(
            kwargs["basisset"], psi4.core.BasisSet
        ):
            basisset = kwargs["basisset"]
        else:
            basisset = psi4.core.Wavefunction.build(mol).basisset()

    else:
        raise TypeError(
            f"`mol` should be of type `psi4.core.Molecule` was `{type(mol)}`!"
        )

    # get molecular geometry and atomic numbers
    geom, _, _, elez, _ = mol.to_arrays()

    # prepare electric field operator integrals
    basisset.print_out()
    mints = psi4.core.MintsHelper(basisset)
    field_ints = [ints.to_array() for ints in mints.electric_field(geom[atidx])]

    # calculate force
    forces_list = [
        np.array(
            [elez[atidx] * np.einsum("ij,ji", rho_idx, field_ints[i]) for i in range(3)]
        )
        for rho_idx in rho
    ]

    # add nuclear-nuclear part
    if add_nuclear:
        force_nuc = calc_force_nuc_nuc(mol, atidx)
        for f in forces_list:
            f += force_nuc

    if return_single:
        return forces_list[0]

    return forces_list


def calc_force_nuc_nuc(mol: psi4.core.Molecule, atidx: int) -> np.ndarray:
    """Computes the nuclear-nuclear repulsion force on nucleus `atidx` from all other nuclei.

    Uses Coulomb's law: ``F_a = sum_{b != a} Z_a * Z_b * (R_a - R_b) / |R_a - R_b|^3``.

    Args:
        mol (psi4.core.Molecule): Psi4 molecule defining the geometry and nuclear charges.
        atidx (int): Zero-based index of the nucleus on which the force is computed.

    Returns:
        np.ndarray: Nuclear-nuclear force vector of shape (3,) in atomic units (Hartree/Bohr).
    """

    geom, _, _, elez, _ = mol.to_arrays()

    Z_a = elez[atidx]
    force = np.array([0.0, 0.0, 0.0])
    for i, Z_b in enumerate(elez):

        if i == atidx:
            continue

        R_ab = geom[i] - geom[atidx]

        force += -Z_a * Z_b * R_ab / np.linalg.vector_norm(R_ab) ** 3

    return force
