import numpy as np
import opt_einsum as oe

from ..molecule import Dimer
from ..utils import CalcTimer


def calc_exch1_s2_energy(mol: Dimer):
    """Calculate the first-order SAPT exchange energy in S2 approximation.

    Args:
        mol (Dimer): Dimer object containing the necessary matrices and tensors.

    Returns:
        _type_: First-order SAPT exchange energy in S2 approximation.
    """

    with CalcTimer("Exchange (S^2) energy calculation"):

        s_ab = mol.s("ab")
        s_ba = mol.s("ba")
        s_sa = mol.s("sa")
        s_rb = mol.s("rb")

        _, K1 = mol.compute_jk(
            mol.orbitals["r"],
            mol.orbitals["b"],
            tensor=s_rb,
        )

        ### Exch100_S2
        exch1_s2 = -2 * oe.contract("ab,sa,bs", s_ab, s_sa, mol.omegaA_bs)
        exch1_s2 -= 2 * oe.contract("ba,rb,ar", s_ba, s_rb, mol.omegaB_ar)
        exch1_s2 -= 2 * np.trace(mol.orbitals["s"] @ s_sa @ mol.orbitals["a"].T @ K1)

    return exch1_s2


def calc_exch1_energy(mol: Dimer):
    """Calculate the first-order SAPT exchange energy with Sinfinity.

    Args:
        mol (Dimer): Dimer object containing the necessary matrices and tensors.

    Returns:
        _type_: First-order SAPT exchange energy with Sinfinity.
    """

    ### Exch100_Sinfinity
    with CalcTimer("Exchange (Sinf) energy calculation"):

        _, K1 = mol.compute_jk(
            mol.orbitals["r"],
            mol.orbitals["b"],
            tensor=mol.I_rb,
        )
        J2, _ = mol.compute_jk(
            mol.orbitals["s"],
            mol.orbitals["b"],
            tensor=mol.F_sb,
        )

        exch1_sinf = -2 * oe.contract("sb,bs", mol.F_sb, mol.omegaA_bs)
        exch1_sinf -= 2 * oe.contract("ra,ar", mol.E_ra, mol.omegaB_ar)
        exch1_sinf -= 2 * np.trace(
            mol.orbitals["s"] @ mol.J_sa @ mol.orbitals["a"].T @ K1
        )
        exch1_sinf += 4 * np.trace(
            mol.orbitals["r"] @ mol.E_ra @ mol.orbitals["a"].T @ J2
        )

    return exch1_sinf
