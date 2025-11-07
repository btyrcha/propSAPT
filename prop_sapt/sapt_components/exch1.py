import numpy as np
import opt_einsum as oe

from ..molecule import Dimer
from ..utils import CalcTimer


def calc_exch1_s2_energy(dimer: Dimer):
    """Calculate the first-order SAPT exchange energy in S2 approximation.

    Args:
        dimer (Dimer): Dimer object containing the necessary matrices and tensors.

    Returns:
        _type_: First-order SAPT exchange energy in S2 approximation.
    """

    with CalcTimer("Exchange (S^2) energy calculation"):

        s_ab = dimer.s("ab")
        s_ba = dimer.s("ba")
        s_sa = dimer.s("sa")
        s_rb = dimer.s("rb")

        _, K1 = dimer.compute_jk(
            dimer.orbitals["r"],
            dimer.orbitals["b"],
            tensor=s_rb,
        )

        ### Exch100_S2
        exch1_s2 = -2 * oe.contract("ab,sa,bs", s_ab, s_sa, dimer.omegaA_bs)
        exch1_s2 -= 2 * oe.contract("ba,rb,ar", s_ba, s_rb, dimer.omegaB_ar)
        exch1_s2 -= 2 * np.trace(
            dimer.orbitals["s"] @ s_sa @ dimer.orbitals["a"].T @ K1
        )

    return exch1_s2


def calc_exch1_energy(dimer: Dimer):
    """Calculate the first-order SAPT exchange energy with Sinfinity.

    Args:
        dimer (Dimer): Dimer object containing the necessary matrices and tensors.

    Returns:
        _type_: First-order SAPT exchange energy with Sinfinity.
    """

    ### Exch100_Sinfinity
    with CalcTimer("Exchange (Sinf) energy calculation"):

        _, K1 = dimer.compute_jk(
            dimer.orbitals["r"],
            dimer.orbitals["b"],
            tensor=dimer.I_rb,
        )
        J2, _ = dimer.compute_jk(
            dimer.orbitals["s"],
            dimer.orbitals["b"],
            tensor=dimer.F_sb,
        )

        exch1_sinf = -2 * oe.contract("sb,bs", dimer.F_sb, dimer.omegaA_bs)
        exch1_sinf -= 2 * oe.contract("ra,ar", dimer.E_ra, dimer.omegaB_ar)
        exch1_sinf -= 2 * np.trace(
            dimer.orbitals["s"] @ dimer.J_sa @ dimer.orbitals["a"].T @ K1
        )
        exch1_sinf += 4 * np.trace(
            dimer.orbitals["r"] @ dimer.E_ra @ dimer.orbitals["a"].T @ J2
        )

    return exch1_sinf
