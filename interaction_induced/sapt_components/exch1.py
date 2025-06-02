import opt_einsum as oe

from ..molecule import Dimer


def calc_exch1_s2_energy(dimer: Dimer):
    """Calculate the first-order SAPT exchange energy in S2 approximation.

    Args:
        dimer (Dimer): Dimer object containing the necessary matrices and tensors.

    Returns:
        _type_: First-order SAPT exchange energy in S2 approximation.
    """

    s_ab = dimer.s("ab")
    s_ba = dimer.s("ba")
    s_sa = dimer.s("sa")
    s_rb = dimer.s("rb")

    ### Exch100_S2
    exch1_s2 = -2 * oe.contract("ab,sa,bs", s_ab, s_sa, dimer.omegaA_bs)
    exch1_s2 -= 2 * oe.contract("ba,rb,ar", s_ba, s_rb, dimer.omegaB_ar)
    exch1_s2 -= 2 * oe.contract("sa,rb,Qar,Qbs", s_sa, s_rb, dimer.Qar, dimer.Qbs)

    return exch1_s2


def calc_exch1_energy(dimer: Dimer):
    """Calculate the first-order SAPT exchange energy with Sinfinity.

    Args:
        dimer (Dimer): Dimer object containing the necessary matrices and tensors.

    Returns:
        _type_: First-order SAPT exchange energy with Sinfinity.
    """

    ###  Matrices and tensors from helper_dimer
    s_ab = dimer.s("ab")
    s_ba = dimer.s("ba")
    s_sa = dimer.s("sa")
    s_rb = dimer.s("rb")

    ### Exch100_Sinfinity
    exch1_sinf = -2 * oe.contract(
        "Bb,aB,sa,bs", dimer.A_bb, s_ab, s_sa, dimer.omegaA_bs
    )
    exch1_sinf -= 2 * oe.contract(
        "Aa,bA,rb,ar", dimer.B_aa, s_ba, s_rb, dimer.omegaB_ar
    )
    exch1_sinf -= 2 * oe.contract(
        "Bb,Aa,sA,rB,Qar,Qbs", dimer.A_bb, dimer.B_aa, s_sa, s_rb, dimer.Qar, dimer.Qbs
    )
    exch1_sinf += 4 * oe.contract(
        "BD,AC,Da,Cb,sA,rB,Qar,Qbs",
        dimer.A_bb,
        dimer.B_aa,
        s_ba,
        s_ab,
        s_sa,
        s_rb,
        dimer.Qar,
        dimer.Qbs,
    )

    return exch1_sinf
