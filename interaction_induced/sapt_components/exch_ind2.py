import opt_einsum as oe

from ..molecule import Dimer


def calc_exch_ind2_s2_a_energy(dimer: Dimer, coupled: bool = True):
    """Calculate the second-order SAPT exchange-induction energy for the A monomer
    utilising the S^2 approximation.

    Args:
        dimer (Dimer): Dimer object containing the necessary data for the calculation.
        coupled (bool, optional): Whether to calculate response induction. Defaults to True.

    Returns:
        _type_: The second-order SAPT exchange-induction energy for the A monomer
        using the S^2 approximation.
    """

    # Prepere the necessary overlap matrices
    s_ab = dimer.s("ab")
    s_ba = dimer.s("ba")
    s_sa = dimer.s("sa")
    s_rb = dimer.s("rb")
    s_sr = dimer.s("sr")
    s_br = dimer.s("br")

    if coupled:
        t_ra = dimer.get_cpscf_ra()

    else:
        t_ra = dimer.tB_ra

    exch_ind2_s2_a = -2 * oe.contract("sr,ab,ra,bs", s_sr, s_ab, t_ra, dimer.omegaA_bs)
    exch_ind2_s2_a -= 2 * oe.contract("bR,rb,Ra,ar", s_br, s_rb, t_ra, dimer.omegaB_ar)
    exch_ind2_s2_a -= 4 * oe.contract(
        "sA,Ab,ra,Qar,Qbs",
        s_sa,
        s_ab,
        t_ra,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_ind2_s2_a -= 2 * oe.contract(
        "sR,rb,Ra,Qar,Qbs",
        s_sr,
        s_rb,
        t_ra,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_ind2_s2_a += 2 * oe.contract(
        "sa,Ab,rA,Qar,Qbs",
        s_sa,
        s_ab,
        t_ra,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_ind2_s2_a += 2 * oe.contract("ba,Ab,rA,ar", s_ba, s_ab, t_ra, dimer.omegaB_ar)

    return exch_ind2_s2_a


def calc_exch_ind2_s2_b_energy(dimer: Dimer, coupled: bool = True):
    """Calculate the second-order SAPT exchange-induction energy for the B monomer
    utilising the S^2 approximation.

    Args:
        dimer (Dimer): Dimer object containing the necessary data for the calculation.
        coupled (bool, optional): Whether to calculate response induction. Defaults to True.

    Returns:
        _type_: The second-order SAPT exchange-induction energy for the B monomer
        using the S^2 approximation.
    """

    # Prepere the necessary overlap matrices
    s_ab = dimer.s("ab")
    s_ba = dimer.s("ba")
    s_sa = dimer.s("sa")
    s_as = dimer.s("as")
    s_rb = dimer.s("rb")
    s_rs = dimer.s("rs")

    if coupled:
        t_sb = dimer.get_cpscf_sb()
    else:
        t_sb = dimer.tA_sb

    ### ExchInd200_S2_B
    exch_ind2_s2_b = -2 * oe.contract("rs,ba,sb,ar", s_rs, s_ba, t_sb, dimer.omegaB_ar)
    exch_ind2_s2_b -= 2 * oe.contract("aS,sa,Sb,bs", s_as, s_sa, t_sb, dimer.omegaA_bs)
    exch_ind2_s2_b -= 4 * oe.contract(
        "rB,Ba,sb,Qar,Qbs",
        s_rb,
        s_ba,
        t_sb,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_ind2_s2_b -= 2 * oe.contract(
        "rS,sa,Sb,Qar,Qbs",
        s_rs,
        s_sa,
        t_sb,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_ind2_s2_b += 2 * oe.contract(
        "rb,Ba,sB,Qar,Qbs",
        s_rb,
        s_ba,
        t_sb,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_ind2_s2_b += 2 * oe.contract("ab,Ba,sB,bs", s_ab, s_ba, t_sb, dimer.omegaA_bs)

    return exch_ind2_s2_b


def calc_exch_ind2_a_energy(dimer: Dimer, coupled: bool = True):
    """Calculate the second-orde SAPT exchange-induction energy for the A monomer.

    Args:
        dimer (Dimer): The dimer object containing the necessary data for the calculation.
        coupled (bool, optional): Whether to calculate the response induciton. Defaults to True.

    Returns:
        _type_: The second-order SAPT exchange-induction energy for the A monomer.
    """

    if coupled:
        t_ra = dimer.get_cpscf_ra()

    else:
        t_ra = dimer.tB_ra

    ### ExchInd200_A_Sinfinity
    AStB_rb = oe.contract("ab,ra->rb", dimer.H_ab, t_ra)
    BtB_ra = oe.contract("Aa,rA->ra", dimer.B_aa, t_ra)

    term1_br = (
        oe.contract("sr,bs->br", dimer.G_sr, dimer.omegaA_bs)
        - oe.contract(
            "sa,rR,Qar,Qbs->bR",
            dimer.J_sa,
            dimer.C_rr,
            dimer.Qar,
            dimer.Qbs,
        )
        - 2
        * oe.contract(
            "ra,sR,Qar,Qbs->bR",
            dimer.E_ra,
            dimer.G_sr,
            dimer.Qar,
            dimer.Qbs,
        )
    )
    term2_ar = (
        oe.contract("rR,ar->aR", dimer.C_rr, dimer.omegaB_ar)
        - oe.contract(
            "rb,sR,Qar,Qbs->aR",
            dimer.I_rb,
            dimer.G_sr,
            dimer.Qar,
            dimer.Qbs,
        )
        - 2
        * oe.contract(
            "sb,rR,Qar,Qbs->aR",
            dimer.F_sb,
            dimer.C_rr,
            dimer.Qar,
            dimer.Qbs,
        )
    )

    exch_ind2_a_dimer = -2 * (
        oe.contract("rb,br", AStB_rb, term1_br) - oe.contract("ra,ar", BtB_ra, term2_ar)
    )

    # removing Ind200(A)
    exch_ind2_a_dimer -= 2 * oe.contract("ra,ar", t_ra, dimer.omegaB_ar)

    return exch_ind2_a_dimer


def calc_exch_ind2_b_energy(dimer: Dimer, coupled: bool = True):
    """Calculate the second-order SAPT exchange-induction energy for the B monomer.

    Args:
        dimer (Dimer): The dimer object containing the necessary data for the calculation.
        coupled (bool, optional): Whether to calculate the response induction. Defaults to True.

    Returns:
        _type_: The second-order SAPT exchange-induction energy for the B monomer.
    """

    if coupled:
        t_sb = dimer.get_cpscf_sb()

    else:
        t_sb = dimer.tA_sb

    ### ExchInd200_B_Sinfinity
    BStA_sa = oe.contract("ba,sb->sa", dimer.H_ba, t_sb)
    AtA_sb = oe.contract("Bb,sB->sb", dimer.A_bb, t_sb)

    term1_as = (
        oe.contract("rs,ar->as", dimer.G_rs, dimer.omegaB_ar)
        - oe.contract(
            "rb,sS,Qar,Qbs->aS",
            dimer.I_rb,
            dimer.D_ss,
            dimer.Qar,
            dimer.Qbs,
        )
        - 2
        * oe.contract(
            "sb,rS,Qar,Qbs->aS",
            dimer.F_sb,
            dimer.G_rs,
            dimer.Qar,
            dimer.Qbs,
        )
    )
    term2_bs = (
        oe.contract("sS,bs->bS", dimer.D_ss, dimer.omegaA_bs)
        - oe.contract(
            "sa,rS,Qar,Qbs->bS",
            dimer.J_sa,
            dimer.G_rs,
            dimer.Qar,
            dimer.Qbs,
        )
        - 2
        * oe.contract(
            "ra,sS,Qar,Qbs->bS",
            dimer.E_ra,
            dimer.D_ss,
            dimer.Qar,
            dimer.Qbs,
        )
    )

    exch_ind2_b_sinf = -2 * (
        oe.contract("sa,as", BStA_sa, term1_as) - oe.contract("sb,bs", AtA_sb, term2_bs)
    )

    # removing Ind200(B)
    exch_ind2_b_sinf -= 2 * oe.contract("sb,bs", t_sb, dimer.omegaA_bs)

    return exch_ind2_b_sinf
