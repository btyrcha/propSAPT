import opt_einsum as oe

from ..molecule import Dimer


def calc_exch_ind2_s2_a_energy(mol: Dimer, coupled: bool = True):
    """Calculate the second-order SAPT exchange-induction energy for the A monomer
    utilising the S^2 approximation.

    Args:
        mol (Dimer): Dimer object containing the necessary data for the calculation.
        coupled (bool, optional): Whether to calculate response induction. Defaults to True.

    Returns:
        _type_: The second-order SAPT exchange-induction energy for the A monomer
        using the S^2 approximation.
    """

    # Prepere the necessary overlap matrices
    s_ab = mol.s("ab")
    s_ba = mol.s("ba")
    s_sa = mol.s("sa")
    s_rb = mol.s("rb")
    s_sr = mol.s("sr")
    s_br = mol.s("br")

    if coupled:
        t_ra = mol.get_cpscf_ra()

    else:
        t_ra = mol.tB_ra

    exch_ind2_s2_a = -2 * oe.contract("sr,ab,ra,bs", s_sr, s_ab, t_ra, mol.omegaA_bs)
    exch_ind2_s2_a -= 2 * oe.contract("bR,rb,Ra,ar", s_br, s_rb, t_ra, mol.omegaB_ar)
    exch_ind2_s2_a -= 4 * oe.contract(
        "sA,Ab,ra,Qar,Qbs",
        s_sa,
        s_ab,
        t_ra,
        mol.Qar,
        mol.Qbs,
    )
    exch_ind2_s2_a -= 2 * oe.contract(
        "sR,rb,Ra,Qar,Qbs",
        s_sr,
        s_rb,
        t_ra,
        mol.Qar,
        mol.Qbs,
    )
    exch_ind2_s2_a += 2 * oe.contract(
        "sa,Ab,rA,Qar,Qbs",
        s_sa,
        s_ab,
        t_ra,
        mol.Qar,
        mol.Qbs,
    )
    exch_ind2_s2_a += 2 * oe.contract("ba,Ab,rA,ar", s_ba, s_ab, t_ra, mol.omegaB_ar)

    return exch_ind2_s2_a


def calc_exch_ind2_s2_b_energy(mol: Dimer, coupled: bool = True):
    """Calculate the second-order SAPT exchange-induction energy for the B monomer
    utilising the S^2 approximation.

    Args:
        mol (Dimer): Dimer object containing the necessary data for the calculation.
        coupled (bool, optional): Whether to calculate response induction. Defaults to True.

    Returns:
        _type_: The second-order SAPT exchange-induction energy for the B monomer
        using the S^2 approximation.
    """

    # Prepere the necessary overlap matrices
    s_ab = mol.s("ab")
    s_ba = mol.s("ba")
    s_sa = mol.s("sa")
    s_as = mol.s("as")
    s_rb = mol.s("rb")
    s_rs = mol.s("rs")

    if coupled:
        t_sb = mol.get_cpscf_sb()
    else:
        t_sb = mol.tA_sb

    ### ExchInd200_S2_B
    exch_ind2_s2_b = -2 * oe.contract("rs,ba,sb,ar", s_rs, s_ba, t_sb, mol.omegaB_ar)
    exch_ind2_s2_b -= 2 * oe.contract("aS,sa,Sb,bs", s_as, s_sa, t_sb, mol.omegaA_bs)
    exch_ind2_s2_b -= 4 * oe.contract(
        "rB,Ba,sb,Qar,Qbs",
        s_rb,
        s_ba,
        t_sb,
        mol.Qar,
        mol.Qbs,
    )
    exch_ind2_s2_b -= 2 * oe.contract(
        "rS,sa,Sb,Qar,Qbs",
        s_rs,
        s_sa,
        t_sb,
        mol.Qar,
        mol.Qbs,
    )
    exch_ind2_s2_b += 2 * oe.contract(
        "rb,Ba,sB,Qar,Qbs",
        s_rb,
        s_ba,
        t_sb,
        mol.Qar,
        mol.Qbs,
    )
    exch_ind2_s2_b += 2 * oe.contract("ab,Ba,sB,bs", s_ab, s_ba, t_sb, mol.omegaA_bs)

    return exch_ind2_s2_b


def calc_exch_ind2_a_energy(mol: Dimer, coupled: bool = True):
    """Calculate the second-orde SAPT exchange-induction energy for the A monomer.

    Args:
        mol (Dimer): The dimer object containing the necessary data for the calculation.
        coupled (bool, optional): Whether to calculate the response induciton. Defaults to True.

    Returns:
        _type_: The second-order SAPT exchange-induction energy for the A monomer.
    """

    if coupled:
        t_ra = mol.get_cpscf_ra()

    else:
        t_ra = mol.tB_ra

    ### ExchInd200_A_Sinfinity
    AStB_rb = oe.contract("ab,ra->rb", mol.H_ab, t_ra)
    BtB_ra = oe.contract("Aa,rA->ra", mol.B_aa, t_ra)

    term1_br = (
        oe.contract("sr,bs->br", mol.G_sr, mol.omegaA_bs)
        - oe.contract(
            "sa,rR,Qar,Qbs->bR",
            mol.J_sa,
            mol.C_rr,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ra,sR,Qar,Qbs->bR",
            mol.E_ra,
            mol.G_sr,
            mol.Qar,
            mol.Qbs,
        )
    )
    term2_ar = (
        oe.contract("rR,ar->aR", mol.C_rr, mol.omegaB_ar)
        - oe.contract(
            "rb,sR,Qar,Qbs->aR",
            mol.I_rb,
            mol.G_sr,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "sb,rR,Qar,Qbs->aR",
            mol.F_sb,
            mol.C_rr,
            mol.Qar,
            mol.Qbs,
        )
    )

    exch_ind2_a_dimer = -2 * (
        oe.contract("rb,br", AStB_rb, term1_br) - oe.contract("ra,ar", BtB_ra, term2_ar)
    )

    # removing Ind200(A)
    exch_ind2_a_dimer -= 2 * oe.contract("ra,ar", t_ra, mol.omegaB_ar)

    return exch_ind2_a_dimer


def calc_exch_ind2_b_energy(mol: Dimer, coupled: bool = True):
    """Calculate the second-order SAPT exchange-induction energy for the B monomer.

    Args:
        mol (Dimer): The dimer object containing the necessary data for the calculation.
        coupled (bool, optional): Whether to calculate the response induction. Defaults to True.

    Returns:
        _type_: The second-order SAPT exchange-induction energy for the B monomer.
    """

    if coupled:
        t_sb = mol.get_cpscf_sb()

    else:
        t_sb = mol.tA_sb

    ### ExchInd200_B_Sinfinity
    BStA_sa = oe.contract("ba,sb->sa", mol.H_ba, t_sb)
    AtA_sb = oe.contract("Bb,sB->sb", mol.A_bb, t_sb)

    term1_as = (
        oe.contract("rs,ar->as", mol.G_rs, mol.omegaB_ar)
        - oe.contract(
            "rb,sS,Qar,Qbs->aS",
            mol.I_rb,
            mol.D_ss,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "sb,rS,Qar,Qbs->aS",
            mol.F_sb,
            mol.G_rs,
            mol.Qar,
            mol.Qbs,
        )
    )
    term2_bs = (
        oe.contract("sS,bs->bS", mol.D_ss, mol.omegaA_bs)
        - oe.contract(
            "sa,rS,Qar,Qbs->bS",
            mol.J_sa,
            mol.G_rs,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ra,sS,Qar,Qbs->bS",
            mol.E_ra,
            mol.D_ss,
            mol.Qar,
            mol.Qbs,
        )
    )

    exch_ind2_b_sinf = -2 * (
        oe.contract("sa,as", BStA_sa, term1_as) - oe.contract("sb,bs", AtA_sb, term2_bs)
    )

    # removing Ind200(B)
    exch_ind2_b_sinf -= 2 * oe.contract("sb,bs", t_sb, mol.omegaA_bs)

    return exch_ind2_b_sinf
