import opt_einsum as oe

from ..molecule import Dimer


def calc_exch_disp2_s2_energy(mol: Dimer):
    """Calculate the second-order SAPT exchange-dispersion energy using the S^2 approximation.

    Args:
        mol (Dimer): A Dimer object containing the molecular information.

    Returns:
        _type_: The second-order SAPT exchange-dispersion energy using the S^2 approximation.
    """

    # Prepere the necessary overlap matrices
    s_ab = mol.s("ab")
    s_ba = mol.s("ba")
    s_sa = mol.s("sa")
    s_as = mol.s("as")
    s_rb = mol.s("rb")
    s_sr = mol.s("sr")
    s_rs = mol.s("rs")
    s_br = mol.s("br")

    vA_bs = mol.potential("bs", "A")
    vB_ar = mol.potential("ar", "B")

    exch_disp20_s2 = -8 * oe.contract(
        "br,ab,rsaB,QAA,QBs",
        s_br,
        s_ab,
        mol.t_rsab,
        mol.Qaa,
        mol.Qbs,
    )
    exch_disp20_s2 += -8 * oe.contract(
        "as,ba,rsAb,QAr,QBB",
        s_as,
        s_ba,
        mol.t_rsab,
        mol.Qar,
        mol.Qbb,
    )
    exch_disp20_s2 += -4 * oe.contract(
        "Sr,as,rsab,QAA,QbS",
        s_sr,
        s_as,
        mol.t_rsab,
        mol.Qaa,
        mol.Qbs,
    )
    exch_disp20_s2 += -4 * oe.contract(
        "br,Rs,rsab,QaR,QBB",
        s_br,
        s_rs,
        mol.t_rsab,
        mol.Qar,
        mol.Qbb,
    )
    exch_disp20_s2 += -4 * oe.contract("br,ab,rsaB,Bs", s_br, s_ab, mol.t_rsab, vA_bs)
    exch_disp20_s2 += -4 * oe.contract(
        "bR,rb,RsaB,Qar,QBs",
        s_br,
        s_rb,
        mol.t_rsab,
        mol.Qar,
        mol.Qbs,
    )
    exch_disp20_s2 += -4 * oe.contract("as,ba,rsAb,Ar", s_as, s_ba, mol.t_rsab, vB_ar)
    exch_disp20_s2 += -4 * oe.contract(
        "aS,sa,rSAb,QAr,Qbs",
        s_as,
        s_sa,
        mol.t_rsab,
        mol.Qar,
        mol.Qbs,
    )
    exch_disp20_s2 += -2 * oe.contract(
        "Sr,Rs,rsab,QaR,QbS",
        s_sr,
        s_rs,
        mol.t_rsab,
        mol.Qar,
        mol.Qbs,
    )
    exch_disp20_s2 += -2 * oe.contract("Sr,as,rsab,bS", s_sr, s_as, mol.t_rsab, vA_bs)
    exch_disp20_s2 += -2 * oe.contract("br,Rs,rsab,aR", s_br, s_rs, mol.t_rsab, vB_ar)
    exch_disp20_s2 += -2 * oe.contract(
        "bA,aB,rsab,QAr,QBs",
        s_ba,
        s_ab,
        mol.t_rsab,
        mol.Qar,
        mol.Qbs,
    )
    exch_disp20_s2 += 2 * oe.contract("Br,ab,rsaB,bs", s_br, s_ab, mol.t_rsab, vA_bs)
    exch_disp20_s2 += 2 * oe.contract(
        "BR,rb,RsaB,Qar,Qbs",
        s_br,
        s_rb,
        mol.t_rsab,
        mol.Qar,
        mol.Qbs,
    )
    exch_disp20_s2 += 2 * oe.contract("As,ba,rsAb,ar", s_as, s_ba, mol.t_rsab, vB_ar)
    exch_disp20_s2 += 2 * oe.contract(
        "AS,sa,rSAb,Qar,Qbs",
        s_as,
        s_sa,
        mol.t_rsab,
        mol.Qar,
        mol.Qbs,
    )
    exch_disp20_s2 += 4 * oe.contract(
        "Br,ab,rsaB,QAA,Qbs",
        s_br,
        s_ab,
        mol.t_rsab,
        mol.Qaa,
        mol.Qbs,
    )
    exch_disp20_s2 += 4 * oe.contract(
        "As,ba,rsAb,Qar,QBB",
        s_as,
        s_ba,
        mol.t_rsab,
        mol.Qar,
        mol.Qbb,
    )
    exch_disp20_s2 += 4 * oe.contract(
        "ba,aB,rsAb,QAr,QBs",
        s_ba,
        s_ab,
        mol.t_rsab,
        mol.Qar,
        mol.Qbs,
    )
    exch_disp20_s2 += 4 * oe.contract(
        "bA,ab,rsaB,QAr,QBs",
        s_ba,
        s_ab,
        mol.t_rsab,
        mol.Qar,
        mol.Qbs,
    )

    return exch_disp20_s2


def calc_exch_disp2_energy(mol: Dimer):
    """Calculate the second-order SAPT exchange-dispersion energy in Sinfinity.
    IMPORTANT: This function calculates the exchange-dispersion energy in the uncoupled approximation.

    Args:
        mol (Dimer): A Dimer object containing the molecular information.

    Returns:
        _type_: The second-order SAPT exchange-dispersion energy in Sinfinity.
    """

    ### ExchDisp200_Sinfinity

    term_rsab = -4 * oe.contract(
        "cd,eb,sf,bs->dfce", mol.E_ar, mol.A_bb, mol.D_ss, mol.omegaA_bs
    )
    term_rsab -= 4 * oe.contract(
        "ca,rd,ef,ar->dfce", mol.B_aa, mol.C_rr, mol.F_bs, mol.omegaB_ar
    )
    term_rsab -= 2 * oe.contract(
        "cd,ea,rf,ar->dfec", mol.I_br, mol.B_aa, mol.G_rs, mol.omegaB_ar
    )
    term_rsab -= 2 * oe.contract(
        "cb,sd,ef,bs->dfec", mol.A_bb, mol.G_sr, mol.J_as, mol.omegaA_bs
    )
    term_rsab += 2 * oe.contract(
        "cd,se,fb,bs->defc", mol.I_br, mol.D_ss, mol.H_ab, mol.omegaA_bs
    )
    term_rsab += 2 * oe.contract(
        "rc,da,ef,ar->cfed", mol.C_rr, mol.H_ba, mol.J_as, mol.omegaB_ar
    )
    term_rsab += 4 * oe.contract(
        "cd,re,fa,ar->decf", mol.E_ar, mol.G_rs, mol.H_ba, mol.omegaB_ar
    )
    term_rsab += 4 * oe.contract(
        "cd,se,fb,bs->edfc", mol.F_bs, mol.G_sr, mol.H_ab, mol.omegaA_bs
    )
    term_rsab -= 8 * oe.contract(
        "cd,sb,re,fa,Qar,Qbs->decf",
        mol.E_ar,
        mol.F_sb,
        mol.G_rs,
        mol.H_ba,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab -= 8 * oe.contract(
        "ra,cd,se,fb,Qar,Qbs->edfc",
        mol.E_ra,
        mol.F_bs,
        mol.G_sr,
        mol.H_ab,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab -= 4 * oe.contract(
        "cd,rb,se,fa,Qar,Qbs->decf",
        mol.E_ar,
        mol.I_rb,
        mol.D_ss,
        mol.H_ba,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab -= 4 * oe.contract(
        "ra,cd,se,fb,Qar,Qbs->defc",
        mol.E_ra,
        mol.I_br,
        mol.D_ss,
        mol.H_ab,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab -= 4 * oe.contract(
        "rc,de,fb,sa,Qar,Qbs->cefd",
        mol.C_rr,
        mol.F_bs,
        mol.H_ab,
        mol.J_sa,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab -= 4 * oe.contract(
        "rc,sb,da,ef,Qar,Qbs->cfed",
        mol.C_rr,
        mol.F_sb,
        mol.H_ba,
        mol.J_as,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab -= 2 * oe.contract(
        "cd,re,fb,sa,Qar,Qbs->defc",
        mol.I_br,
        mol.G_rs,
        mol.H_ab,
        mol.J_sa,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab -= 2 * oe.contract(
        "rb,sc,da,ef,Qar,Qbs->cfed",
        mol.I_rb,
        mol.G_sr,
        mol.H_ba,
        mol.J_as,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab -= 2 * oe.contract(
        "cb,da,se,rf,Qar,Qbs->efdc",
        mol.A_bb,
        mol.B_aa,
        mol.G_sr,
        mol.G_rs,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab -= 2 * oe.contract(
        "rc,sd,ea,fb,Qar,Qbs->cdfe",
        mol.C_rr,
        mol.D_ss,
        mol.H_ba,
        mol.H_ab,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab += 2 * oe.contract(
        "cd,rb,ea,sf,Qar,Qbs->dfec",
        mol.I_br,
        mol.I_rb,
        mol.B_aa,
        mol.D_ss,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab += 2 * oe.contract(
        "cb,rd,ef,sa,Qar,Qbs->dfec",
        mol.A_bb,
        mol.C_rr,
        mol.J_as,
        mol.J_sa,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab += 4 * oe.contract(
        "cd,eb,rf,sa,Qar,Qbs->dfce",
        mol.E_ar,
        mol.A_bb,
        mol.G_rs,
        mol.J_sa,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab += 4 * oe.contract(
        "ra,cb,sd,ef,Qar,Qbs->dfec",
        mol.E_ra,
        mol.A_bb,
        mol.G_sr,
        mol.J_as,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab += 4 * oe.contract(
        "cd,ea,sb,rf,Qar,Qbs->dfec",
        mol.I_br,
        mol.B_aa,
        mol.F_sb,
        mol.G_rs,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab += 4 * oe.contract(
        "rb,ca,de,sf,Qar,Qbs->fecd",
        mol.I_rb,
        mol.B_aa,
        mol.F_bs,
        mol.G_sr,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab += 4 * oe.contract(
        "cb,da,re,sf,Qar,Qbs->efdc",
        mol.A_bb,
        mol.B_aa,
        mol.C_rr,
        mol.D_ss,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab += 4 * oe.contract(
        "sc,rd,ea,fb,Qar,Qbs->cdfe",
        mol.G_sr,
        mol.G_rs,
        mol.H_ba,
        mol.H_ab,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab += 8 * oe.contract(
        "cd,ra,eb,sf,Qar,Qbs->dfce",
        mol.E_ar,
        mol.E_ra,
        mol.A_bb,
        mol.D_ss,
        mol.Qar,
        mol.Qbs,
    )
    term_rsab += 8 * oe.contract(
        "ca,rd,ef,sb,Qar,Qbs->dfce",
        mol.B_aa,
        mol.C_rr,
        mol.F_bs,
        mol.F_sb,
        mol.Qar,
        mol.Qbs,
    )

    disp2_all = oe.contract("rsab,rsab", mol.t_rsab, term_rsab)
    disp2 = 4 * oe.contract("rsab,Qar,Qbs", mol.t_rsab, mol.Qar, mol.Qbs)
    exch_disp2_sinf = disp2_all - disp2

    return exch_disp2_sinf
