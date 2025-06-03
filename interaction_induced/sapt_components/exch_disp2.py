import opt_einsum as oe

from ..molecule import Dimer


def calc_exch_disp2_s2_energy(dimer: Dimer):
    """Calculate the second-order SAPT exchange-dispersion energy using the S^2 approximation.

    Args:
        dimer (Dimer): A Dimer object containing the molecular information.

    Returns:
        _type_: The second-order SAPT exchange-dispersion energy using the S^2 approximation.
    """

    # Prepere the necessary overlap matrices
    s_ab = dimer.s("ab")
    s_ba = dimer.s("ba")
    s_sa = dimer.s("sa")
    s_as = dimer.s("as")
    s_rb = dimer.s("rb")
    s_sr = dimer.s("sr")
    s_rs = dimer.s("rs")
    s_br = dimer.s("br")

    vA_bs = dimer.potential("bs", "A")
    vB_ar = dimer.potential("ar", "B")

    exch_disp20_s2 = -8 * oe.contract(
        "br,ab,rsaB,QAA,QBs",
        s_br,
        s_ab,
        dimer.t_rsab,
        dimer.Qaa,
        dimer.Qbs,
    )
    exch_disp20_s2 += -8 * oe.contract(
        "as,ba,rsAb,QAr,QBB",
        s_as,
        s_ba,
        dimer.t_rsab,
        dimer.Qar,
        dimer.Qbb,
    )
    exch_disp20_s2 += -4 * oe.contract(
        "Sr,as,rsab,QAA,QbS",
        s_sr,
        s_as,
        dimer.t_rsab,
        dimer.Qaa,
        dimer.Qbs,
    )
    exch_disp20_s2 += -4 * oe.contract(
        "br,Rs,rsab,QaR,QBB",
        s_br,
        s_rs,
        dimer.t_rsab,
        dimer.Qar,
        dimer.Qbb,
    )
    exch_disp20_s2 += -4 * oe.contract("br,ab,rsaB,Bs", s_br, s_ab, dimer.t_rsab, vA_bs)
    exch_disp20_s2 += -4 * oe.contract(
        "bR,rb,RsaB,Qar,QBs",
        s_br,
        s_rb,
        dimer.t_rsab,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_disp20_s2 += -4 * oe.contract("as,ba,rsAb,Ar", s_as, s_ba, dimer.t_rsab, vB_ar)
    exch_disp20_s2 += -4 * oe.contract(
        "aS,sa,rSAb,QAr,Qbs",
        s_as,
        s_sa,
        dimer.t_rsab,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_disp20_s2 += -2 * oe.contract(
        "Sr,Rs,rsab,QaR,QbS",
        s_sr,
        s_rs,
        dimer.t_rsab,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_disp20_s2 += -2 * oe.contract("Sr,as,rsab,bS", s_sr, s_as, dimer.t_rsab, vA_bs)
    exch_disp20_s2 += -2 * oe.contract("br,Rs,rsab,aR", s_br, s_rs, dimer.t_rsab, vB_ar)
    exch_disp20_s2 += -2 * oe.contract(
        "bA,aB,rsab,QAr,QBs",
        s_ba,
        s_ab,
        dimer.t_rsab,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_disp20_s2 += 2 * oe.contract("Br,ab,rsaB,bs", s_br, s_ab, dimer.t_rsab, vA_bs)
    exch_disp20_s2 += 2 * oe.contract(
        "BR,rb,RsaB,Qar,Qbs",
        s_br,
        s_rb,
        dimer.t_rsab,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_disp20_s2 += 2 * oe.contract("As,ba,rsAb,ar", s_as, s_ba, dimer.t_rsab, vB_ar)
    exch_disp20_s2 += 2 * oe.contract(
        "AS,sa,rSAb,Qar,Qbs",
        s_as,
        s_sa,
        dimer.t_rsab,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_disp20_s2 += 4 * oe.contract(
        "Br,ab,rsaB,QAA,Qbs",
        s_br,
        s_ab,
        dimer.t_rsab,
        dimer.Qaa,
        dimer.Qbs,
    )
    exch_disp20_s2 += 4 * oe.contract(
        "As,ba,rsAb,Qar,QBB",
        s_as,
        s_ba,
        dimer.t_rsab,
        dimer.Qar,
        dimer.Qbb,
    )
    exch_disp20_s2 += 4 * oe.contract(
        "ba,aB,rsAb,QAr,QBs",
        s_ba,
        s_ab,
        dimer.t_rsab,
        dimer.Qar,
        dimer.Qbs,
    )
    exch_disp20_s2 += 4 * oe.contract(
        "bA,ab,rsaB,QAr,QBs",
        s_ba,
        s_ab,
        dimer.t_rsab,
        dimer.Qar,
        dimer.Qbs,
    )

    return exch_disp20_s2


def calc_exch_disp2_energy(dimer: Dimer):
    """Calculate the second-order SAPT exchange-dispersion energy in Sinfinity.
    IMPORTANT: This function calculates the exchange-dispersion energy in the uncoupled approximation.

    Args:
        dimer (Dimer): A Dimer object containing the molecular information.

    Returns:
        _type_: The second-order SAPT exchange-dispersion energy in Sinfinity.
    """

    ### ExchDisp200_Sinfinity

    term_abrs = -4 * oe.contract(
        "cd,eb,sf,bs->dfce", dimer.E_ar, dimer.A_bb, dimer.D_ss, dimer.omegaA_bs
    )
    term_abrs -= 4 * oe.contract(
        "ca,rd,ef,ar->dfce", dimer.B_aa, dimer.C_rr, dimer.F_bs, dimer.omegaB_ar
    )
    term_abrs -= 2 * oe.contract(
        "cd,ea,rf,ar->dfec", dimer.I_br, dimer.B_aa, dimer.G_rs, dimer.omegaB_ar
    )
    term_abrs -= 2 * oe.contract(
        "cb,sd,ef,bs->dfec", dimer.A_bb, dimer.G_sr, dimer.J_as, dimer.omegaA_bs
    )
    term_abrs += 2 * oe.contract(
        "cd,se,fb,bs->defc", dimer.I_br, dimer.D_ss, dimer.H_ab, dimer.omegaA_bs
    )
    term_abrs += 2 * oe.contract(
        "rc,da,ef,ar->cfed", dimer.C_rr, dimer.H_ba, dimer.J_as, dimer.omegaB_ar
    )
    term_abrs += 4 * oe.contract(
        "cd,re,fa,ar->decf", dimer.E_ar, dimer.G_rs, dimer.H_ba, dimer.omegaB_ar
    )
    term_abrs += 4 * oe.contract(
        "cd,se,fb,bs->edfc", dimer.F_bs, dimer.G_sr, dimer.H_ab, dimer.omegaA_bs
    )
    term_abrs -= 8 * oe.contract(
        "cd,sb,re,fa,Qar,Qbs->decf",
        dimer.E_ar,
        dimer.F_sb,
        dimer.G_rs,
        dimer.H_ba,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs -= 8 * oe.contract(
        "ra,cd,se,fb,Qar,Qbs->edfc",
        dimer.E_ra,
        dimer.F_bs,
        dimer.G_sr,
        dimer.H_ab,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs -= 4 * oe.contract(
        "cd,rb,se,fa,Qar,Qbs->decf",
        dimer.E_ar,
        dimer.I_rb,
        dimer.D_ss,
        dimer.H_ba,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs -= 4 * oe.contract(
        "ra,cd,se,fb,Qar,Qbs->defc",
        dimer.E_ra,
        dimer.I_br,
        dimer.D_ss,
        dimer.H_ab,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs -= 4 * oe.contract(
        "rc,de,fb,sa,Qar,Qbs->cefd",
        dimer.C_rr,
        dimer.F_bs,
        dimer.H_ab,
        dimer.J_sa,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs -= 4 * oe.contract(
        "rc,sb,da,ef,Qar,Qbs->cfed",
        dimer.C_rr,
        dimer.F_sb,
        dimer.H_ba,
        dimer.J_as,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs -= 2 * oe.contract(
        "cd,re,fb,sa,Qar,Qbs->defc",
        dimer.I_br,
        dimer.G_rs,
        dimer.H_ab,
        dimer.J_sa,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs -= 2 * oe.contract(
        "rb,sc,da,ef,Qar,Qbs->cfed",
        dimer.I_rb,
        dimer.G_sr,
        dimer.H_ba,
        dimer.J_as,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs -= 2 * oe.contract(
        "cb,da,se,rf,Qar,Qbs->efdc",
        dimer.A_bb,
        dimer.B_aa,
        dimer.G_sr,
        dimer.G_rs,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs -= 2 * oe.contract(
        "rc,sd,ea,fb,Qar,Qbs->cdfe",
        dimer.C_rr,
        dimer.D_ss,
        dimer.H_ba,
        dimer.H_ab,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs += 2 * oe.contract(
        "cd,rb,ea,sf,Qar,Qbs->dfec",
        dimer.I_br,
        dimer.I_rb,
        dimer.B_aa,
        dimer.D_ss,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs += 2 * oe.contract(
        "cb,rd,ef,sa,Qar,Qbs->dfec",
        dimer.A_bb,
        dimer.C_rr,
        dimer.J_as,
        dimer.J_sa,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs += 4 * oe.contract(
        "cd,eb,rf,sa,Qar,Qbs->dfce",
        dimer.E_ar,
        dimer.A_bb,
        dimer.G_rs,
        dimer.J_sa,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs += 4 * oe.contract(
        "ra,cb,sd,ef,Qar,Qbs->dfec",
        dimer.E_ra,
        dimer.A_bb,
        dimer.G_sr,
        dimer.J_as,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs += 4 * oe.contract(
        "cd,ea,sb,rf,Qar,Qbs->dfec",
        dimer.I_br,
        dimer.B_aa,
        dimer.F_sb,
        dimer.G_rs,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs += 4 * oe.contract(
        "rb,ca,de,sf,Qar,Qbs->fecd",
        dimer.I_rb,
        dimer.B_aa,
        dimer.F_bs,
        dimer.G_sr,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs += 4 * oe.contract(
        "cb,da,re,sf,Qar,Qbs->efdc",
        dimer.A_bb,
        dimer.B_aa,
        dimer.C_rr,
        dimer.D_ss,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs += 4 * oe.contract(
        "sc,rd,ea,fb,Qar,Qbs->cdfe",
        dimer.G_sr,
        dimer.G_rs,
        dimer.H_ba,
        dimer.H_ab,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs += 8 * oe.contract(
        "cd,ra,eb,sf,Qar,Qbs->dfce",
        dimer.E_ar,
        dimer.E_ra,
        dimer.A_bb,
        dimer.D_ss,
        dimer.Qar,
        dimer.Qbs,
    )
    term_abrs += 8 * oe.contract(
        "ca,rd,ef,sb,Qar,Qbs->dfce",
        dimer.B_aa,
        dimer.C_rr,
        dimer.F_bs,
        dimer.F_sb,
        dimer.Qar,
        dimer.Qbs,
    )

    disp2_all = oe.contract("rsab,abrs", dimer.t_rsab, term_abrs)
    disp2 = 4 * oe.contract("rsab,Qar,Qbs", dimer.t_rsab, dimer.Qar, dimer.Qbs)
    exch_disp2_sinf = disp2_all - disp2

    return exch_disp2_sinf
