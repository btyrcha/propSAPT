import numpy as np
import opt_einsum as oe

from prop_sapt.molecule import Dimer

from ..exch_disp2 import get_u_rsab_amplitudes


def get_exch_disp2_sinf_property_term2(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_rr: np.ndarray,
    prop_B_bb: np.ndarray,
    prop_B_ss: np.ndarray,
):

    u_rsab = get_u_rsab_amplitudes(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        prop_A_aa=prop_A_aa,
        prop_A_rr=prop_A_rr,
        prop_B_bb=prop_B_bb,
        prop_B_ss=prop_B_ss,
    )

    # < V P R([V, R(X)]) > + < V P R([X, R(V)]) >
    x2_exch_disp_sinf_term2 = (
        -4
        * oe.contract(
            "cd,eb,sf,bs,dfce", mol.E_ar, mol.A_bb, mol.D_ss, mol.omegaA_bs, u_rsab
        )
        - 4
        * oe.contract(
            "ca,rd,ef,ar,dfce", mol.B_aa, mol.C_rr, mol.F_bs, mol.omegaB_ar, u_rsab
        )
        - 2
        * oe.contract(
            "cd,ea,rf,ar,dfec", mol.I_br, mol.B_aa, mol.G_rs, mol.omegaB_ar, u_rsab
        )
        - 2
        * oe.contract(
            "cb,sd,ef,bs,dfec", mol.A_bb, mol.G_sr, mol.J_as, mol.omegaA_bs, u_rsab
        )
        + 2
        * oe.contract(
            "cd,se,fb,bs,defc", mol.I_br, mol.D_ss, mol.H_ab, mol.omegaA_bs, u_rsab
        )
        + 2
        * oe.contract(
            "rc,da,ef,ar,cfed", mol.C_rr, mol.H_ba, mol.J_as, mol.omegaB_ar, u_rsab
        )
        + 4
        * oe.contract(
            "cd,re,fa,ar,decf", mol.E_ar, mol.G_rs, mol.H_ba, mol.omegaB_ar, u_rsab
        )
        + 4
        * oe.contract(
            "cd,se,fb,bs,edfc", mol.F_bs, mol.G_sr, mol.H_ab, mol.omegaA_bs, u_rsab
        )
        - 8
        * oe.contract(
            "cd,sb,re,fa,decf,Qar,Qbs",
            mol.E_ar,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        - 8
        * oe.contract(
            "ra,cd,se,fb,edfc,Qar,Qbs",
            mol.E_ra,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        - 4
        * oe.contract(
            "cd,rb,se,fa,decf,Qar,Qbs",
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        - 4
        * oe.contract(
            "ra,cd,se,fb,defc,Qar,Qbs",
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        - 4
        * oe.contract(
            "rc,de,fb,sa,cefd,Qar,Qbs",
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        - 4
        * oe.contract(
            "rc,sb,da,ef,cfed,Qar,Qbs",
            mol.C_rr,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "cd,re,fb,sa,defc,Qar,Qbs",
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "rb,sc,da,ef,cfed,Qar,Qbs",
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "cb,da,se,rf,efdc,Qar,Qbs",
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "rc,sd,ea,fb,cdfe,Qar,Qbs",
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cd,rb,ea,sf,dfec,Qar,Qbs",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cb,rd,ef,sa,dfec,Qar,Qbs",
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        + 4
        * oe.contract(
            "cd,eb,rf,sa,dfce,Qar,Qbs",
            mol.E_ar,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        + 4
        * oe.contract(
            "ra,cb,sd,ef,dfec,Qar,Qbs",
            mol.E_ra,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        + 4
        * oe.contract(
            "cd,ea,sb,rf,dfec,Qar,Qbs",
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.G_rs,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        + 4
        * oe.contract(
            "rb,ca,de,sf,fecd,Qar,Qbs",
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        + 4
        * oe.contract(
            "cb,da,re,sf,efdc,Qar,Qbs",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        + 4
        * oe.contract(
            "sc,rd,ea,fb,cdfe,Qar,Qbs",
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        + 8
        * oe.contract(
            "cd,ra,eb,sf,dfce,Qar,Qbs",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
        + 8
        * oe.contract(
            "ca,rd,ef,sb,dfce,Qar,Qbs",
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            u_rsab,
            mol.Qar,
            mol.Qbs,
        )
    )

    return x2_exch_disp_sinf_term2
