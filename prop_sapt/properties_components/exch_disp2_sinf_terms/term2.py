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
            "ar,cb,sd,bs,rdac", mol.E_ar, mol.A_bb, mol.D_ss, mol.omegaA_bs, u_rsab
        )
        - 4
        * oe.contract(
            "ca,rd,bs,ar,dscb", mol.B_aa, mol.C_rr, mol.F_bs, mol.omegaB_ar, u_rsab
        )
        - 2
        * oe.contract(
            "bc,da,rs,ar,csdb", mol.I_br, mol.B_aa, mol.G_rs, mol.omegaB_ar, u_rsab
        )
        - 2
        * oe.contract(
            "cb,sr,ad,bs,rdac", mol.A_bb, mol.G_sr, mol.J_as, mol.omegaA_bs, u_rsab
        )
        + 2
        * oe.contract(
            "cr,sd,ab,bs,rdac", mol.I_br, mol.D_ss, mol.H_ab, mol.omegaA_bs, u_rsab
        )
        + 2
        * oe.contract(
            "rc,ba,ds,ar,csdb", mol.C_rr, mol.H_ba, mol.J_as, mol.omegaB_ar, u_rsab
        )
        + 4
        * oe.contract(
            "ar,cs,bd,dc,rsab", mol.E_ar, mol.G_rs, mol.H_ba, mol.omegaB_ar, u_rsab
        )
        + 4
        * oe.contract(
            "bs,cr,ad,dc,rsab", mol.F_bs, mol.G_sr, mol.H_ab, mol.omegaA_bs, u_rsab
        )
        - 8
        * oe.contract(
            "ar,sb,cd,ef,rdae,Qfc,Qbs",
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
            "ra,bs,cd,ef,dseb,Qar,Qfc",
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
            "ar,cd,es,bf,rsab,Qfc,Qde",
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
            "ra,cd,es,fb,dsfc,Qar,Qbe",
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
            "cr,bs,ad,ef,rsab,Qfc,Qde",
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
            "cr,sb,da,ef,rfed,Qac,Qbs",
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
            "cd,rs,ab,ef,dsac,Qfr,Qbe",
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
            "cd,sr,ba,ef,rfeb,Qac,Qds",
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
            "bc,ad,sr,ef,rfab,Qde,Qcs",
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
            "cr,ds,ba,ef,rseb,Qac,Qfd",
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
            "br,cd,ae,fs,rsab,Qec,Qdf",
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
            "bc,dr,as,ef,rsab,Qfd,Qce",
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
            "ar,bc,ds,ef,rsab,Qfd,Qce",
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
            "ra,bc,sd,ef,dfeb,Qar,Qcs",
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
            "cd,ae,sb,rf,dfac,Qer,Qbs",
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
            "cd,ae,bs,fr,rsab,Qec,Qdf",
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
            "bc,ad,er,fs,rsab,Qde,Qcf",
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
            "sr,cd,ba,ef,rdeb,Qac,Qfs",
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
            "ar,cd,be,fs,rsab,Qdc,Qef",
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
            "ac,dr,bs,ef,rsab,Qcd,Qfe",
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
