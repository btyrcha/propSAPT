import numpy as np
import opt_einsum as oe

from prop_sapt.molecule import Dimer


def get_u_rsab_amplitudes(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_ar: np.ndarray,
    prop_A_rr: np.ndarray,
    prop_B_bb: np.ndarray,
    prop_B_bs: np.ndarray,
    prop_B_ss: np.ndarray,
) -> np.ndarray:

    u_rsab = np.zeros(
        (mol.nvirt_A, mol.nvirt_B, mol.ndocc_A, mol.ndocc_B)
    )  # TODO: Implement the calculation for u_rsab

    return u_rsab


def calc_exch_disp2_s2_property(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_ar: np.ndarray,
    prop_A_rr: np.ndarray,
    prop_B_bb: np.ndarray,
    prop_B_bs: np.ndarray,
    prop_B_ss: np.ndarray,
) -> np.ndarray:

    s_ab = mol.s("ab")
    s_ba = s_ab.T
    s_as = mol.s("as")
    s_sa = s_as.T
    s_rs = mol.s("rs")
    s_sr = s_rs.T
    s_rb = mol.s("rb")
    s_br = s_rb.T

    u_rsab = get_u_rsab_amplitudes(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        prop_A_aa=prop_A_aa,
        prop_A_ar=prop_A_ar,
        prop_A_rr=prop_A_rr,
        prop_B_bb=prop_B_bb,
        prop_B_bs=prop_B_bs,
        prop_B_ss=prop_B_ss,
    )

    x2_exch_disp_s2 = np.array(
        [
            # < R(X) | V P2 R(V) >
            0.0
            # < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
            - 4 * oe.contract("br,ab,rsac,cs", s_br, s_ab, u_rsab, mol.omegaA_bs)
            + 2 * oe.contract("br,ac,rsab,cs", s_br, s_ab, u_rsab, mol.omegaA_bs)
            - 2 * oe.contract("sr,ac,rcab,bs", s_sr, s_as, u_rsab, mol.omegaA_bs)
            - 4 * oe.contract("as,ba,rscb,cr", s_as, s_ba, u_rsab, mol.omegaB_ar)
            + 2 * oe.contract("as,bc,rsab,cr", s_as, s_ba, u_rsab, mol.omegaB_ar)
            - 2 * oe.contract("br,cs,rsab,ac", s_br, s_rs, u_rsab, mol.omegaB_ar)
            - 2
            * oe.contract("ba,cd,rscb,Qar,Qds", s_ba, s_ab, u_rsab, mol.Qar, mol.Qbs)
            + 4
            * oe.contract("ba,cb,rsce,Qar,Qes", s_ba, s_ab, u_rsab, mol.Qar, mol.Qbs)
            + 4
            * oe.contract("ba,ac,rsdb,Qdr,Qcs", s_ba, s_ab, u_rsab, mol.Qar, mol.Qbs)
            - 4
            * oe.contract("br,cb,rsad,Qac,Qds", s_br, s_rb, u_rsab, mol.Qar, mol.Qbs)
            + 2
            * oe.contract("br,cd,rsab,Qac,Qds", s_br, s_rb, u_rsab, mol.Qar, mol.Qbs)
            - 2
            * oe.contract("sr,cd,rdab,Qac,Qbs", s_sr, s_rs, u_rsab, mol.Qar, mol.Qbs)
            - 4
            * oe.contract("as,ca,rsdb,Qdr,Qbc", s_as, s_sa, u_rsab, mol.Qar, mol.Qbs)
            + 2
            * oe.contract("as,cd,rsab,Qdr,Qbc", s_as, s_sa, u_rsab, mol.Qar, mol.Qbs)
        ]
    )

    return x2_exch_disp_s2
