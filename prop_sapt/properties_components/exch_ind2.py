import numpy as np
import opt_einsum as oe

from prop_sapt.molecule import Dimer


def get_u_ra_amplitudes(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_rr: np.ndarray,
    prop_B_bs: np.ndarray,
) -> np.ndarray:

    u_ra = (
        oe.contract("rR,Ra->ra", prop_A_rr, mol.get_cpscf_ra())
        - oe.contract("Aa,rA->ra", prop_A_aa, mol.get_cpscf_ra())
        + 2 * oe.contract("bs,rsab->ra", prop_B_bs, mol.t_rsab)
        + oe.contract("Ra,rR->ra", xt_A_ra, mol.omegaB_rr)
        - oe.contract("rA,Aa->ra", xt_A_ra, mol.omegaB_aa)
        + 2 * oe.contract("sb,Qar,Qbs->ra", xt_B_sb, mol.Qar, mol.Qbs)
    )
    u_ra = mol.cpscf("A", perturbation=u_ra.T)

    return u_ra


def get_u_sb_amplitudes(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_ar: np.ndarray,
    prop_B_bb: np.ndarray,
    prop_B_ss: np.ndarray,
) -> np.ndarray:

    u_sb = (
        oe.contract("sS,Sb->sb", prop_B_ss, mol.get_cpscf_sb())
        - oe.contract("Bb,sB->sb", prop_B_bb, mol.get_cpscf_sb())
        + 2 * oe.contract("ar,rsab->sb", prop_A_ar, mol.t_rsab)
        + oe.contract("Sb,sS->sb", xt_B_sb, mol.omegaA_ss)
        - oe.contract("sB,Bb->sb", xt_B_sb, mol.omegaA_bb)
        + 2 * oe.contract("ra,Qar,Qbs->sb", xt_A_ra, mol.Qar, mol.Qbs)
    )
    u_sb = mol.cpscf("B", perturbation=u_sb.T)

    return u_sb


def calc_exch_ind2_resp_s2_property(
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

    xt_A_ar = xt_A_ra.T
    xt_B_bs = xt_B_sb.T

    s_ab = mol.s("ab")
    s_ba = s_ab.T
    s_as = mol.s("as")
    s_sa = s_as.T
    s_rs = mol.s("rs")
    s_sr = s_rs.T
    s_rb = mol.s("rb")
    s_br = s_rb.T

    u_ra = get_u_ra_amplitudes(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        prop_A_aa=prop_A_aa,
        prop_A_rr=prop_A_rr,
        prop_B_bs=prop_B_bs,
    )
    u_sb = get_u_sb_amplitudes(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        prop_A_ar=prop_A_ar,
        prop_B_bb=prop_B_bb,
        prop_B_ss=prop_B_ss,
    )

    x2_exch_ind_resp_s2 = np.array(
        [
            # < R(X) | V P2 R(V) >
            +2
            * oe.contract(
                "bs,sr,ac,ra,cb", xt_B_bs, s_sr, s_ab, mol.get_cpscf_ra(), mol.omegaA_bb
            )
            - 2
            * oe.contract(
                "bs,ca,ad,sc,db", xt_B_bs, s_ba, s_ab, mol.get_cpscf_sb(), mol.omegaA_bb
            )
            + 2
            * oe.contract(
                "bs,ac,sa,ce,eb", xt_B_bs, s_as, s_sa, mol.get_cpscf_sb(), mol.omegaA_bb
            )
            + 2
            * oe.contract(
                "ar,sa,cb,rc,bs", xt_A_ar, s_sa, s_ab, mol.get_cpscf_ra(), mol.omegaA_bs
            )
            - 2
            * oe.contract(
                "ar,sc,rb,ca,bs", xt_A_ar, s_sr, s_rb, mol.get_cpscf_ra(), mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "bs,sa,ac,db,cd", xt_B_bs, s_sa, s_ab, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "bs,ca,ab,sd,dc", xt_B_bs, s_sa, s_ab, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "ar,ba,rc,sb,cs", xt_A_ar, s_ba, s_rb, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            - 2
            * oe.contract(
                "ar,rs,ca,sb,bc", xt_A_ar, s_rs, s_sa, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            - 2
            * oe.contract(
                "bs,cr,ab,ra,sc", xt_B_bs, s_sr, s_ab, mol.get_cpscf_ra(), mol.omegaA_ss
            )
            + 2
            * oe.contract(
                "bs,ca,ab,dc,sd", xt_B_bs, s_ba, s_ab, mol.get_cpscf_sb(), mol.omegaA_ss
            )
            - 2
            * oe.contract(
                "bs,ac,da,cb,sd", xt_B_bs, s_as, s_sa, mol.get_cpscf_sb(), mol.omegaA_ss
            )
            - 2
            * oe.contract(
                "ar,bc,db,rd,ca", xt_A_ar, s_ba, s_ab, mol.get_cpscf_ra(), mol.omegaB_aa
            )
            + 2
            * oe.contract(
                "ar,bc,rb,ce,ea", xt_A_ar, s_br, s_rb, mol.get_cpscf_ra(), mol.omegaB_aa
            )
            + 2
            * oe.contract(
                "ar,rs,bc,sb,ca", xt_A_ar, s_rs, s_ba, mol.get_cpscf_sb(), mol.omegaB_aa
            )
            + 2
            * oe.contract(
                "bs,sa,cb,rc,ar", xt_B_bs, s_sa, s_ab, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "ar,ba,cb,rd,dc", xt_A_ar, s_ba, s_rb, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "ar,bc,rb,da,cd", xt_A_ar, s_ba, s_rb, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            - 2
            * oe.contract(
                "bs,sr,cb,ra,ac", xt_B_bs, s_sr, s_rb, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "bs,ca,rb,sc,ar", xt_B_bs, s_ba, s_rb, mol.get_cpscf_sb(), mol.omegaB_ar
            )
            - 2
            * oe.contract(
                "bs,rc,sa,cb,ar", xt_B_bs, s_rs, s_sa, mol.get_cpscf_sb(), mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "ar,ba,cb,dc,rd", xt_A_ar, s_ba, s_ab, mol.get_cpscf_ra(), mol.omegaB_rr
            )
            - 2
            * oe.contract(
                "ar,bc,db,ca,rd", xt_A_ar, s_br, s_rb, mol.get_cpscf_ra(), mol.omegaB_rr
            )
            - 2
            * oe.contract(
                "ar,cs,ba,sb,rc", xt_A_ar, s_rs, s_ba, mol.get_cpscf_sb(), mol.omegaB_rr
            )
            - 2
            * oe.contract(
                "ar,sc,db,rd,Qca,Qbs",
                xt_A_ar,
                s_sa,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qaa,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,sc,cb,re,Qea,Qbs",
                xt_A_ar,
                s_sa,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qaa,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,sc,rb,ce,Qea,Qbs",
                xt_A_ar,
                s_sr,
                s_rb,
                mol.get_cpscf_ra(),
                mol.Qaa,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,bc,rd,sb,Qca,Qds",
                xt_A_ar,
                s_ba,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qaa,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,bc,rb,sd,Qca,Qds",
                xt_A_ar,
                s_ba,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qaa,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,rs,cd,sb,Qda,Qbc",
                xt_A_ar,
                s_rs,
                s_sa,
                mol.get_cpscf_sb(),
                mol.Qaa,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "bs,sa,cd,rc,Qar,Qdb",
                xt_B_bs,
                s_sa,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbb,
            )
            + 4
            * oe.contract(
                "bs,sa,ac,rd,Qdr,Qcb",
                xt_B_bs,
                s_sa,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbb,
            )
            + 2
            * oe.contract(
                "bs,sr,cd,ra,Qac,Qdb",
                xt_B_bs,
                s_sr,
                s_rb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbb,
            )
            - 2
            * oe.contract(
                "bs,ca,rd,sc,Qar,Qdb",
                xt_B_bs,
                s_ba,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbb,
            )
            + 4
            * oe.contract(
                "bs,ca,rc,se,Qar,Qeb",
                xt_B_bs,
                s_ba,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbb,
            )
            + 2
            * oe.contract(
                "bs,rc,sa,ce,Qar,Qeb",
                xt_B_bs,
                s_rs,
                s_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbb,
            )
            - 4
            * oe.contract(
                "ar,sa,rb,cd,Qdc,Qbs",
                xt_A_ar,
                s_sa,
                s_rb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,sa,cb,rd,Qdc,Qbs",
                xt_A_ar,
                s_sa,
                s_rb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,sc,rb,da,Qcd,Qbs",
                xt_A_ar,
                s_sa,
                s_rb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,sa,rb,cd,Qar,Qdc",
                xt_B_bs,
                s_sa,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "bs,sa,rc,db,Qar,Qcd",
                xt_B_bs,
                s_sa,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "bs,ca,rb,sd,Qar,Qdc",
                xt_B_bs,
                s_sa,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "bs,ca,dc,rd,Qar,Qbs",
                xt_B_bs,
                s_ba,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,cr,dc,ra,Qad,Qbs",
                xt_B_bs,
                s_br,
                s_rb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,rc,da,cd,Qar,Qbs",
                xt_B_bs,
                s_rs,
                s_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,ca,ab,rd,Qdr,Qsc",
                xt_B_bs,
                s_sa,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qss,
            )
            + 2
            * oe.contract(
                "bs,ca,db,rd,Qar,Qsc",
                xt_B_bs,
                s_sa,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qss,
            )
            - 2
            * oe.contract(
                "bs,cr,db,ra,Qad,Qsc",
                xt_B_bs,
                s_sr,
                s_rb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qss,
            )
            - 4
            * oe.contract(
                "bs,ca,rc,eb,Qar,Qse",
                xt_B_bs,
                s_ba,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qss,
            )
            + 2
            * oe.contract(
                "bs,ca,rb,dc,Qar,Qsd",
                xt_B_bs,
                s_ba,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qss,
            )
            - 2
            * oe.contract(
                "bs,rc,da,cb,Qar,Qsd",
                xt_B_bs,
                s_rs,
                s_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qss,
            )
            - 4
            * oe.contract(
                "ar,sc,db,cd,Qar,Qbs",
                xt_A_ar,
                s_sr,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,bc,ce,sb,Qar,Qes",
                xt_A_ar,
                s_ba,
                s_ab,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,cs,dc,sb,Qar,Qbd",
                xt_A_ar,
                s_as,
                s_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,sc,cb,ea,Qre,Qbs",
                xt_A_ar,
                s_sa,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qrr,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,sa,cb,dc,Qrd,Qbs",
                xt_A_ar,
                s_sa,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qrr,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,sc,db,ca,Qrd,Qbs",
                xt_A_ar,
                s_sr,
                s_rb,
                mol.get_cpscf_ra(),
                mol.Qrr,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,ba,cb,sd,Qrc,Qds",
                xt_A_ar,
                s_ba,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qrr,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,ba,cd,sb,Qrc,Qds",
                xt_A_ar,
                s_ba,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qrr,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,cs,da,sb,Qrc,Qbd",
                xt_A_ar,
                s_rs,
                s_sa,
                mol.get_cpscf_sb(),
                mol.Qrr,
                mol.Qbs,
            )
            # < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
            - 2 * oe.contract("sr,ab,ra,bs", s_sr, s_ab, u_ra, mol.omegaA_bs)
            + 2 * oe.contract("ba,cb,rc,ar", s_ba, s_ab, u_ra, mol.omegaB_ar)
            - 2 * oe.contract("br,cb,ra,ac", s_br, s_rb, u_ra, mol.omegaB_ar)
            - 4 * oe.contract("sa,ab,rc,Qcr,Qbs", s_sa, s_ab, u_ra, mol.Qar, mol.Qbs)
            + 2 * oe.contract("sa,cb,rc,Qar,Qbs", s_sa, s_ab, u_ra, mol.Qar, mol.Qbs)
            - 2 * oe.contract("sr,cb,ra,Qac,Qbs", s_sr, s_rb, u_ra, mol.Qar, mol.Qbs)
            + 2 * oe.contract("ba,ac,sb,cs", s_ba, s_ab, u_sb, mol.omegaA_bs)
            - 2 * oe.contract("as,ca,sb,bc", s_as, s_sa, u_sb, mol.omegaA_bs)
            - 2 * oe.contract("rs,ba,sb,ar", s_rs, s_ba, u_sb, mol.omegaB_ar)
            - 4 * oe.contract("ba,rb,sc,Qar,Qcs", s_ba, s_rb, u_sb, mol.Qar, mol.Qbs)
            + 2 * oe.contract("ba,rc,sb,Qar,Qcs", s_ba, s_rb, u_sb, mol.Qar, mol.Qbs)
            - 2 * oe.contract("rs,ca,sb,Qar,Qbc", s_rs, s_sa, u_sb, mol.Qar, mol.Qbs)
            # < V P2 R(X) R(V) >
            + 2
            * oe.contract(
                "sb,br,ac,ra,cs", xt_B_sb, s_br, s_ab, mol.get_cpscf_ra(), mol.omegaA_bs
            )
            - 2
            * oe.contract(
                "sb,cr,as,ra,bc", xt_B_sb, s_sr, s_as, mol.get_cpscf_ra(), mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "ra,br,ac,sb,cs", xt_A_ra, s_br, s_ab, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            - 2
            * oe.contract(
                "ra,sr,ac,cb,bs", xt_A_ra, s_sr, s_as, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "sb,as,ca,dc,bd", xt_B_sb, s_as, s_ba, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "sb,ac,ba,ce,es", xt_B_sb, s_as, s_ba, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "ra,br,cb,dc,ad", xt_A_ra, s_br, s_ab, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "ra,bc,ab,ce,er", xt_A_ra, s_br, s_ab, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "sb,as,bc,ra,cr", xt_B_sb, s_as, s_ba, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            - 2
            * oe.contract(
                "sb,br,cs,ra,ac", xt_B_sb, s_br, s_rs, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "ra,as,bc,sb,cr", xt_A_ra, s_as, s_ba, mol.get_cpscf_sb(), mol.omegaB_ar
            )
            - 2
            * oe.contract(
                "ra,br,cs,sb,ac", xt_A_ra, s_br, s_rs, mol.get_cpscf_sb(), mol.omegaB_ar
            )
            - 2
            * oe.contract(
                "sb,ba,cd,rc,Qar,Qds",
                xt_B_sb,
                s_ba,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "sb,ba,ac,rd,Qdr,Qcs",
                xt_B_sb,
                s_ba,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "sb,ca,dc,rd,Qar,Qbs",
                xt_B_sb,
                s_ba,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ra,sr,ab,cd,Qdc,Qbs",
                xt_A_ra,
                s_sr,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ra,sc,db,cd,Qar,Qbs",
                xt_A_ra,
                s_sr,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ra,sr,cb,dc,Qad,Qbs",
                xt_A_ra,
                s_sr,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ra,sc,ab,ce,Qer,Qbs",
                xt_A_ra,
                s_sr,
                s_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,cr,dc,ra,Qad,Qbs",
                xt_B_sb,
                s_br,
                s_rb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "sb,br,cd,ra,Qac,Qds",
                xt_B_sb,
                s_br,
                s_rb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "sb,cr,ds,ra,Qad,Qbc",
                xt_B_sb,
                s_sr,
                s_rs,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,as,ca,rd,Qdr,Qbc",
                xt_B_sb,
                s_as,
                s_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "sb,as,cd,ra,Qdr,Qbc",
                xt_B_sb,
                s_as,
                s_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ra,bc,ad,sb,Qcr,Qds",
                xt_A_ra,
                s_ba,
                s_ab,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ra,bc,ab,sd,Qcr,Qds",
                xt_A_ra,
                s_ba,
                s_ab,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ra,bc,ce,sb,Qar,Qes",
                xt_A_ra,
                s_ba,
                s_ab,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,rs,ba,cd,Qar,Qdc",
                xt_B_sb,
                s_rs,
                s_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,rc,da,cd,Qar,Qbs",
                xt_B_sb,
                s_rs,
                s_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "sb,rs,ca,dc,Qar,Qbd",
                xt_B_sb,
                s_rs,
                s_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "sb,rc,ba,ce,Qar,Qes",
                xt_B_sb,
                s_rs,
                s_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ra,br,cb,sd,Qac,Qds",
                xt_A_ra,
                s_br,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ra,br,cd,sb,Qac,Qds",
                xt_A_ra,
                s_br,
                s_rb,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ra,sr,cd,db,Qac,Qbs",
                xt_A_ra,
                s_sr,
                s_rs,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ra,cs,dc,sb,Qar,Qbd",
                xt_A_ra,
                s_as,
                s_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ra,as,cd,sb,Qdr,Qbc",
                xt_A_ra,
                s_as,
                s_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
            )
        ]
    )

    return x2_exch_ind_resp_s2
