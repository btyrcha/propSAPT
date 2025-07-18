import numpy as np
import opt_einsum as oe

from prop_sapt.molecule import Dimer


def get_u_rsab_amplitudes(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_rr: np.ndarray,
    prop_B_bb: np.ndarray,
    prop_B_ss: np.ndarray,
) -> np.ndarray:

    u_rsab = (
        # R([X, R(V)])
        oe.contract("rR,Rsab->rsab", prop_A_rr, mol.t_rsab)
        - oe.contract("Aa,rsAb->rsab", prop_A_aa, mol.t_rsab)
        + oe.contract("sS,rSab->rsab", prop_B_ss, mol.t_rsab)
        - oe.contract("Bb,rsaB->rsab", prop_B_bb, mol.t_rsab)
        # R([V, R(X)])
        + oe.contract("Ra,QrR,Qbs->rsab", xt_A_ra, mol.Qrr, mol.Qbs)
        - oe.contract("rA,QAa,Qbs->rsab", xt_A_ra, mol.Qaa, mol.Qbs)
        + oe.contract("Sb,Qar,QsS->rsab", xt_B_sb, mol.Qar, mol.Qss)
        - oe.contract("sB,Qar,QBb->rsab", xt_B_sb, mol.Qar, mol.Qbb)
    )

    u_rsab = oe.contract("rsab,rsab->rsab", u_rsab, mol.e_rsab)

    return u_rsab


def calc_exch_disp2_s2_property(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_rr: np.ndarray,
    prop_B_bb: np.ndarray,
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

    u_rsab = get_u_rsab_amplitudes(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        prop_A_aa=prop_A_aa,
        prop_A_rr=prop_A_rr,
        prop_B_bb=prop_B_bb,
        prop_B_ss=prop_B_ss,
    )

    # < R(X) | V P2 R(V) >
    x2_exch_disp_s2 = np.array(
        [
            -2
            * oe.contract(
                "bs,cr,ad,rsac,db", xt_B_bs, s_br, s_ab, mol.t_rsab, mol.omegaA_bb
            )
            + 4
            * oe.contract(
                "bs,cr,ac,rsae,eb", xt_B_bs, s_br, s_ab, mol.t_rsab, mol.omegaA_bb
            )
            + 2
            * oe.contract(
                "bs,sr,ac,rcae,eb", xt_B_bs, s_sr, s_as, mol.t_rsab, mol.omegaA_bb
            )
            - 2
            * oe.contract(
                "ar,ba,cd,rscb,ds", xt_A_ar, s_ba, s_ab, mol.t_rsab, mol.omegaA_bs
            )
            + 4
            * oe.contract(
                "ar,ba,cb,rsce,es", xt_A_ar, s_ba, s_ab, mol.t_rsab, mol.omegaA_bs
            )
            + 4
            * oe.contract(
                "ar,bc,ce,rsab,es", xt_A_ar, s_ba, s_ab, mol.t_rsab, mol.omegaA_bs
            )
            - 4
            * oe.contract(
                "bs,sr,ab,rcad,dc", xt_B_bs, s_sr, s_ab, mol.t_rsab, mol.omegaA_bs
            )
            - 4
            * oe.contract(
                "bs,cr,ad,rsab,dc", xt_B_bs, s_sr, s_ab, mol.t_rsab, mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "bs,sr,ac,rdab,cd", xt_B_bs, s_sr, s_ab, mol.t_rsab, mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "bs,cr,ab,rsad,dc", xt_B_bs, s_sr, s_ab, mol.t_rsab, mol.omegaA_bs
            )
            - 4
            * oe.contract(
                "ar,bc,rb,csae,es", xt_A_ar, s_br, s_rb, mol.t_rsab, mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "ar,bc,rd,csab,ds", xt_A_ar, s_br, s_rb, mol.t_rsab, mol.omegaA_bs
            )
            - 2
            * oe.contract(
                "ar,sc,rd,cdab,bs", xt_A_ar, s_sr, s_rs, mol.t_rsab, mol.omegaA_bs
            )
            - 4
            * oe.contract(
                "ar,cs,dc,rsab,bd", xt_A_ar, s_as, s_sa, mol.t_rsab, mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "ar,cs,da,rscb,bd", xt_A_ar, s_as, s_sa, mol.t_rsab, mol.omegaA_bs
            )
            - 4
            * oe.contract(
                "bs,cr,ac,reab,se", xt_B_bs, s_br, s_ab, mol.t_rsab, mol.omegaA_ss
            )
            + 2
            * oe.contract(
                "bs,cr,ab,rdac,sd", xt_B_bs, s_br, s_ab, mol.t_rsab, mol.omegaA_ss
            )
            - 2
            * oe.contract(
                "bs,cr,ad,rdab,sc", xt_B_bs, s_sr, s_as, mol.t_rsab, mol.omegaA_ss
            )
            - 2
            * oe.contract(
                "ar,cs,bd,rscb,da", xt_A_ar, s_as, s_ba, mol.t_rsab, mol.omegaB_aa
            )
            + 4
            * oe.contract(
                "ar,cs,bc,rseb,ea", xt_A_ar, s_as, s_ba, mol.t_rsab, mol.omegaB_aa
            )
            + 2
            * oe.contract(
                "ar,bc,rs,cseb,ea", xt_A_ar, s_br, s_rs, mol.t_rsab, mol.omegaB_aa
            )
            - 2
            * oe.contract(
                "bs,ca,db,rsdc,ar", xt_B_bs, s_ba, s_ab, mol.t_rsab, mol.omegaB_ar
            )
            + 4
            * oe.contract(
                "bs,ca,ab,rsdc,dr", xt_B_bs, s_ba, s_ab, mol.t_rsab, mol.omegaB_ar
            )
            + 4
            * oe.contract(
                "bs,ca,dc,rsdb,ar", xt_B_bs, s_ba, s_ab, mol.t_rsab, mol.omegaB_ar
            )
            - 4
            * oe.contract(
                "ar,rs,ba,csdb,dc", xt_A_ar, s_rs, s_ba, mol.t_rsab, mol.omegaB_ar
            )
            - 4
            * oe.contract(
                "ar,cs,bd,rsab,dc", xt_A_ar, s_rs, s_ba, mol.t_rsab, mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "ar,rs,bc,dsab,cd", xt_A_ar, s_rs, s_ba, mol.t_rsab, mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "ar,cs,ba,rsdb,dc", xt_A_ar, s_rs, s_ba, mol.t_rsab, mol.omegaB_ar
            )
            - 4
            * oe.contract(
                "bs,cr,dc,rsab,ad", xt_B_bs, s_br, s_rb, mol.t_rsab, mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "bs,cr,db,rsac,ad", xt_B_bs, s_br, s_rb, mol.t_rsab, mol.omegaB_ar
            )
            - 2
            * oe.contract(
                "bs,sr,cd,rdab,ac", xt_B_bs, s_sr, s_rs, mol.t_rsab, mol.omegaB_ar
            )
            - 4
            * oe.contract(
                "bs,ac,sa,rceb,er", xt_B_bs, s_as, s_sa, mol.t_rsab, mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "bs,ac,sd,rcab,dr", xt_B_bs, s_as, s_sa, mol.t_rsab, mol.omegaB_ar
            )
            - 4
            * oe.contract(
                "ar,cs,bc,esab,re", xt_A_ar, s_as, s_ba, mol.t_rsab, mol.omegaB_rr
            )
            + 2
            * oe.contract(
                "ar,cs,ba,dscb,rd", xt_A_ar, s_as, s_ba, mol.t_rsab, mol.omegaB_rr
            )
            - 2
            * oe.contract(
                "ar,bc,ds,csab,rd", xt_A_ar, s_br, s_rs, mol.t_rsab, mol.omegaB_rr
            )
            - 4
            * oe.contract(
                "ar,bc,db,rsdf,Qca,Qfs",
                xt_A_ar,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qaa,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,bc,ce,rsfb,Qfa,Qes",
                xt_A_ar,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qaa,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,bc,de,rsdb,Qca,Qes",
                xt_A_ar,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qaa,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,bc,rd,csfb,Qfa,Qds",
                xt_A_ar,
                s_br,
                s_rb,
                mol.t_rsab,
                mol.Qaa,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,bc,rb,csef,Qea,Qfs",
                xt_A_ar,
                s_br,
                s_rb,
                mol.t_rsab,
                mol.Qaa,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,sc,rd,cdgb,Qga,Qbs",
                xt_A_ar,
                s_sr,
                s_rs,
                mol.t_rsab,
                mol.Qaa,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,cs,de,rscb,Qea,Qbd",
                xt_A_ar,
                s_as,
                s_sa,
                mol.t_rsab,
                mol.Qaa,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,cs,dc,rsfb,Qfa,Qbd",
                xt_A_ar,
                s_as,
                s_sa,
                mol.t_rsab,
                mol.Qaa,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,ca,dc,rsdg,Qar,Qgb",
                xt_B_bs,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbb,
            )
            - 4
            * oe.contract(
                "bs,ca,ad,rsec,Qer,Qdb",
                xt_B_bs,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbb,
            )
            + 2
            * oe.contract(
                "bs,ca,de,rsdc,Qar,Qeb",
                xt_B_bs,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbb,
            )
            - 2
            * oe.contract(
                "bs,cr,de,rsac,Qad,Qeb",
                xt_B_bs,
                s_br,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbb,
            )
            + 4
            * oe.contract(
                "bs,cr,dc,rsaf,Qad,Qfb",
                xt_B_bs,
                s_br,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbb,
            )
            + 2
            * oe.contract(
                "bs,sr,cd,rdaf,Qac,Qfb",
                xt_B_bs,
                s_sr,
                s_rs,
                mol.t_rsab,
                mol.Qar,
                mol.Qbb,
            )
            - 2
            * oe.contract(
                "bs,ac,sd,rcaf,Qdr,Qfb",
                xt_B_bs,
                s_as,
                s_sa,
                mol.t_rsab,
                mol.Qar,
                mol.Qbb,
            )
            + 4
            * oe.contract(
                "bs,ac,sa,rcef,Qer,Qfb",
                xt_B_bs,
                s_as,
                s_sa,
                mol.t_rsab,
                mol.Qar,
                mol.Qbb,
            )
            - 8
            * oe.contract(
                "bs,ca,ad,rseb,Qer,Qdc",
                xt_B_bs,
                s_sa,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "bs,sa,cd,recb,Qar,Qde",
                xt_B_bs,
                s_sa,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "bs,ca,db,rsdf,Qar,Qfc",
                xt_B_bs,
                s_sa,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "bs,sa,cb,rdcf,Qar,Qfd",
                xt_B_bs,
                s_sa,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "bs,sa,ac,rdeb,Qer,Qcd",
                xt_B_bs,
                s_sa,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "bs,ca,ab,rsde,Qdr,Qec",
                xt_B_bs,
                s_sa,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "bs,ca,de,rsdb,Qar,Qec",
                xt_B_bs,
                s_sa,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 8
            * oe.contract(
                "ar,bc,db,rsae,Qcd,Qes",
                xt_A_ar,
                s_ba,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,ba,cd,rseb,Qec,Qds",
                xt_A_ar,
                s_ba,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,bc,rd,esab,Qce,Qds",
                xt_A_ar,
                s_ba,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,ba,cb,rsde,Qdc,Qes",
                xt_A_ar,
                s_ba,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,ba,rc,dseb,Qed,Qcs",
                xt_A_ar,
                s_ba,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,bc,rb,dsae,Qcd,Qes",
                xt_A_ar,
                s_ba,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,bc,de,rsab,Qcd,Qes",
                xt_A_ar,
                s_ba,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,sr,cb,rdae,Qac,Qed",
                xt_B_bs,
                s_sr,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,cr,de,rsab,Qad,Qec",
                xt_B_bs,
                s_sr,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "bs,sr,cd,reab,Qac,Qde",
                xt_B_bs,
                s_sr,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "bs,cr,db,rsae,Qad,Qec",
                xt_B_bs,
                s_sr,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,rs,ca,dseb,Qed,Qbc",
                xt_A_ar,
                s_rs,
                s_sa,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,cs,de,rsab,Qec,Qbd",
                xt_A_ar,
                s_rs,
                s_sa,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,rs,cd,esab,Qde,Qbc",
                xt_A_ar,
                s_rs,
                s_sa,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,cs,da,rseb,Qec,Qbd",
                xt_A_ar,
                s_rs,
                s_sa,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 8
            * oe.contract(
                "bs,ac,da,rcfd,Qfr,Qbs",
                xt_B_bs,
                s_as,
                s_ba,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "bs,ac,de,rcad,Qer,Qbs",
                xt_B_bs,
                s_as,
                s_ba,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,cr,de,reac,Qad,Qbs",
                xt_B_bs,
                s_br,
                s_rs,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "bs,ca,db,redc,Qar,Qse",
                xt_B_bs,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qss,
            )
            + 4
            * oe.contract(
                "bs,ca,ab,rdec,Qer,Qsd",
                xt_B_bs,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qss,
            )
            + 4
            * oe.contract(
                "bs,ca,dc,rfdb,Qar,Qsf",
                xt_B_bs,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qss,
            )
            - 4
            * oe.contract(
                "bs,cr,dc,rfab,Qad,Qsf",
                xt_B_bs,
                s_br,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qss,
            )
            + 2
            * oe.contract(
                "bs,cr,db,reac,Qad,Qse",
                xt_B_bs,
                s_br,
                s_rb,
                mol.t_rsab,
                mol.Qar,
                mol.Qss,
            )
            - 2
            * oe.contract(
                "bs,cr,de,reab,Qad,Qsc",
                xt_B_bs,
                s_sr,
                s_rs,
                mol.t_rsab,
                mol.Qar,
                mol.Qss,
            )
            - 4
            * oe.contract(
                "bs,ac,da,rcfb,Qfr,Qsd",
                xt_B_bs,
                s_as,
                s_sa,
                mol.t_rsab,
                mol.Qar,
                mol.Qss,
            )
            + 2
            * oe.contract(
                "bs,ac,de,rcab,Qer,Qsd",
                xt_B_bs,
                s_as,
                s_sa,
                mol.t_rsab,
                mol.Qar,
                mol.Qss,
            )
            - 8
            * oe.contract(
                "ar,bc,db,csdg,Qar,Qgs",
                xt_A_ar,
                s_br,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,bc,de,csdb,Qar,Qes",
                xt_A_ar,
                s_br,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,sc,de,cedb,Qar,Qbs",
                xt_A_ar,
                s_sr,
                s_as,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,ba,cd,escb,Qre,Qds",
                xt_A_ar,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qrr,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,ba,cb,dscf,Qrd,Qfs",
                xt_A_ar,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qrr,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,bc,ce,fsab,Qrf,Qes",
                xt_A_ar,
                s_ba,
                s_ab,
                mol.t_rsab,
                mol.Qrr,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,bc,db,csaf,Qrd,Qfs",
                xt_A_ar,
                s_br,
                s_rb,
                mol.t_rsab,
                mol.Qrr,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,bc,de,csab,Qrd,Qes",
                xt_A_ar,
                s_br,
                s_rb,
                mol.t_rsab,
                mol.Qrr,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,sc,de,ceab,Qrd,Qbs",
                xt_A_ar,
                s_sr,
                s_rs,
                mol.t_rsab,
                mol.Qrr,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,cs,dc,fsab,Qrf,Qbd",
                xt_A_ar,
                s_as,
                s_sa,
                mol.t_rsab,
                mol.Qrr,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,cs,da,escb,Qre,Qbd",
                xt_A_ar,
                s_as,
                s_sa,
                mol.t_rsab,
                mol.Qrr,
                mol.Qbs,
            )
        ]
    )

    # < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
    x2_exch_disp_s2 += np.array(
        [
            -4 * oe.contract("br,ab,rsac,cs", s_br, s_ab, u_rsab, mol.omegaA_bs)
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

    # < V P2 R(X) R(V) >
    x2_exch_disp_s2 += np.array(
        [
            -4
            * oe.contract(
                "sb,br,as,rcad,dc", xt_B_sb, s_br, s_as, mol.t_rsab, mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "sb,br,ac,rcae,es", xt_B_sb, s_br, s_as, mol.t_rsab, mol.omegaA_bs
            )
            + 2
            * oe.contract(
                "sb,cr,as,rdac,bd", xt_B_sb, s_br, s_as, mol.t_rsab, mol.omegaA_bs
            )
            - 4
            * oe.contract(
                "ra,br,as,csdb,dc", xt_A_ra, s_br, s_as, mol.t_rsab, mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "ra,br,cs,dscb,ad", xt_A_ra, s_br, s_as, mol.t_rsab, mol.omegaB_ar
            )
            + 2
            * oe.contract(
                "ra,bc,as,cseb,er", xt_A_ra, s_br, s_as, mol.t_rsab, mol.omegaB_ar
            )
            - 8
            * oe.contract(
                "ra,bc,db,csdg,Qar,Qgs",
                xt_A_ra,
                s_br,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ra,br,cd,escb,Qae,Qds",
                xt_A_ra,
                s_br,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ra,bc,ad,csfb,Qfr,Qds",
                xt_A_ra,
                s_br,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ra,br,cb,dscf,Qad,Qfs",
                xt_A_ra,
                s_br,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ra,br,ac,dseb,Qed,Qcs",
                xt_A_ra,
                s_br,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ra,bc,ab,csef,Qer,Qfs",
                xt_A_ra,
                s_br,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ra,bc,de,csdb,Qar,Qes",
                xt_A_ra,
                s_br,
                s_ab,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ra,sr,ac,dcfb,Qfd,Qbs",
                xt_A_ra,
                s_sr,
                s_as,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ra,sc,de,cedb,Qar,Qbs",
                xt_A_ra,
                s_sr,
                s_as,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ra,sr,cd,edcb,Qae,Qbs",
                xt_A_ra,
                s_sr,
                s_as,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ra,sc,ad,cdgb,Qgr,Qbs",
                xt_A_ra,
                s_sr,
                s_as,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 8
            * oe.contract(
                "sb,ac,da,rcfd,Qfr,Qbs",
                xt_B_sb,
                s_as,
                s_ba,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "sb,as,cd,reac,Qdr,Qbe",
                xt_B_sb,
                s_as,
                s_ba,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "sb,ac,bd,rcaf,Qdr,Qfs",
                xt_B_sb,
                s_as,
                s_ba,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "sb,as,ca,rdec,Qer,Qbd",
                xt_B_sb,
                s_as,
                s_ba,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "sb,as,bc,rdae,Qcr,Qed",
                xt_B_sb,
                s_as,
                s_ba,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "sb,ac,ba,rcef,Qer,Qfs",
                xt_B_sb,
                s_as,
                s_ba,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "sb,ac,de,rcad,Qer,Qbs",
                xt_B_sb,
                s_as,
                s_ba,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,br,cs,rdae,Qac,Qed",
                xt_B_sb,
                s_br,
                s_rs,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,cr,de,reac,Qad,Qbs",
                xt_B_sb,
                s_br,
                s_rs,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "sb,br,cd,rdaf,Qac,Qfs",
                xt_B_sb,
                s_br,
                s_rs,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "sb,cr,ds,reac,Qad,Qbe",
                xt_B_sb,
                s_br,
                s_rs,
                mol.t_rsab,
                mol.Qar,
                mol.Qbs,
            )
        ]
    )

    return x2_exch_disp_s2
