import numpy as np
import opt_einsum as oe

from prop_sapt.molecule import Dimer


def calc_exch_ind2_resp_sinf_property(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    u_ra: np.ndarray,
    u_sb: np.ndarray,
) -> np.ndarray:

    xt_A_ar = xt_A_ra.T
    xt_B_bs = xt_B_sb.T

    vrx_ar = (
        -oe.contract("Ar,aA->ar", xt_A_ar, mol.omegaB_aa)
        + oe.contract("aR,Rr->ar", xt_A_ar, mol.omegaB_rr)
        + 2 * oe.contract("bs,Qar,Qbs->ar", xt_B_bs, mol.Qar, mol.Qbs)
    )

    vrx_bs = (
        -oe.contract("Bs,bB->bs", xt_B_bs, mol.omegaA_bb)
        + oe.contract("bS,Ss->bs", xt_B_bs, mol.omegaA_ss)
        + 2 * oe.contract("ar,Qar,Qbs->bs", xt_A_ar, mol.Qar, mol.Qbs)
    )

    vrx_abrs = (
        oe.contract("aR,QRr,Qbs->abrs", xt_A_ar, mol.Qrr, mol.Qbs)
        - oe.contract("Ar,QaA,Qbs->abrs", xt_A_ar, mol.Qaa, mol.Qbs)
        + oe.contract("bS,Qar,QSs->abrs", xt_B_bs, mol.Qar, mol.Qss)
        - oe.contract("Bs,Qar,QbB->abrs", xt_B_bs, mol.Qar, mol.Qbb)
    )

    x2_exch_ind_resp_sinf = np.array(
        [
            # < R(X) | V P R(V) >
            -2
            * oe.contract("cd,ef,de,fc", mol.G_sr, mol.H_ab, mol.get_cpscf_ra(), vrx_bs)
            - 2
            * oe.contract("cd,ef,de,fc", mol.G_rs, mol.H_ba, mol.get_cpscf_sb(), vrx_ar)
            + 2
            * oe.contract("cd,ef,fc,de", mol.A_bb, mol.D_ss, mol.get_cpscf_sb(), vrx_bs)
            + 2
            * oe.contract("cd,ef,fc,de", mol.B_aa, mol.C_rr, mol.get_cpscf_ra(), vrx_ar)
            - 4
            * oe.contract(
                "cd,ef,gh,he,dfcg",
                mol.E_ra,
                mol.A_bb,
                mol.D_ss,
                mol.get_cpscf_sb(),
                vrx_abrs,
            )
            - 4
            * oe.contract(
                "cd,ef,gh,fc,dheg",
                mol.B_aa,
                mol.C_rr,
                mol.F_sb,
                mol.get_cpscf_ra(),
                vrx_abrs,
            )
            - 2
            * oe.contract(
                "cd,ef,gh,he,fdcg",
                mol.I_rb,
                mol.B_aa,
                mol.G_sr,
                mol.get_cpscf_ra(),
                vrx_abrs,
            )
            - 2
            * oe.contract(
                "cd,ef,gh,fc,hdeg",
                mol.A_bb,
                mol.G_rs,
                mol.J_sa,
                mol.get_cpscf_sb(),
                vrx_abrs,
            )
            + 2
            * oe.contract(
                "cd,ef,gh,fg,hdce",
                mol.I_rb,
                mol.D_ss,
                mol.H_ba,
                mol.get_cpscf_sb(),
                vrx_abrs,
            )
            + 2
            * oe.contract(
                "cd,ef,gh,de,hfcg",
                mol.C_rr,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                vrx_abrs,
            )
            + 4
            * oe.contract(
                "cd,ef,gh,fg,dhce",
                mol.E_ra,
                mol.G_sr,
                mol.H_ab,
                mol.get_cpscf_ra(),
                vrx_abrs,
            )
            + 4
            * oe.contract(
                "cd,ef,gh,fg,hdec",
                mol.F_sb,
                mol.G_rs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                vrx_abrs,
            )
            - 2
            * oe.contract(
                "ca,rd,ef,ar,de,fc",
                mol.E_ra,
                mol.G_rs,
                mol.H_ba,
                mol.omegaB_ar,
                mol.get_cpscf_sb(),
                xt_A_ar,
            )
            - 2
            * oe.contract(
                "rc,de,fa,ar,ef,cd",
                mol.E_ra,
                mol.G_rs,
                mol.H_ba,
                mol.omegaB_ar,
                mol.get_cpscf_sb(),
                xt_A_ar,
            )
            - 2
            * oe.contract(
                "cb,de,sf,bs,fd,ec",
                mol.I_rb,
                mol.B_aa,
                mol.G_sr,
                mol.omegaA_bs,
                mol.get_cpscf_ra(),
                xt_A_ar,
            )
            - 2
            * oe.contract(
                "rc,da,ef,ar,fd,ce",
                mol.I_rb,
                mol.B_aa,
                mol.G_sr,
                mol.omegaB_ar,
                mol.get_cpscf_ra(),
                xt_B_bs,
            )
            - 2
            * oe.contract(
                "cb,de,sf,bs,ec,fd",
                mol.A_bb,
                mol.G_rs,
                mol.J_sa,
                mol.omegaA_bs,
                mol.get_cpscf_sb(),
                xt_A_ar,
            )
            - 2
            * oe.contract(
                "cd,re,fa,ar,ec,df",
                mol.A_bb,
                mol.G_rs,
                mol.J_sa,
                mol.omegaB_ar,
                mol.get_cpscf_sb(),
                xt_B_bs,
            )
            - 2
            * oe.contract(
                "cb,sd,ef,bs,de,fc",
                mol.F_sb,
                mol.G_sr,
                mol.H_ab,
                mol.omegaA_bs,
                mol.get_cpscf_ra(),
                xt_B_bs,
            )
            - 2
            * oe.contract(
                "sc,de,fb,bs,ef,cd",
                mol.F_sb,
                mol.G_sr,
                mol.H_ab,
                mol.omegaA_bs,
                mol.get_cpscf_ra(),
                xt_B_bs,
            )
            + 2
            * oe.contract(
                "ca,de,rf,ar,fd,ec",
                mol.E_ra,
                mol.B_aa,
                mol.C_rr,
                mol.omegaB_ar,
                mol.get_cpscf_ra(),
                xt_A_ar,
            )
            + 2
            * oe.contract(
                "rc,da,ef,ar,fd,ce",
                mol.E_ra,
                mol.B_aa,
                mol.C_rr,
                mol.omegaB_ar,
                mol.get_cpscf_ra(),
                xt_A_ar,
            )
            + 2
            * oe.contract(
                "cb,sd,ef,bs,de,fc",
                mol.I_rb,
                mol.D_ss,
                mol.H_ba,
                mol.omegaA_bs,
                mol.get_cpscf_sb(),
                xt_A_ar,
            )
            + 2
            * oe.contract(
                "rc,de,fa,ar,ef,cd",
                mol.I_rb,
                mol.D_ss,
                mol.H_ba,
                mol.omegaB_ar,
                mol.get_cpscf_sb(),
                xt_B_bs,
            )
            + 2
            * oe.contract(
                "cb,de,sf,bs,ec,fd",
                mol.A_bb,
                mol.D_ss,
                mol.F_sb,
                mol.omegaA_bs,
                mol.get_cpscf_sb(),
                xt_B_bs,
            )
            + 2
            * oe.contract(
                "cd,se,fb,bs,ec,df",
                mol.A_bb,
                mol.D_ss,
                mol.F_sb,
                mol.omegaA_bs,
                mol.get_cpscf_sb(),
                xt_B_bs,
            )
            + 2
            * oe.contract(
                "rc,de,fa,ar,cd,ef",
                mol.C_rr,
                mol.H_ab,
                mol.J_sa,
                mol.omegaB_ar,
                mol.get_cpscf_ra(),
                xt_B_bs,
            )
            + 2
            * oe.contract(
                "cd,eb,sf,bs,de,fc",
                mol.C_rr,
                mol.H_ab,
                mol.J_sa,
                mol.omegaA_bs,
                mol.get_cpscf_ra(),
                xt_A_ar,
            )
            - 4
            * oe.contract(
                "ra,cb,sd,ef,de,Qar,Qbs,fc",
                mol.E_ra,
                mol.I_rb,
                mol.D_ss,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 4
            * oe.contract(
                "ra,cb,de,sf,ec,Qar,Qbs,fd",
                mol.E_ra,
                mol.A_bb,
                mol.D_ss,
                mol.F_sb,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            - 4
            * oe.contract(
                "ra,cd,se,fb,ec,Qar,Qbs,df",
                mol.E_ra,
                mol.A_bb,
                mol.D_ss,
                mol.F_sb,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            - 4
            * oe.contract(
                "ra,cd,eb,sf,de,Qar,Qbs,fc",
                mol.E_ra,
                mol.C_rr,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 4
            * oe.contract(
                "ca,rd,eb,sf,fe,Qar,Qbs,dc",
                mol.E_ra,
                mol.E_ra,
                mol.A_bb,
                mol.D_ss,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 4
            * oe.contract(
                "ca,de,rf,sb,fd,Qar,Qbs,ec",
                mol.E_ra,
                mol.B_aa,
                mol.C_rr,
                mol.F_sb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 4
            * oe.contract(
                "rc,da,ef,sb,fd,Qar,Qbs,ce",
                mol.E_ra,
                mol.B_aa,
                mol.C_rr,
                mol.F_sb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 4
            * oe.contract(
                "cb,da,re,sf,ed,Qar,Qbs,fc",
                mol.I_rb,
                mol.B_aa,
                mol.C_rr,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 4
            * oe.contract(
                "rc,db,se,fa,ed,Qar,Qbs,cf",
                mol.I_rb,
                mol.A_bb,
                mol.D_ss,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            - 4
            * oe.contract(
                "rc,de,sb,fa,ef,Qar,Qbs,cd",
                mol.I_rb,
                mol.D_ss,
                mol.F_sb,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            - 4
            * oe.contract(
                "ca,rd,eb,sf,dc,Qar,Qbs,fe",
                mol.B_aa,
                mol.C_rr,
                mol.F_sb,
                mol.F_sb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            - 4
            * oe.contract(
                "rc,sb,de,fa,cd,Qar,Qbs,ef",
                mol.C_rr,
                mol.F_sb,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            - 2
            * oe.contract(
                "ca,rb,de,sf,fd,Qar,Qbs,ec",
                mol.E_ra,
                mol.I_rb,
                mol.B_aa,
                mol.G_sr,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 2
            * oe.contract(
                "ca,db,re,sf,ed,Qar,Qbs,fc",
                mol.E_ra,
                mol.A_bb,
                mol.G_rs,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 2
            * oe.contract(
                "rc,db,ea,sf,fe,Qar,Qbs,cd",
                mol.E_ra,
                mol.I_rb,
                mol.B_aa,
                mol.G_sr,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 2
            * oe.contract(
                "rc,db,ef,sa,fd,Qar,Qbs,ce",
                mol.E_ra,
                mol.A_bb,
                mol.G_rs,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 2
            * oe.contract(
                "rb,ca,sd,ef,fc,Qar,Qbs,de",
                mol.I_rb,
                mol.B_aa,
                mol.F_sb,
                mol.G_sr,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            - 2
            * oe.contract(
                "rb,sc,de,fa,cd,Qar,Qbs,ef",
                mol.I_rb,
                mol.G_sr,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            - 2
            * oe.contract(
                "rb,cd,ea,sf,de,Qar,Qbs,fc",
                mol.I_rb,
                mol.G_rs,
                mol.H_ba,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 2
            * oe.contract(
                "cb,rd,ef,sa,de,Qar,Qbs,fc",
                mol.I_rb,
                mol.G_rs,
                mol.H_ba,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            - 2
            * oe.contract(
                "rc,da,eb,sf,fd,Qar,Qbs,ce",
                mol.I_rb,
                mol.B_aa,
                mol.F_sb,
                mol.G_sr,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            - 2
            * oe.contract(
                "rc,de,fb,sa,ef,Qar,Qbs,cd",
                mol.I_rb,
                mol.G_sr,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            - 2
            * oe.contract(
                "cb,sd,re,fa,ec,Qar,Qbs,df",
                mol.A_bb,
                mol.F_sb,
                mol.G_rs,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            - 2
            * oe.contract(
                "cd,eb,rf,sa,fc,Qar,Qbs,de",
                mol.A_bb,
                mol.F_sb,
                mol.G_rs,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 2
            * oe.contract(
                "ca,rb,sd,ef,de,Qar,Qbs,fc",
                mol.E_ra,
                mol.I_rb,
                mol.D_ss,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 2
            * oe.contract(
                "ca,rd,eb,sf,de,Qar,Qbs,fc",
                mol.E_ra,
                mol.C_rr,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 2
            * oe.contract(
                "rc,db,se,fa,ef,Qar,Qbs,cd",
                mol.E_ra,
                mol.I_rb,
                mol.D_ss,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 2
            * oe.contract(
                "rc,de,fb,sa,ef,Qar,Qbs,cd",
                mol.E_ra,
                mol.C_rr,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 2
            * oe.contract(
                "rb,cd,se,fa,ec,Qar,Qbs,df",
                mol.I_rb,
                mol.A_bb,
                mol.D_ss,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 2
            * oe.contract(
                "rb,ca,de,sf,ec,Qar,Qbs,fd",
                mol.I_rb,
                mol.B_aa,
                mol.C_rr,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 2
            * oe.contract(
                "rb,cd,se,fa,df,Qar,Qbs,ec",
                mol.I_rb,
                mol.D_ss,
                mol.F_sb,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 2
            * oe.contract(
                "cb,de,rf,sa,fd,Qar,Qbs,ec",
                mol.I_rb,
                mol.B_aa,
                mol.C_rr,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 2
            * oe.contract(
                "rc,db,ef,sa,fd,Qar,Qbs,ce",
                mol.I_rb,
                mol.A_bb,
                mol.D_ss,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 2
            * oe.contract(
                "rc,sd,eb,fa,df,Qar,Qbs,ce",
                mol.I_rb,
                mol.D_ss,
                mol.F_sb,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 2
            * oe.contract(
                "rc,db,ef,sa,ce,Qar,Qbs,fd",
                mol.C_rr,
                mol.F_sb,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 2
            * oe.contract(
                "rc,sd,eb,fa,ce,Qar,Qbs,df",
                mol.C_rr,
                mol.F_sb,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 4
            * oe.contract(
                "ra,cb,de,sf,fd,Qar,Qbs,ec",
                mol.E_ra,
                mol.I_rb,
                mol.B_aa,
                mol.G_sr,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 4
            * oe.contract(
                "ra,cb,de,sf,ec,Qar,Qbs,fd",
                mol.E_ra,
                mol.A_bb,
                mol.G_rs,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 4
            * oe.contract(
                "ra,cb,sd,ef,de,Qar,Qbs,fc",
                mol.E_ra,
                mol.F_sb,
                mol.G_sr,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 4
            * oe.contract(
                "ra,sc,de,fb,ef,Qar,Qbs,cd",
                mol.E_ra,
                mol.F_sb,
                mol.G_sr,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 4
            * oe.contract(
                "ca,rd,se,fb,ef,Qar,Qbs,dc",
                mol.E_ra,
                mol.E_ra,
                mol.G_sr,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 4
            * oe.contract(
                "ca,sb,rd,ef,de,Qar,Qbs,fc",
                mol.E_ra,
                mol.F_sb,
                mol.G_rs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 4
            * oe.contract(
                "rc,sb,de,fa,ef,Qar,Qbs,cd",
                mol.E_ra,
                mol.F_sb,
                mol.G_rs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 4
            * oe.contract(
                "cb,rd,ea,sf,de,Qar,Qbs,fc",
                mol.I_rb,
                mol.G_rs,
                mol.H_ba,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ar,
            )
            + 4
            * oe.contract(
                "rc,da,sb,ef,fd,Qar,Qbs,ce",
                mol.I_rb,
                mol.B_aa,
                mol.F_sb,
                mol.G_sr,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 4
            * oe.contract(
                "rc,sd,eb,fa,de,Qar,Qbs,cf",
                mol.I_rb,
                mol.G_sr,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 4
            * oe.contract(
                "cd,sb,re,fa,ec,Qar,Qbs,df",
                mol.A_bb,
                mol.F_sb,
                mol.G_rs,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            + 4
            * oe.contract(
                "cb,sd,re,fa,ef,Qar,Qbs,dc",
                mol.F_sb,
                mol.F_sb,
                mol.G_rs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_bs,
            )
            # < V P R([V, R(X)]) > + < V P R([X, R(V)]) >
            - 2 * oe.contract("sc,db,bs,cd", mol.G_sr, mol.H_ab, mol.omegaA_bs, u_ra)
            - 2 * oe.contract("rc,da,ar,cd", mol.G_rs, mol.H_ba, mol.omegaB_ar, u_sb)
            + 2 * oe.contract("cb,sd,bs,dc", mol.A_bb, mol.D_ss, mol.omegaA_bs, u_sb)
            + 2 * oe.contract("ca,rd,ar,dc", mol.B_aa, mol.C_rr, mol.omegaB_ar, u_ra)
            - 4
            * oe.contract(
                "ra,cb,sd,dc,Qar,Qbs",
                mol.E_ra,
                mol.A_bb,
                mol.D_ss,
                u_sb,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ca,rd,sb,dc,Qar,Qbs",
                mol.B_aa,
                mol.C_rr,
                mol.F_sb,
                u_ra,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "rb,ca,sd,dc,Qar,Qbs",
                mol.I_rb,
                mol.B_aa,
                mol.G_sr,
                u_ra,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "cb,rd,sa,dc,Qar,Qbs",
                mol.A_bb,
                mol.G_rs,
                mol.J_sa,
                u_sb,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "rb,sc,da,cd,Qar,Qbs",
                mol.I_rb,
                mol.D_ss,
                mol.H_ba,
                u_sb,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "rc,db,sa,cd,Qar,Qbs",
                mol.C_rr,
                mol.H_ab,
                mol.J_sa,
                u_ra,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ra,sc,db,cd,Qar,Qbs",
                mol.E_ra,
                mol.G_sr,
                mol.H_ab,
                u_ra,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "sb,rc,da,cd,Qar,Qbs",
                mol.F_sb,
                mol.G_rs,
                mol.H_ba,
                u_sb,
                mol.Qar,
                mol.Qbs,
            )
            # < V P R(X) R(V) >
            - 2
            * oe.contract(
                "cd,se,fb,bs,ec,df",
                mol.E_ar,
                mol.G_sr,
                mol.H_ab,
                mol.omegaA_bs,
                mol.get_cpscf_ra(),
                xt_A_ra,
            )
            - 2
            * oe.contract(
                "cd,se,fb,bs,df,ec",
                mol.E_ar,
                mol.G_sr,
                mol.H_ab,
                mol.omegaA_bs,
                mol.get_cpscf_ra(),
                xt_A_ra,
            )
            - 2
            * oe.contract(
                "cd,ea,rf,ar,fc,de",
                mol.I_br,
                mol.B_aa,
                mol.G_rs,
                mol.omegaB_ar,
                mol.get_cpscf_sb(),
                xt_A_ra,
            )
            - 2
            * oe.contract(
                "cd,ea,rf,ar,de,fc",
                mol.I_br,
                mol.B_aa,
                mol.G_rs,
                mol.omegaB_ar,
                mol.get_cpscf_ra(),
                xt_B_sb,
            )
            - 2
            * oe.contract(
                "cb,sd,ef,bs,de,fc",
                mol.A_bb,
                mol.G_sr,
                mol.J_as,
                mol.omegaA_bs,
                mol.get_cpscf_ra(),
                xt_B_sb,
            )
            - 2
            * oe.contract(
                "cb,sd,ef,bs,fc,de",
                mol.A_bb,
                mol.G_sr,
                mol.J_as,
                mol.omegaA_bs,
                mol.get_cpscf_sb(),
                xt_A_ra,
            )
            - 2
            * oe.contract(
                "cd,re,fa,ar,ec,df",
                mol.F_bs,
                mol.G_rs,
                mol.H_ba,
                mol.omegaB_ar,
                mol.get_cpscf_sb(),
                xt_B_sb,
            )
            - 2
            * oe.contract(
                "cd,re,fa,ar,df,ec",
                mol.F_bs,
                mol.G_rs,
                mol.H_ba,
                mol.omegaB_ar,
                mol.get_cpscf_sb(),
                xt_B_sb,
            )
            + 2
            * oe.contract(
                "cd,ea,rf,ar,fc,de",
                mol.E_ar,
                mol.B_aa,
                mol.C_rr,
                mol.omegaB_ar,
                mol.get_cpscf_ra(),
                xt_A_ra,
            )
            + 2
            * oe.contract(
                "cd,ea,rf,ar,de,fc",
                mol.E_ar,
                mol.B_aa,
                mol.C_rr,
                mol.omegaB_ar,
                mol.get_cpscf_ra(),
                xt_A_ra,
            )
            + 2
            * oe.contract(
                "cd,se,fb,bs,ec,df",
                mol.I_br,
                mol.D_ss,
                mol.H_ab,
                mol.omegaA_bs,
                mol.get_cpscf_sb(),
                xt_A_ra,
            )
            + 2
            * oe.contract(
                "cd,se,fb,bs,df,ec",
                mol.I_br,
                mol.D_ss,
                mol.H_ab,
                mol.omegaA_bs,
                mol.get_cpscf_ra(),
                xt_B_sb,
            )
            + 2
            * oe.contract(
                "cb,sd,ef,bs,de,fc",
                mol.A_bb,
                mol.D_ss,
                mol.F_bs,
                mol.omegaA_bs,
                mol.get_cpscf_sb(),
                xt_B_sb,
            )
            + 2
            * oe.contract(
                "cb,sd,ef,bs,fc,de",
                mol.A_bb,
                mol.D_ss,
                mol.F_bs,
                mol.omegaA_bs,
                mol.get_cpscf_sb(),
                xt_B_sb,
            )
            + 2
            * oe.contract(
                "rc,da,ef,ar,fd,ce",
                mol.C_rr,
                mol.H_ba,
                mol.J_as,
                mol.omegaB_ar,
                mol.get_cpscf_sb(),
                xt_A_ra,
            )
            + 2
            * oe.contract(
                "rc,da,ef,ar,ce,fd",
                mol.C_rr,
                mol.H_ba,
                mol.J_as,
                mol.omegaB_ar,
                mol.get_cpscf_ra(),
                xt_B_sb,
            )
            - 4
            * oe.contract(
                "cd,ea,rf,sb,fc,Qar,Qbs,de",
                mol.E_ar,
                mol.B_aa,
                mol.C_rr,
                mol.F_sb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 4
            * oe.contract(
                "cd,ea,rf,sb,de,Qar,Qbs,fc",
                mol.E_ar,
                mol.B_aa,
                mol.C_rr,
                mol.F_sb,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 4
            * oe.contract(
                "ra,cd,se,fb,ec,Qar,Qbs,df",
                mol.E_ra,
                mol.I_br,
                mol.D_ss,
                mol.H_ab,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 4
            * oe.contract(
                "ra,cd,se,fb,df,Qar,Qbs,ec",
                mol.E_ra,
                mol.I_br,
                mol.D_ss,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            - 4
            * oe.contract(
                "ra,cb,sd,ef,de,Qar,Qbs,fc",
                mol.E_ra,
                mol.A_bb,
                mol.D_ss,
                mol.F_bs,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            - 4
            * oe.contract(
                "ra,cb,sd,ef,fc,Qar,Qbs,de",
                mol.E_ra,
                mol.A_bb,
                mol.D_ss,
                mol.F_bs,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            - 4
            * oe.contract(
                "cb,sd,re,fa,ef,Qar,Qbs,dc",
                mol.A_bb,
                mol.D_ss,
                mol.G_rs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            - 4
            * oe.contract(
                "cb,sd,re,fa,dc,Qar,Qbs,ef",
                mol.A_bb,
                mol.D_ss,
                mol.G_rs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            - 4
            * oe.contract(
                "ca,rd,se,fb,ef,Qar,Qbs,dc",
                mol.B_aa,
                mol.C_rr,
                mol.G_sr,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 4
            * oe.contract(
                "ca,rd,se,fb,dc,Qar,Qbs,ef",
                mol.B_aa,
                mol.C_rr,
                mol.G_sr,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 4
            * oe.contract(
                "rc,sb,da,ef,fd,Qar,Qbs,ce",
                mol.C_rr,
                mol.F_sb,
                mol.H_ba,
                mol.J_as,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 4
            * oe.contract(
                "rc,sb,da,ef,ce,Qar,Qbs,fd",
                mol.C_rr,
                mol.F_sb,
                mol.H_ba,
                mol.J_as,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            - 2
            * oe.contract(
                "cd,rb,ea,sf,fc,Qar,Qbs,de",
                mol.E_ar,
                mol.I_rb,
                mol.B_aa,
                mol.G_sr,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 2
            * oe.contract(
                "cd,rb,ea,sf,de,Qar,Qbs,fc",
                mol.E_ar,
                mol.I_rb,
                mol.B_aa,
                mol.G_sr,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 2
            * oe.contract(
                "cd,re,fb,sa,ec,Qar,Qbs,df",
                mol.I_br,
                mol.G_rs,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 2
            * oe.contract(
                "cd,re,fb,sa,df,Qar,Qbs,ec",
                mol.I_br,
                mol.G_rs,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            - 2
            * oe.contract(
                "rb,sc,da,ef,fd,Qar,Qbs,ce",
                mol.I_rb,
                mol.G_sr,
                mol.H_ba,
                mol.J_as,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 2
            * oe.contract(
                "rb,sc,da,ef,ce,Qar,Qbs,fd",
                mol.I_rb,
                mol.G_sr,
                mol.H_ba,
                mol.J_as,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            - 2
            * oe.contract(
                "cb,da,se,rf,ed,Qar,Qbs,fc",
                mol.A_bb,
                mol.B_aa,
                mol.G_sr,
                mol.G_rs,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            - 2
            * oe.contract(
                "cb,de,rf,sa,fd,Qar,Qbs,ec",
                mol.A_bb,
                mol.F_bs,
                mol.G_rs,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            - 2
            * oe.contract(
                "cb,da,se,rf,fc,Qar,Qbs,ed",
                mol.A_bb,
                mol.B_aa,
                mol.G_sr,
                mol.G_rs,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 2
            * oe.contract(
                "cb,de,rf,sa,ec,Qar,Qbs,fd",
                mol.A_bb,
                mol.F_bs,
                mol.G_rs,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            - 2
            * oe.contract(
                "rc,sd,ea,fb,de,Qar,Qbs,cf",
                mol.C_rr,
                mol.D_ss,
                mol.H_ba,
                mol.H_ab,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            - 2
            * oe.contract(
                "rc,sd,ea,fb,cf,Qar,Qbs,de",
                mol.C_rr,
                mol.D_ss,
                mol.H_ba,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 2
            * oe.contract(
                "cd,re,fb,sa,ec,Qar,Qbs,df",
                mol.E_ar,
                mol.C_rr,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 2
            * oe.contract(
                "cd,re,fb,sa,df,Qar,Qbs,ec",
                mol.E_ar,
                mol.C_rr,
                mol.H_ab,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 2
            * oe.contract(
                "cd,rb,ea,sf,fc,Qar,Qbs,de",
                mol.I_br,
                mol.I_rb,
                mol.B_aa,
                mol.D_ss,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 2
            * oe.contract(
                "cd,rb,ea,sf,de,Qar,Qbs,fc",
                mol.I_br,
                mol.I_rb,
                mol.B_aa,
                mol.D_ss,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 2
            * oe.contract(
                "rb,sc,de,fa,ef,Qar,Qbs,cd",
                mol.I_rb,
                mol.D_ss,
                mol.F_bs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 2
            * oe.contract(
                "rb,sc,de,fa,cd,Qar,Qbs,ef",
                mol.I_rb,
                mol.D_ss,
                mol.F_bs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 2
            * oe.contract(
                "cb,rd,ef,sa,de,Qar,Qbs,fc",
                mol.A_bb,
                mol.C_rr,
                mol.J_as,
                mol.J_sa,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 2
            * oe.contract(
                "cb,sd,re,fa,df,Qar,Qbs,ec",
                mol.A_bb,
                mol.D_ss,
                mol.G_rs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 2
            * oe.contract(
                "cb,rd,ef,sa,fc,Qar,Qbs,de",
                mol.A_bb,
                mol.C_rr,
                mol.J_as,
                mol.J_sa,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 2
            * oe.contract(
                "cb,sd,re,fa,ec,Qar,Qbs,df",
                mol.A_bb,
                mol.D_ss,
                mol.G_rs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 2
            * oe.contract(
                "ca,rd,se,fb,df,Qar,Qbs,ec",
                mol.B_aa,
                mol.C_rr,
                mol.G_sr,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 2
            * oe.contract(
                "ca,rd,se,fb,ec,Qar,Qbs,df",
                mol.B_aa,
                mol.C_rr,
                mol.G_sr,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 4
            * oe.contract(
                "cd,ra,se,fb,ec,Qar,Qbs,df",
                mol.E_ar,
                mol.E_ra,
                mol.G_sr,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 4
            * oe.contract(
                "cd,ra,se,fb,df,Qar,Qbs,ec",
                mol.E_ar,
                mol.E_ra,
                mol.G_sr,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 4
            * oe.contract(
                "ra,cb,sd,ef,de,Qar,Qbs,fc",
                mol.E_ra,
                mol.A_bb,
                mol.G_sr,
                mol.J_as,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 4
            * oe.contract(
                "ra,cb,sd,ef,fc,Qar,Qbs,de",
                mol.E_ra,
                mol.A_bb,
                mol.G_sr,
                mol.J_as,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 4
            * oe.contract(
                "cd,ea,sb,rf,fc,Qar,Qbs,de",
                mol.I_br,
                mol.B_aa,
                mol.F_sb,
                mol.G_rs,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 4
            * oe.contract(
                "cd,ea,sb,rf,de,Qar,Qbs,fc",
                mol.I_br,
                mol.B_aa,
                mol.F_sb,
                mol.G_rs,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 4
            * oe.contract(
                "cb,da,re,sf,ed,Qar,Qbs,fc",
                mol.A_bb,
                mol.B_aa,
                mol.C_rr,
                mol.D_ss,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 4
            * oe.contract(
                "cb,da,re,sf,fc,Qar,Qbs,ed",
                mol.A_bb,
                mol.B_aa,
                mol.C_rr,
                mol.D_ss,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 4
            * oe.contract(
                "cd,sb,re,fa,ec,Qar,Qbs,df",
                mol.F_bs,
                mol.F_sb,
                mol.G_rs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 4
            * oe.contract(
                "cd,sb,re,fa,df,Qar,Qbs,ec",
                mol.F_bs,
                mol.F_sb,
                mol.G_rs,
                mol.H_ba,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
            + 4
            * oe.contract(
                "sc,rd,ea,fb,de,Qar,Qbs,cf",
                mol.G_sr,
                mol.G_rs,
                mol.H_ba,
                mol.H_ab,
                mol.get_cpscf_sb(),
                mol.Qar,
                mol.Qbs,
                xt_A_ra,
            )
            + 4
            * oe.contract(
                "sc,rd,ea,fb,cf,Qar,Qbs,de",
                mol.G_sr,
                mol.G_rs,
                mol.H_ba,
                mol.H_ab,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qbs,
                xt_B_sb,
            )
        ]
    )

    return x2_exch_ind_resp_sinf
