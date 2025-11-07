import numpy as np
import opt_einsum as oe

from prop_sapt import Dimer


def get_exch_ind_density_sb(
    mol: Dimer, theta_sinf_t_ra: np.ndarray, theta_sinf_t_sb: np.ndarray
) -> np.ndarray:

    rho_MO_exch_ind_sb = 0.5 * (
        # < R(X) | V P2 R(V) >
        # vrx_ar terms
        -2
        * oe.contract(
            "rs,ba,sb,Qar,QBS->SB",
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ca,rd,dc,Qar,QBS->SB",
            mol.B_aa,
            mol.C_rr,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        # vrx_bs terms
        + oe.contract(
            "sr,ab,ra,bB->sB", mol.G_sr, mol.H_ab, mol.get_cpscf_ra(), mol.omegaA_bb
        )
        - oe.contract(
            "sr,ab,ra,Ss->Sb", mol.G_sr, mol.H_ab, mol.get_cpscf_ra(), mol.omegaA_ss
        )
        - oe.contract(
            "cb,sd,dc,bB->sB", mol.A_bb, mol.D_ss, mol.get_cpscf_sb(), mol.omegaA_bb
        )
        + oe.contract(
            "cb,sd,dc,Ss->Sb", mol.A_bb, mol.D_ss, mol.get_cpscf_sb(), mol.omegaA_ss
        )
        # vrx_abrs terms
        - 2
        * oe.contract(
            "ra,eb,sh,he,Qar,QSs->Sb",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qss,
        )
        + 2
        * oe.contract(
            "ra,eb,sh,he,Qar,QbB->sB",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbb,
        )
        - 2
        * oe.contract(
            "ca,rf,sb,fc,Qar,QSs->Sb",
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qss,
        )
        + 2
        * oe.contract(
            "ca,rf,sb,fc,Qar,QbB->sB",
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbb,
        )
        - oe.contract(
            "rb,ea,sh,he,Qar,QSs->Sb",
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qss,
        )
        + oe.contract(
            "rb,ea,sh,he,Qar,QbB->sB",
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbb,
        )
        - oe.contract(
            "cb,rf,sa,fc,Qar,QSs->Sb",
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qss,
        )
        + oe.contract(
            "cb,rf,sa,fc,Qar,QbB->sB",
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbb,
        )
        + oe.contract(
            "rb,sf,ga,fg,Qar,QSs->Sb",
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qss,
        )
        - oe.contract(
            "rb,sf,ga,fg,Qar,QbB->sB",
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbb,
        )
        + oe.contract(
            "rd,eb,sa,de,Qar,QSs->Sb",
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qss,
        )
        - oe.contract(
            "rd,eb,sa,de,Qar,QbB->sB",
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbb,
        )
        + 2
        * oe.contract(
            "ra,sf,gb,fg,Qar,QSs->Sb",
            mol.E_ra,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qss,
        )
        - 2
        * oe.contract(
            "ra,sf,gb,fg,Qar,QbB->sB",
            mol.E_ra,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbb,
        )
        + 2
        * oe.contract(
            "sb,rf,ga,fg,Qar,QSs->Sb",
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qss,
        )
        - 2
        * oe.contract(
            "sb,rf,ga,fg,Qar,QbB->sB",
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbb,
        )
        # product-like terms
        - oe.contract(
            "rc,da,ef,ar,fd->ec",
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.omegaB_ar,
            mol.get_cpscf_ra(),
        )
        - oe.contract(
            "cd,re,fa,ar,ec->fd",
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.omegaB_ar,
            mol.get_cpscf_sb(),
        )
        - oe.contract(
            "cb,sd,ef,bs,de->cf",
            mol.F_sb,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.get_cpscf_ra(),
        )
        - oe.contract(
            "sc,de,fb,bs,ef->dc",
            mol.F_sb,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.get_cpscf_ra(),
        )
        + oe.contract(
            "rc,de,fa,ar,ef->dc",
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.omegaB_ar,
            mol.get_cpscf_sb(),
        )
        + oe.contract(
            "cb,de,sf,bs,ec->df",
            mol.A_bb,
            mol.D_ss,
            mol.F_sb,
            mol.omegaA_bs,
            mol.get_cpscf_sb(),
        )
        + oe.contract(
            "cd,se,fb,bs,ec->fd",
            mol.A_bb,
            mol.D_ss,
            mol.F_sb,
            mol.omegaA_bs,
            mol.get_cpscf_sb(),
        )
        + oe.contract(
            "rc,de,fa,ar,cd->fe",
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            mol.omegaB_ar,
            mol.get_cpscf_ra(),
        )
        - 2
        * oe.contract(
            "ra,cb,de,sf,ec,Qar,Qbs->df",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.F_sb,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ra,cd,se,fb,ec,Qar,Qbs->fd",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.F_sb,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "rc,db,se,fa,ed,Qar,Qbs->fc",
            mol.I_rb,
            mol.A_bb,
            mol.D_ss,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "rc,de,sb,fa,ef,Qar,Qbs->dc",
            mol.I_rb,
            mol.D_ss,
            mol.F_sb,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ca,rd,eb,sf,dc,Qar,Qbs->ef",
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.F_sb,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "rc,sb,de,fa,cd,Qar,Qbs->fe",
            mol.C_rr,
            mol.F_sb,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rb,ca,sd,ef,fc,Qar,Qbs->ed",
            mol.I_rb,
            mol.B_aa,
            mol.F_sb,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rb,sc,de,fa,cd,Qar,Qbs->fe",
            mol.I_rb,
            mol.G_sr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rc,da,eb,sf,fd,Qar,Qbs->ec",
            mol.I_rb,
            mol.B_aa,
            mol.F_sb,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rc,de,fb,sa,ef,Qar,Qbs->dc",
            mol.I_rb,
            mol.G_sr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cb,sd,re,fa,ec,Qar,Qbs->fd",
            mol.A_bb,
            mol.F_sb,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cd,eb,rf,sa,fc,Qar,Qbs->ed",
            mol.A_bb,
            mol.F_sb,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "rb,cd,se,fa,ec,Qar,Qbs->fd",
            mol.I_rb,
            mol.A_bb,
            mol.D_ss,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "rb,cd,se,fa,df,Qar,Qbs->ce",
            mol.I_rb,
            mol.D_ss,
            mol.F_sb,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "rc,db,ef,sa,fd,Qar,Qbs->ec",
            mol.I_rb,
            mol.A_bb,
            mol.D_ss,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "rc,sd,eb,fa,df,Qar,Qbs->ec",
            mol.I_rb,
            mol.D_ss,
            mol.F_sb,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "rc,db,ef,sa,ce,Qar,Qbs->df",
            mol.C_rr,
            mol.F_sb,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "rc,sd,eb,fa,ce,Qar,Qbs->fd",
            mol.C_rr,
            mol.F_sb,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ra,cb,sd,ef,de,Qar,Qbs->cf",
            mol.E_ra,
            mol.F_sb,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ra,sc,de,fb,ef,Qar,Qbs->dc",
            mol.E_ra,
            mol.F_sb,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "rc,da,sb,ef,fd,Qar,Qbs->ec",
            mol.I_rb,
            mol.B_aa,
            mol.F_sb,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "rc,sd,eb,fa,de,Qar,Qbs->fc",
            mol.I_rb,
            mol.G_sr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cd,sb,re,fa,ec,Qar,Qbs->fd",
            mol.A_bb,
            mol.F_sb,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cb,sd,re,fa,ef,Qar,Qbs->cd",
            mol.F_sb,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        # < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
        + oe.contract("sS,Sb->sb", mol.omegaA_ss, theta_sinf_t_sb)
        - oe.contract("Bb,sB->sb", mol.omegaA_bb, theta_sinf_t_sb)
        + 2 * oe.contract("Qar,Qbs,ra->sb", mol.Qar, mol.Qbs, theta_sinf_t_ra)
        # < V P2 R(X) R(V) >
        - oe.contract(
            "cd,ea,rf,ar,de->fc",
            mol.I_br,
            mol.B_aa,
            mol.G_rs,
            mol.omegaB_ar,
            mol.get_cpscf_ra(),
        )
        - oe.contract(
            "cb,sd,ef,bs,de->fc",
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.omegaA_bs,
            mol.get_cpscf_ra(),
        )
        - oe.contract(
            "cd,re,fa,ar,ec->df",
            mol.F_bs,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.get_cpscf_sb(),
        )
        - oe.contract(
            "cd,re,fa,ar,df->ec",
            mol.F_bs,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.get_cpscf_sb(),
        )
        + oe.contract(
            "cd,se,fb,bs,df->ec",
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.omegaA_bs,
            mol.get_cpscf_ra(),
        )
        + oe.contract(
            "cb,sd,ef,bs,de->fc",
            mol.A_bb,
            mol.D_ss,
            mol.F_bs,
            mol.omegaA_bs,
            mol.get_cpscf_sb(),
        )
        + oe.contract(
            "cb,sd,ef,bs,fc->de",
            mol.A_bb,
            mol.D_ss,
            mol.F_bs,
            mol.omegaA_bs,
            mol.get_cpscf_sb(),
        )
        + oe.contract(
            "rc,da,ef,ar,ce->fd",
            mol.C_rr,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.get_cpscf_ra(),
        )
        - 2
        * oe.contract(
            "ra,cd,se,fb,df,Qar,Qbs->ec",
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ra,cb,sd,ef,de,Qar,Qbs->fc",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.F_bs,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ra,cb,sd,ef,fc,Qar,Qbs->de",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.F_bs,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "cb,sd,re,fa,ef,Qar,Qbs->dc",
            mol.A_bb,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "cb,sd,re,fa,dc,Qar,Qbs->ef",
            mol.A_bb,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "rc,sb,da,ef,ce,Qar,Qbs->fd",
            mol.C_rr,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cd,re,fb,sa,df,Qar,Qbs->ec",
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rb,sc,da,ef,ce,Qar,Qbs->fd",
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cb,da,se,rf,ed,Qar,Qbs->fc",
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cb,de,rf,sa,fd,Qar,Qbs->ec",
            mol.A_bb,
            mol.F_bs,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cb,de,rf,sa,ec,Qar,Qbs->fd",
            mol.A_bb,
            mol.F_bs,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rc,sd,ea,fb,cf,Qar,Qbs->de",
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "cd,rb,ea,sf,de,Qar,Qbs->fc",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "rb,sc,de,fa,ef,Qar,Qbs->cd",
            mol.I_rb,
            mol.D_ss,
            mol.F_bs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "rb,sc,de,fa,cd,Qar,Qbs->ef",
            mol.I_rb,
            mol.D_ss,
            mol.F_bs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "cb,rd,ef,sa,de,Qar,Qbs->fc",
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "cb,sd,re,fa,df,Qar,Qbs->ec",
            mol.A_bb,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "cb,sd,re,fa,ec,Qar,Qbs->df",
            mol.A_bb,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ra,cb,sd,ef,de,Qar,Qbs->fc",
            mol.E_ra,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cd,ea,sb,rf,de,Qar,Qbs->fc",
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.G_rs,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cb,da,re,sf,ed,Qar,Qbs->fc",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cd,sb,re,fa,ec,Qar,Qbs->df",
            mol.F_bs,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cd,sb,re,fa,df,Qar,Qbs->ec",
            mol.F_bs,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "sc,rd,ea,fb,cf,Qar,Qbs->de",
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
    )

    rho_MO_exch_ind_sb = mol.cpscf(
        "B", perturbation=rho_MO_exch_ind_sb.T
    ) + oe.contract("rsab,ar->sb", mol.t_rsab, theta_sinf_t_ra.T)

    return rho_MO_exch_ind_sb
