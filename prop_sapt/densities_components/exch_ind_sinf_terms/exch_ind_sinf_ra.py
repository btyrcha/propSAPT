import numpy as np
import opt_einsum as oe

from prop_sapt import Dimer


def get_exch_ind_density_ra(
    mol: Dimer, theta_sinf_t_ra: np.ndarray, theta_sinf_t_sb: np.ndarray
) -> np.ndarray:

    rho_MO_exch_ind_ra = 0.5 * (
        # < R(X) | V P2 R(V) >
        # vrx_ar terms
        oe.contract(
            "rs,ba,sb,aA->rA", mol.G_rs, mol.H_ba, mol.get_cpscf_sb(), mol.omegaB_aa
        )
        - oe.contract(
            "rs,ba,sb,Rr->Ra", mol.G_rs, mol.H_ba, mol.get_cpscf_sb(), mol.omegaB_rr
        )
        - oe.contract(
            "ca,rd,dc,aA->rA", mol.B_aa, mol.C_rr, mol.get_cpscf_ra(), mol.omegaB_aa
        )
        + oe.contract(
            "ca,rd,dc,Rr->Ra", mol.B_aa, mol.C_rr, mol.get_cpscf_ra(), mol.omegaB_rr
        )
        # vrx_bs terms
        - 2
        * oe.contract(
            "sr,ab,ra,QAR,Qbs->RA",
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cb,sd,dc,QAR,Qbs->RA",
            mol.A_bb,
            mol.D_ss,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        # vrx_abrs terms
        - 2
        * oe.contract(
            "ra,eb,sh,he,QRr,Qbs->Ra",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.get_cpscf_sb(),
            mol.Qrr,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ra,eb,sh,he,QaA,Qbs->rA",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.get_cpscf_sb(),
            mol.Qaa,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ca,rf,sb,fc,QRr,Qbs->Ra",
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.get_cpscf_ra(),
            mol.Qrr,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ca,rf,sb,fc,QaA,Qbs->rA",
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.get_cpscf_ra(),
            mol.Qaa,
            mol.Qbs,
        )
        - oe.contract(
            "rb,ea,sh,he,QRr,Qbs->Ra",
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qrr,
            mol.Qbs,
        )
        + oe.contract(
            "rb,ea,sh,he,QaA,Qbs->rA",
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qaa,
            mol.Qbs,
        )
        - oe.contract(
            "cb,rf,sa,fc,QRr,Qbs->Ra",
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qrr,
            mol.Qbs,
        )
        + oe.contract(
            "cb,rf,sa,fc,QaA,Qbs->rA",
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qaa,
            mol.Qbs,
        )
        + oe.contract(
            "rb,sf,ga,fg,QRr,Qbs->Ra",
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qrr,
            mol.Qbs,
        )
        - oe.contract(
            "rb,sf,ga,fg,QaA,Qbs->rA",
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qaa,
            mol.Qbs,
        )
        + oe.contract(
            "rd,eb,sa,de,QRr,Qbs->Ra",
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qrr,
            mol.Qbs,
        )
        - oe.contract(
            "rd,eb,sa,de,QaA,Qbs->rA",
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qaa,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ra,sf,gb,fg,QRr,Qbs->Ra",
            mol.E_ra,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qrr,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ra,sf,gb,fg,QaA,Qbs->rA",
            mol.E_ra,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qaa,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "sb,rf,ga,fg,QRr,Qbs->Ra",
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qrr,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "sb,rf,ga,fg,QaA,Qbs->rA",
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qaa,
            mol.Qbs,
        )
        # product-like terms
        - oe.contract(
            "ca,rd,ef,ar,de->cf",
            mol.E_ra,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.get_cpscf_sb(),
        )
        - oe.contract(
            "rc,de,fa,ar,ef->dc",
            mol.E_ra,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.get_cpscf_sb(),
        )
        - oe.contract(
            "cb,de,sf,bs,fd->ce",
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.omegaA_bs,
            mol.get_cpscf_ra(),
        )
        - oe.contract(
            "cb,de,sf,bs,ec->df",
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.omegaA_bs,
            mol.get_cpscf_sb(),
        )
        + oe.contract(
            "ca,de,rf,ar,fd->ce",
            mol.E_ra,
            mol.B_aa,
            mol.C_rr,
            mol.omegaB_ar,
            mol.get_cpscf_ra(),
        )
        + oe.contract(
            "rc,da,ef,ar,fd->ec",
            mol.E_ra,
            mol.B_aa,
            mol.C_rr,
            mol.omegaB_ar,
            mol.get_cpscf_ra(),
        )
        + oe.contract(
            "cb,sd,ef,bs,de->cf",
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.omegaA_bs,
            mol.get_cpscf_sb(),
        )
        + oe.contract(
            "cd,eb,sf,bs,de->cf",
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            mol.omegaA_bs,
            mol.get_cpscf_ra(),
        )
        - 2
        * oe.contract(
            "ra,cb,sd,ef,de,Qar,Qbs->cf",
            mol.E_ra,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ra,cd,eb,sf,de,Qar,Qbs->cf",
            mol.E_ra,
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ca,rd,eb,sf,fe,Qar,Qbs->cd",
            mol.E_ra,
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ca,de,rf,sb,fd,Qar,Qbs->ce",
            mol.E_ra,
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "rc,da,ef,sb,fd,Qar,Qbs->ec",
            mol.E_ra,
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "cb,da,re,sf,ed,Qar,Qbs->cf",
            mol.I_rb,
            mol.B_aa,
            mol.C_rr,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "ca,rb,de,sf,fd,Qar,Qbs->ce",
            mol.E_ra,
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "ca,db,re,sf,ed,Qar,Qbs->cf",
            mol.E_ra,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rc,db,ea,sf,fe,Qar,Qbs->dc",
            mol.E_ra,
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rc,db,ef,sa,fd,Qar,Qbs->ec",
            mol.E_ra,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rb,cd,ea,sf,de,Qar,Qbs->cf",
            mol.I_rb,
            mol.G_rs,
            mol.H_ba,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cb,rd,ef,sa,de,Qar,Qbs->cf",
            mol.I_rb,
            mol.G_rs,
            mol.H_ba,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "ca,rb,sd,ef,de,Qar,Qbs->cf",
            mol.E_ra,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "ca,rd,eb,sf,de,Qar,Qbs->cf",
            mol.E_ra,
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "rc,db,se,fa,ef,Qar,Qbs->dc",
            mol.E_ra,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "rc,de,fb,sa,ef,Qar,Qbs->dc",
            mol.E_ra,
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "rb,ca,de,sf,ec,Qar,Qbs->df",
            mol.I_rb,
            mol.B_aa,
            mol.C_rr,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "cb,de,rf,sa,fd,Qar,Qbs->ce",
            mol.I_rb,
            mol.B_aa,
            mol.C_rr,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ra,cb,de,sf,fd,Qar,Qbs->ce",
            mol.E_ra,
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ra,cb,de,sf,ec,Qar,Qbs->df",
            mol.E_ra,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ca,rd,se,fb,ef,Qar,Qbs->cd",
            mol.E_ra,
            mol.E_ra,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ca,sb,rd,ef,de,Qar,Qbs->cf",
            mol.E_ra,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "rc,sb,de,fa,ef,Qar,Qbs->dc",
            mol.E_ra,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cb,rd,ea,sf,de,Qar,Qbs->cf",
            mol.I_rb,
            mol.G_rs,
            mol.H_ba,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        # < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
        + oe.contract("rR,Ra->ra", mol.omegaB_rr, theta_sinf_t_ra)
        - oe.contract("Aa,rA->ra", mol.omegaB_aa, theta_sinf_t_ra)
        + 2 * oe.contract("Qar,Qbs,sb->ra", mol.Qar, mol.Qbs, theta_sinf_t_sb)
        # < V P2 R(X) R(V) >
        - oe.contract(
            "cd,se,fb,bs,ec->df",
            mol.E_ar,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.get_cpscf_ra(),
        )
        - oe.contract(
            "cd,se,fb,bs,df->ec",
            mol.E_ar,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.get_cpscf_ra(),
        )
        - oe.contract(
            "cd,ea,rf,ar,fc->de",
            mol.I_br,
            mol.B_aa,
            mol.G_rs,
            mol.omegaB_ar,
            mol.get_cpscf_sb(),
        )
        - oe.contract(
            "cb,sd,ef,bs,fc->de",
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.omegaA_bs,
            mol.get_cpscf_sb(),
        )
        + oe.contract(
            "cd,ea,rf,ar,fc->de",
            mol.E_ar,
            mol.B_aa,
            mol.C_rr,
            mol.omegaB_ar,
            mol.get_cpscf_ra(),
        )
        + oe.contract(
            "cd,ea,rf,ar,de->fc",
            mol.E_ar,
            mol.B_aa,
            mol.C_rr,
            mol.omegaB_ar,
            mol.get_cpscf_ra(),
        )
        + oe.contract(
            "cd,se,fb,bs,ec->df",
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.omegaA_bs,
            mol.get_cpscf_sb(),
        )
        + oe.contract(
            "rc,da,ef,ar,fd->ce",
            mol.C_rr,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.get_cpscf_sb(),
        )
        - 2
        * oe.contract(
            "cd,ea,rf,sb,fc,Qar,Qbs->de",
            mol.E_ar,
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "cd,ea,rf,sb,de,Qar,Qbs->fc",
            mol.E_ar,
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ra,cd,se,fb,ec,Qar,Qbs->df",
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ca,rd,se,fb,ef,Qar,Qbs->dc",
            mol.B_aa,
            mol.C_rr,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ca,rd,se,fb,dc,Qar,Qbs->ef",
            mol.B_aa,
            mol.C_rr,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "rc,sb,da,ef,fd,Qar,Qbs->ce",
            mol.C_rr,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cd,rb,ea,sf,fc,Qar,Qbs->de",
            mol.E_ar,
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cd,rb,ea,sf,de,Qar,Qbs->fc",
            mol.E_ar,
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cd,re,fb,sa,ec,Qar,Qbs->df",
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rb,sc,da,ef,fd,Qar,Qbs->ce",
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cb,da,se,rf,fc,Qar,Qbs->ed",
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rc,sd,ea,fb,de,Qar,Qbs->cf",
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "cd,re,fb,sa,ec,Qar,Qbs->df",
            mol.E_ar,
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "cd,re,fb,sa,df,Qar,Qbs->ec",
            mol.E_ar,
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "cd,rb,ea,sf,fc,Qar,Qbs->de",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "cb,rd,ef,sa,fc,Qar,Qbs->de",
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "ca,rd,se,fb,df,Qar,Qbs->ec",
            mol.B_aa,
            mol.C_rr,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "ca,rd,se,fb,ec,Qar,Qbs->df",
            mol.B_aa,
            mol.C_rr,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cd,ra,se,fb,ec,Qar,Qbs->df",
            mol.E_ar,
            mol.E_ra,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cd,ra,se,fb,df,Qar,Qbs->ec",
            mol.E_ar,
            mol.E_ra,
            mol.G_sr,
            mol.H_ab,
            mol.get_cpscf_ra(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ra,cb,sd,ef,fc,Qar,Qbs->de",
            mol.E_ra,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cd,ea,sb,rf,fc,Qar,Qbs->de",
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.G_rs,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cb,da,re,sf,fc,Qar,Qbs->ed",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "sc,rd,ea,fb,de,Qar,Qbs->cf",
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.get_cpscf_sb(),
            mol.Qar,
            mol.Qbs,
        )
    )

    rho_MO_exch_ind_ra = mol.cpscf(
        "A", perturbation=rho_MO_exch_ind_ra.T
    ) + oe.contract("rsab,bs->ra", mol.t_rsab, theta_sinf_t_sb.T)

    return rho_MO_exch_ind_ra
