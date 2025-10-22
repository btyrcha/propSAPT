import numpy as np
import opt_einsum as oe

from prop_sapt.molecule import Dimer


def get_exch_disp2_sinf_property_term3(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
):

    # < V P R(X) R(V) >
    x2_exch_disp_sinf_term3 = (
        -4
        * oe.contract(
            "cd,cd,gb,sh,bs,dhcg,dc",
            mol.E_ar,
            mol.E_ar,
            mol.A_bb,
            mol.D_ss,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,ca,rd,gh,ar,dhcg,dc",
            mol.E_ar,
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,ca,rd,gh,ar,dhcg,dc",
            mol.E_ar,
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,ed,sg,cb,bs,dgce,dc",
            mol.E_ar,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,eb,sf,ef,bs,dfce,fe",
            mol.E_ar,
            mol.A_bb,
            mol.D_ss,
            mol.F_bs,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,eb,sf,ef,bs,dfce,fe",
            mol.E_ar,
            mol.A_bb,
            mol.D_ss,
            mol.F_bs,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,rd,fa,ch,ar,dhcf,dc",
            mol.E_ar,
            mol.C_rr,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,ea,rd,eh,ar,dhec,de",
            mol.I_br,
            mol.B_aa,
            mol.C_rr,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,cb,sf,gf,bs,dfgc,fc",
            mol.I_br,
            mol.A_bb,
            mol.D_ss,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,se,ce,hb,bs,dehc,ec",
            mol.I_br,
            mol.D_ss,
            mol.F_bs,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "ca,rd,ef,ef,ar,dfce,fe",
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.F_bs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "rc,de,da,ge,ar,cegd,ed",
            mol.C_rr,
            mol.F_bs,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cd,ed,ca,rh,ar,dhce,dc",
            mol.E_ar,
            mol.I_br,
            mol.B_aa,
            mol.G_rs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,eb,sd,ch,bs,dhce,dc",
            mol.E_ar,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,ed,ca,rh,ar,dhce,dc",
            mol.E_ar,
            mol.I_br,
            mol.B_aa,
            mol.G_rs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,eb,sd,ch,bs,dhce,dc",
            mol.E_ar,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,sd,fb,fh,bs,dhfc,df",
            mol.I_br,
            mol.G_sr,
            mol.H_ab,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,ea,cg,rg,ar,dgec,gc",
            mol.I_br,
            mol.B_aa,
            mol.F_bs,
            mol.G_rs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cd,re,ca,ge,ar,degc,ec",
            mol.I_br,
            mol.G_rs,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cd,ea,cg,rg,ar,dgec,gc",
            mol.I_br,
            mol.B_aa,
            mol.F_bs,
            mol.G_rs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cd,sd,fb,fh,bs,dhfc,df",
            mol.I_br,
            mol.G_sr,
            mol.H_ab,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,re,ca,ge,ar,degc,ec",
            mol.I_br,
            mol.G_rs,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cb,ce,sf,ge,bs,fegc,ec",
            mol.A_bb,
            mol.F_bs,
            mol.G_sr,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cb,ce,sf,ge,bs,fegc,ec",
            mol.A_bb,
            mol.F_bs,
            mol.G_sr,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,ed,sg,cb,bs,dgce,dc",
            mol.E_ar,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,rd,fa,ch,ar,dhcf,dc",
            mol.E_ar,
            mol.C_rr,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,ed,sg,cb,bs,dgce,dc",
            mol.E_ar,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,rd,fa,ch,ar,dhcf,dc",
            mol.E_ar,
            mol.C_rr,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,ea,rd,eh,ar,dhec,de",
            mol.I_br,
            mol.B_aa,
            mol.C_rr,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,cb,sf,gf,bs,dfgc,fc",
            mol.I_br,
            mol.A_bb,
            mol.D_ss,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,se,ce,hb,bs,dehc,ec",
            mol.I_br,
            mol.D_ss,
            mol.F_bs,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,cb,sf,gf,bs,dfgc,fc",
            mol.I_br,
            mol.A_bb,
            mol.D_ss,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,ea,rd,eh,ar,dhec,de",
            mol.I_br,
            mol.B_aa,
            mol.C_rr,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,se,ce,hb,bs,dehc,ec",
            mol.I_br,
            mol.D_ss,
            mol.F_bs,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "rc,de,da,ge,ar,cegd,ed",
            mol.C_rr,
            mol.F_bs,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "rc,de,da,ge,ar,cegd,ed",
            mol.C_rr,
            mol.F_bs,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,cd,rg,ha,ar,dgch,dc",
            mol.E_ar,
            mol.E_ar,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ef,sd,cb,bs,dfce,dc",
            mol.E_ar,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ef,sd,cb,bs,dfce,dc",
            mol.E_ar,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ed,ca,rh,ar,dhce,dc",
            mol.E_ar,
            mol.I_br,
            mol.B_aa,
            mol.G_rs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,eb,sd,ch,bs,dhce,dc",
            mol.E_ar,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ef,rf,ea,ar,dfce,fe",
            mol.E_ar,
            mol.F_bs,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,ef,rf,ea,ar,dfce,fe",
            mol.E_ar,
            mol.F_bs,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,sd,fb,fh,bs,dhfc,df",
            mol.I_br,
            mol.G_sr,
            mol.H_ab,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ea,cg,rg,ar,dgec,gc",
            mol.I_br,
            mol.B_aa,
            mol.F_bs,
            mol.G_rs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,re,ca,ge,ar,degc,ec",
            mol.I_br,
            mol.G_rs,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cb,ce,sf,ge,bs,fegc,ec",
            mol.A_bb,
            mol.F_bs,
            mol.G_sr,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,cd,sg,hb,bs,gdhc,dc",
            mol.F_bs,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_sb,
        )
        - 8
        * oe.contract(
            "cd,cd,sb,rg,ha,dgch,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ar,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 8
        * oe.contract(
            "cd,ra,ef,sd,cb,dfce,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ra,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 8
        * oe.contract(
            "cd,ra,ef,sd,cb,dfce,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ra,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 8
        * oe.contract(
            "cd,ra,eb,sd,ch,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 8
        * oe.contract(
            "cd,ed,ca,sb,rh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 8
        * oe.contract(
            "cd,eb,ca,rd,sh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 8
        * oe.contract(
            "cd,ef,sb,rf,ea,dfce,Qar,Qbs,fe",
            mol.E_ar,
            mol.F_bs,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 8
        * oe.contract(
            "cd,ef,sb,rf,ea,dfce,Qar,Qbs,fe",
            mol.E_ar,
            mol.F_bs,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 8
        * oe.contract(
            "cd,sd,rf,ga,cb,dfcg,Qar,Qbs,dc",
            mol.E_ar,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 8
        * oe.contract(
            "ra,cd,sd,fb,fh,dhfc,Qar,Qbs,df",
            mol.E_ra,
            mol.I_br,
            mol.G_sr,
            mol.H_ab,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 8
        * oe.contract(
            "ra,cb,ce,sf,ge,fegc,Qar,Qbs,ec",
            mol.E_ra,
            mol.A_bb,
            mol.F_bs,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 8
        * oe.contract(
            "ra,cd,cd,sg,hb,gdhc,Qar,Qbs,dc",
            mol.E_ra,
            mol.F_bs,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 8
        * oe.contract(
            "cd,ea,cg,sb,rg,dgec,Qar,Qbs,gc",
            mol.I_br,
            mol.B_aa,
            mol.F_bs,
            mol.F_sb,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 8
        * oe.contract(
            "cd,sb,re,ca,ge,degc,Qar,Qbs,ec",
            mol.I_br,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 8
        * oe.contract(
            "cb,da,re,sf,cf,efdc,Qar,Qbs,fc",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.F_bs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 8
        * oe.contract(
            "cd,se,rd,ca,hb,edhc,Qar,Qbs,dc",
            mol.F_bs,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,cd,rb,sg,ha,dgch,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,ra,ed,sg,cb,dgce,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,rd,fg,cb,sa,dgcf,Qar,Qbs,dc",
            mol.E_ar,
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,rd,sb,fa,ch,dhcf,Qar,Qbs,dc",
            mol.E_ar,
            mol.C_rr,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,ra,ed,sg,cb,dgce,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,rd,fg,cb,sa,dgcf,Qar,Qbs,dc",
            mol.E_ar,
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,rd,sb,fa,ch,dhcf,Qar,Qbs,dc",
            mol.E_ar,
            mol.C_rr,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,ed,rb,ca,sh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,rb,se,fe,fa,decf,Qar,Qbs,ef",
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.F_bs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,rb,se,fe,fa,decf,Qar,Qbs,ef",
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.F_bs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,eb,sf,rf,ea,dfce,Qar,Qbs,fe",
            mol.E_ar,
            mol.A_bb,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,eb,rd,ch,sa,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,eb,sf,rf,ea,dfce,Qar,Qbs,fe",
            mol.E_ar,
            mol.A_bb,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "ra,cd,cb,sf,gf,dfgc,Qar,Qbs,fc",
            mol.E_ra,
            mol.I_br,
            mol.A_bb,
            mol.D_ss,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "ra,cd,se,ce,hb,dehc,Qar,Qbs,ec",
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.F_bs,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "ra,cd,cb,sf,gf,dfgc,Qar,Qbs,fc",
            mol.E_ra,
            mol.I_br,
            mol.A_bb,
            mol.D_ss,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "ra,cd,se,ce,hb,dehc,Qar,Qbs,ec",
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.F_bs,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,ea,rd,sb,eh,dhec,Qar,Qbs,de",
            mol.I_br,
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,rd,fb,fh,sa,dhfc,Qar,Qbs,df",
            mol.I_br,
            mol.C_rr,
            mol.H_ab,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,rb,ea,sf,cf,dfec,Qar,Qbs,fc",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.F_bs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,rb,se,ca,ge,degc,Qar,Qbs,ec",
            mol.I_br,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,cb,fa,sg,rg,dgfc,Qar,Qbs,gc",
            mol.I_br,
            mol.A_bb,
            mol.B_aa,
            mol.D_ss,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,se,re,ca,hb,dehc,Qar,Qbs,ec",
            mol.I_br,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,cb,fa,sg,rg,dgfc,Qar,Qbs,gc",
            mol.I_br,
            mol.A_bb,
            mol.B_aa,
            mol.D_ss,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cd,ea,rd,sb,eh,dhec,Qar,Qbs,de",
            mol.I_br,
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cd,se,re,ca,hb,dehc,Qar,Qbs,ec",
            mol.I_br,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cb,rd,cf,gf,sa,dfgc,Qar,Qbs,fc",
            mol.A_bb,
            mol.C_rr,
            mol.F_bs,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "cb,da,re,se,dh,ehdc,Qar,Qbs,ed",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "cb,da,re,se,dh,ehdc,Qar,Qbs,ed",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "ca,rd,ef,sd,cb,dfce,Qar,Qbs,dc",
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "ca,rd,ef,sd,cb,dfce,Qar,Qbs,dc",
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "rc,sc,ea,fb,fh,chfe,Qar,Qbs,cf",
            mol.C_rr,
            mol.G_sr,
            mol.H_ba,
            mol.H_ab,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 4
        * oe.contract(
            "rc,de,de,hb,sa,cehd,Qar,Qbs,ed",
            mol.C_rr,
            mol.F_bs,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "rc,de,sb,da,ge,cegd,Qar,Qbs,ed",
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "rc,de,sb,da,ge,cegd,Qar,Qbs,ed",
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 4
        * oe.contract(
            "rc,sc,ea,fb,fh,chfe,Qar,Qbs,cf",
            mol.C_rr,
            mol.G_sr,
            mol.H_ba,
            mol.H_ab,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,ed,rg,cb,sa,dgce,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,rb,sd,fa,ch,dhcf,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,eb,ca,sd,rh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,rd,sf,ga,cb,dfcg,Qar,Qbs,dc",
            mol.E_ar,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,ed,rg,cb,sa,dgce,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,rb,sd,fa,ch,dhcf,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,eb,ca,sd,rh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,rd,sf,ga,cb,dfcg,Qar,Qbs,dc",
            mol.E_ar,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,rb,ea,sd,eh,dhec,Qar,Qbs,de",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,ea,rd,sg,eb,dgec,Qar,Qbs,de",
            mol.I_br,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,ea,sd,rg,eb,dgec,Qar,Qbs,de",
            mol.I_br,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,cb,rf,gf,sa,dfgc,Qar,Qbs,fc",
            mol.I_br,
            mol.A_bb,
            mol.G_rs,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cd,cf,rf,hb,sa,dfhc,Qar,Qbs,fc",
            mol.I_br,
            mol.F_bs,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cd,rb,ea,sd,eh,dhec,Qar,Qbs,de",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,cb,rf,gf,sa,dfgc,Qar,Qbs,fc",
            mol.I_br,
            mol.A_bb,
            mol.G_rs,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cd,ea,sd,rg,eb,dgec,Qar,Qbs,de",
            mol.I_br,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,ea,rd,sg,eb,dgec,Qar,Qbs,de",
            mol.I_br,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        - 2
        * oe.contract(
            "cd,cf,rf,hb,sa,dfhc,Qar,Qbs,fc",
            mol.I_br,
            mol.F_bs,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "rb,cd,se,ca,gd,edgc,Qar,Qbs,dc",
            mol.I_rb,
            mol.F_bs,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "rb,cd,se,ca,gd,edgc,Qar,Qbs,dc",
            mol.I_rb,
            mol.F_bs,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cb,da,cf,sg,rf,gfdc,Qar,Qbs,fc",
            mol.A_bb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cb,rd,se,ca,ge,degc,Qar,Qbs,ec",
            mol.A_bb,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cb,sd,re,ca,ge,degc,Qar,Qbs,ec",
            mol.A_bb,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cb,da,cf,sg,rf,gfdc,Qar,Qbs,fc",
            mol.A_bb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cb,rd,se,ca,ge,degc,Qar,Qbs,ec",
            mol.A_bb,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "cb,sd,re,ca,ge,degc,Qar,Qbs,ec",
            mol.A_bb,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "rc,sd,ed,ea,hb,cdhe,Qar,Qbs,de",
            mol.C_rr,
            mol.D_ss,
            mol.F_bs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        - 2
        * oe.contract(
            "rc,sd,ed,ea,hb,cdhe,Qar,Qbs,de",
            mol.C_rr,
            mol.D_ss,
            mol.F_bs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,ed,rb,ca,sh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,eb,rd,ch,sa,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,ed,rb,ca,sh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,eb,rd,ch,sa,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,rd,fb,fh,sa,dhfc,Qar,Qbs,df",
            mol.I_br,
            mol.C_rr,
            mol.H_ab,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,rb,ea,sf,cf,dfec,Qar,Qbs,fc",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.F_bs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,rb,se,ca,ge,degc,Qar,Qbs,ec",
            mol.I_br,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,cb,fa,sg,rg,dgfc,Qar,Qbs,gc",
            mol.I_br,
            mol.A_bb,
            mol.B_aa,
            mol.D_ss,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,se,re,ca,hb,dehc,Qar,Qbs,ec",
            mol.I_br,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,rb,ea,sf,cf,dfec,Qar,Qbs,fc",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.F_bs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,rb,se,ca,ge,degc,Qar,Qbs,ec",
            mol.I_br,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,cb,fa,sg,rg,dgfc,Qar,Qbs,gc",
            mol.I_br,
            mol.A_bb,
            mol.B_aa,
            mol.D_ss,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cd,rd,fb,fh,sa,dhfc,Qar,Qbs,df",
            mol.I_br,
            mol.C_rr,
            mol.H_ab,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cd,se,re,ca,hb,dehc,Qar,Qbs,ec",
            mol.I_br,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cb,rd,cf,gf,sa,dfgc,Qar,Qbs,fc",
            mol.A_bb,
            mol.C_rr,
            mol.F_bs,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "cb,da,re,se,dh,ehdc,Qar,Qbs,ed",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cb,da,re,se,dh,ehdc,Qar,Qbs,ed",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "cb,rd,cf,gf,sa,dfgc,Qar,Qbs,fc",
            mol.A_bb,
            mol.C_rr,
            mol.F_bs,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 2
        * oe.contract(
            "rc,sc,ea,fb,fh,chfe,Qar,Qbs,cf",
            mol.C_rr,
            mol.G_sr,
            mol.H_ba,
            mol.H_ab,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 2
        * oe.contract(
            "rc,sc,ea,fb,fh,chfe,Qar,Qbs,cf",
            mol.C_rr,
            mol.G_sr,
            mol.H_ba,
            mol.H_ab,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,cd,gb,rh,sa,dhcg,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ar,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ra,eb,sd,ch,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ed,ca,sb,rh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,rb,ca,fg,sd,dgcf,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,eb,ca,rd,sh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,sd,rf,ga,cb,dfcg,Qar,Qbs,dc",
            mol.E_ar,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ra,eb,sd,ch,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ed,ca,sb,rh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,rb,ca,fg,sd,dgcf,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,eb,ca,rd,sh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,sd,rf,ga,cb,dfcg,Qar,Qbs,dc",
            mol.E_ar,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ed,rg,cb,sa,dgce,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,rb,sd,fa,ch,dhcf,Qar,Qbs,dc",
            mol.E_ar,
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,eb,eg,rg,sa,dgce,Qar,Qbs,ge",
            mol.E_ar,
            mol.A_bb,
            mol.F_bs,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,eb,ca,sd,rh,dhce,Qar,Qbs,dc",
            mol.E_ar,
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,eb,eg,rg,sa,dgce,Qar,Qbs,ge",
            mol.E_ar,
            mol.A_bb,
            mol.F_bs,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,rd,sf,ga,cb,dfcg,Qar,Qbs,dc",
            mol.E_ar,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "ra,cd,sd,fb,fh,dhfc,Qar,Qbs,df",
            mol.E_ra,
            mol.I_br,
            mol.G_sr,
            mol.H_ab,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "ra,cd,sd,fb,fh,dhfc,Qar,Qbs,df",
            mol.E_ra,
            mol.I_br,
            mol.G_sr,
            mol.H_ab,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "ra,cb,ce,sf,ge,fegc,Qar,Qbs,ec",
            mol.E_ra,
            mol.A_bb,
            mol.F_bs,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "ra,cb,ce,sf,ge,fegc,Qar,Qbs,ec",
            mol.E_ra,
            mol.A_bb,
            mol.F_bs,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,rb,ea,sd,eh,dhec,Qar,Qbs,de",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ea,sd,rg,eb,dgec,Qar,Qbs,de",
            mol.I_br,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ea,rd,sg,eb,dgec,Qar,Qbs,de",
            mol.I_br,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,cb,rf,gf,sa,dfgc,Qar,Qbs,fc",
            mol.I_br,
            mol.A_bb,
            mol.G_rs,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,ea,cg,sb,rg,dgec,Qar,Qbs,gc",
            mol.I_br,
            mol.B_aa,
            mol.F_bs,
            mol.F_sb,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,cf,rf,hb,sa,dfhc,Qar,Qbs,fc",
            mol.I_br,
            mol.F_bs,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,sb,re,ca,ge,degc,Qar,Qbs,ec",
            mol.I_br,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,ea,rd,sg,eb,dgec,Qar,Qbs,de",
            mol.I_br,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,ea,cg,sb,rg,dgec,Qar,Qbs,gc",
            mol.I_br,
            mol.B_aa,
            mol.F_bs,
            mol.F_sb,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,ea,sd,rg,eb,dgec,Qar,Qbs,de",
            mol.I_br,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 4
        * oe.contract(
            "cd,sb,re,ca,ge,degc,Qar,Qbs,ec",
            mol.I_br,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "rb,ca,de,de,sh,hecd,Qar,Qbs,ed",
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.F_bs,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "rb,cd,se,ca,gd,edgc,Qar,Qbs,dc",
            mol.I_rb,
            mol.F_bs,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cb,da,re,sf,cf,efdc,Qar,Qbs,fc",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.F_bs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cb,da,cf,sg,rf,gfdc,Qar,Qbs,fc",
            mol.A_bb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cb,rd,se,ca,ge,degc,Qar,Qbs,ec",
            mol.A_bb,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cb,sd,re,ca,ge,degc,Qar,Qbs,ec",
            mol.A_bb,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cb,da,re,sf,cf,efdc,Qar,Qbs,fc",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.F_bs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cb,rd,se,ca,ge,degc,Qar,Qbs,ec",
            mol.A_bb,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cb,sd,re,ca,ge,degc,Qar,Qbs,ec",
            mol.A_bb,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "rc,sd,ed,ea,hb,cdhe,Qar,Qbs,de",
            mol.C_rr,
            mol.D_ss,
            mol.F_bs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,se,rd,ca,hb,edhc,Qar,Qbs,dc",
            mol.F_bs,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 4
        * oe.contract(
            "cd,se,rd,ca,hb,edhc,Qar,Qbs,dc",
            mol.F_bs,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 8
        * oe.contract(
            "cd,cd,ra,gb,sh,dhcg,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 8
        * oe.contract(
            "cd,ca,rd,gh,sb,dhcg,Qar,Qbs,dc",
            mol.E_ar,
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 8
        * oe.contract(
            "cd,ca,rd,gh,sb,dhcg,Qar,Qbs,dc",
            mol.E_ar,
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 8
        * oe.contract(
            "cd,ra,ed,sg,cb,dgce,Qar,Qbs,dc",
            mol.E_ar,
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 8
        * oe.contract(
            "cd,ra,eb,sf,ef,dfce,Qar,Qbs,fe",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.F_bs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 8
        * oe.contract(
            "cd,ra,eb,sf,ef,dfce,Qar,Qbs,fe",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.F_bs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 8
        * oe.contract(
            "cd,eb,sf,rf,ea,dfce,Qar,Qbs,fe",
            mol.E_ar,
            mol.A_bb,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 8
        * oe.contract(
            "cd,eb,sf,rf,ea,dfce,Qar,Qbs,fe",
            mol.E_ar,
            mol.A_bb,
            mol.D_ss,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 8
        * oe.contract(
            "cd,rd,sb,fa,ch,dhcf,Qar,Qbs,dc",
            mol.E_ar,
            mol.C_rr,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 8
        * oe.contract(
            "ra,cd,cb,sf,gf,dfgc,Qar,Qbs,fc",
            mol.E_ra,
            mol.I_br,
            mol.A_bb,
            mol.D_ss,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 8
        * oe.contract(
            "ra,cd,se,ce,hb,dehc,Qar,Qbs,ec",
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.F_bs,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 8
        * oe.contract(
            "cd,ea,rd,sb,eh,dhec,Qar,Qbs,de",
            mol.I_br,
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 8
        * oe.contract(
            "ca,rd,ef,sd,cb,dfce,Qar,Qbs,dc",
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 8
        * oe.contract(
            "ca,rd,ef,ef,sb,dfce,Qar,Qbs,fe",
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.F_bs,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
        + 8
        * oe.contract(
            "ca,rd,ef,sd,cb,dfce,Qar,Qbs,dc",
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ra,
        )
        + 8
        * oe.contract(
            "rc,de,sb,da,ge,cegd,Qar,Qbs,ed",
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_sb,
        )
    )

    return x2_exch_disp_sinf_term3
