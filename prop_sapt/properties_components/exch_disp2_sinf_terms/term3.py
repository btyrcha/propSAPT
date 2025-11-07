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
            "ar,cd,se,fb,bs,reac,df",
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
            "ar,cb,sd,ef,bs,rfac,de",
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
            "ar,cb,sd,ef,bs,rdae,fc",
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
            "cr,ad,eb,sf,bs,rfae,dc",
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
            "ac,rd,be,fs,er,csab,df",
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
            "cd,ea,rf,bs,ar,dseb,fc",
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
            "cd,ea,rf,bs,ar,fscb,de",
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
            "cr,db,se,af,bs,read,fc",
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
            "cr,sd,be,af,fs,reab,dc",
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
            "bc,da,re,fs,ar,esdb,cf",
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
            "ca,rd,es,bf,ar,dscb,fe",
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
            "rc,bs,da,ef,ar,cseb,fd",
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
            "ar,cb,sd,ef,bs,dfac,re",
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
            "ar,cb,sd,ef,bs,rfec,da",
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
            "cr,bd,ea,fs,af,rseb,dc",
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
            "cr,bd,ea,fs,af,dscb,re",
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
            "cd,ea,bs,rf,ar,dfeb,sc",
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
            "cd,ea,bs,rf,ar,dsec,fb",
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
            "cd,sr,ab,ef,bs,dfac,re",
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
            "cd,sr,ab,ef,bs,rfec,da",
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
            "cd,rs,ba,ef,ar,dfeb,sc",
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
            "cd,rs,ba,ef,ar,dsec,fb",
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
            "cb,ds,er,af,be,rsac,fd",
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
            "cb,ds,er,af,be,rfad,sc",
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
            "ar,cd,se,fb,bs,deac,rf",
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
            "ar,cd,se,fb,bs,refc,da",
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
            "ac,rd,be,fs,er,dsab,cf",
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
            "ac,rd,be,fs,er,csfb,da",
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
            "cr,db,se,af,bs,rfad,ec",
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
            "cr,db,se,af,bs,reac,fd",
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
            "cr,sd,be,af,fs,rdab,ec",
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
            "cr,sd,be,af,fs,reac,db",
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
            "bc,da,re,fs,ar,csdb,ef",
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
            "bc,da,re,fs,ar,esfb,cd",
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
            "rc,bs,da,ef,ar,cfeb,sd",
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
            "rc,bs,da,ef,ar,csed,fb",
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
            "ar,cb,sd,ef,bs,rfac,de",
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
            "ar,bs,cd,ef,fc,dsab,re",
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
            "ar,bs,cd,ef,fc,rseb,da",
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
            "ar,bs,cd,ef,fc,rdab,se",
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
            "ar,bs,cd,ef,fc,rsae,db",
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
            "cr,ad,es,bf,fe,rsab,dc",
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
            "cr,bd,ea,fs,af,rscb,de",
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
            "cd,ea,bs,rf,ar,dseb,fc",
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
            "cd,sr,ab,ef,bs,rfac,de",
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
            "cd,rs,ba,ef,ar,dseb,fc",
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
            "cb,ds,er,af,be,rsad,fc",
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
            "cs,bd,er,af,fe,rsab,dc",
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
            "ar,cd,be,sf,gh,rhab,Qdc,Qes,fg",
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
            "ar,cd,bs,ef,gh,fsab,Qdc,Qhe,rg",
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
            "ar,cd,bs,ef,gh,rsgb,Qdc,Qhe,fa",
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
            "ar,bs,cd,ef,gh,rfab,Qhe,Qdc,sg",
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
            "ar,bs,cd,ef,gh,rsag,Qhe,Qdc,fb",
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
            "ar,sc,de,bf,gh,reab,Qfd,Qhs,cg",
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
            "cr,ad,sb,ef,gh,rfag,Qhe,Qbs,dc",
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
            "cr,de,fa,sb,gh,rhcd,Qag,Qbs,ef",
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
            "cd,be,fa,rg,hs,dscb,Qar,Qeh,gf",
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
            "ra,cd,se,fb,gh,ehfc,Qar,Qbs,dg",
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
            "ra,cb,ds,ef,gh,fsgd,Qar,Qbe,hc",
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
            "ra,cs,bd,ef,gh,fsgb,Qar,Qhe,dc",
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
            "cd,ae,bs,fg,rh,dsab,Qer,Qgf,hc",
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
            "cd,sb,re,fa,gh,degf,Qar,Qbs,hc",
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
            "cb,ad,er,sf,gh,rhag,Qde,Qbs,fc",
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
            "bs,cr,de,fa,gh,rsgb,Qad,Qhc,ef",
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
            "ar,cd,ef,gs,hb,fsae,Qdc,Qbg,rh",
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
            "ar,cd,ef,gs,hb,rshe,Qdc,Qbg,fa",
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
            "ar,cd,es,bf,gh,rsab,Qhc,Qde,fg",
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
            "ar,cd,se,bf,gh,rfag,Qhc,Qds,eb",
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
            "ar,cb,ds,ef,gh,rsag,Qhe,Qbd,fc",
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
            "ar,bc,sd,ef,gh,rfab,Qhe,Qcs,dg",
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
            "cr,ad,ef,gs,bh,rsab,Qhe,Qfg,dc",
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
            "cr,bd,ef,ga,hs,rscb,Qae,Qfh,dg",
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
            "ac,bd,re,fs,gh,csab,Qhr,Qdg,ef",
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
            "ac,dr,bs,ef,gh,rsab,Qhd,Qfg,ce",
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
            "ac,dr,sb,ef,gh,rhae,Qfd,Qbs,cg",
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
            "ac,rd,bs,ef,gh,cseb,Qhr,Qfg,da",
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
            "ac,rd,sb,ef,gh,chge,Qfr,Qbs,da",
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
            "ra,cd,eb,fs,gh,dsgc,Qar,Qbf,he",
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
            "ra,cd,be,sf,gh,dhgb,Qar,Qes,fc",
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
            "ra,cd,es,bf,gh,dsgb,Qar,Qhe,fc",
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
            "ra,cd,se,bf,gh,dfgc,Qar,Qhs,eb",
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
            "cr,de,af,sg,bh,rhab,Qfd,Qes,gc",
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
            "cr,de,fs,ba,gh,rsgb,Qad,Qef,hc",
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
            "cd,eb,af,sg,rh,dhac,Qfr,Qbs,ge",
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
            "cd,be,af,gs,rh,dsab,Qfr,Qeg,hc",
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
            "cd,es,rf,ba,gh,dsgc,Qar,Qhe,fb",
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
            "cd,se,rf,ba,gh,dfgb,Qar,Qhs,ec",
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
            "cd,ea,fr,sb,gh,rhgc,Qaf,Qbs,de",
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
            "cd,ae,rf,sb,gh,dhac,Qer,Qbs,fg",
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
            "cd,er,ab,fs,gh,rsac,Qhe,Qbg,df",
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
            "cb,dr,es,af,gh,rsae,Qhd,Qbg,fc",
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
            "bc,da,re,sf,gh,fhgb,Qar,Qcs,ed",
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
            "bc,ad,er,sf,gh,rhab,Qde,Qcs,fg",
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
            "ca,dr,bs,ef,gh,rsgb,Qad,Qhe,fc",
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
            "ac,rd,bs,ef,gh,fsab,Qcr,Qhe,dg",
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
            "cr,bs,de,fa,gh,rhgb,Qac,Qed,sf",
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
            "cr,bs,de,fa,gh,rsgf,Qac,Qed,hb",
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
            "cr,ds,be,af,gh,rsab,Qhc,Qfg,ed",
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
            "cr,sd,ba,ef,gh,rhgb,Qac,Qfs,de",
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
            "rc,sd,ba,ef,gh,dheb,Qar,Qfs,cg",
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
            "ar,cd,es,fb,gh,dsac,Qhe,Qbg,rf",
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
            "ar,cd,es,fb,gh,rsfc,Qhe,Qbg,da",
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
            "ar,cd,se,bf,gh,ehab,Qfc,Qds,rg",
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
            "ar,cd,se,bf,gh,rhgb,Qfc,Qds,ea",
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
            "cr,bd,ea,sf,gh,fhcb,Qag,Qds,re",
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
            "cr,bd,ae,sf,gh,rhab,Qeg,Qds,fc",
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
            "ac,dr,es,bf,gh,rsab,Qfd,Qhe,cg",
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
            "ac,rd,es,bf,gh,csgb,Qfr,Qhe,da",
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
            "bc,de,fa,sr,gh,rhgb,Qad,Qes,cf",
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
            "bc,de,af,sr,gh,chab,Qfd,Qes,rg",
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
            "cd,eb,rs,af,gh,dsac,Qhr,Qbg,fe",
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
            "cd,be,rs,af,gh,dfab,Qhr,Qeg,sc",
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
            "cd,bs,re,af,gh,deab,Qhr,Qfg,sc",
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
            "cd,bs,re,af,gh,dsac,Qhr,Qfg,eb",
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
            "cd,ea,fr,gs,hb,rshc,Qaf,Qbg,de",
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
            "cd,ea,sr,fg,hb,dghc,Qaf,Qbs,re",
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
            "cd,ae,rf,gs,hb,dsac,Qer,Qbg,fh",
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
            "cd,ae,sr,fg,hb,rgac,Qef,Qbs,dh",
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
            "cd,bs,er,fa,gh,rhgb,Qac,Qde,sf",
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
            "cd,bs,er,fa,gh,rsgf,Qac,Qde,hb",
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
            "cb,ad,es,fr,gh,rhae,Qdg,Qbf,sc",
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
            "cb,dr,es,fa,gh,rsgf,Qad,Qbe,hc",
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
            "cb,sr,de,fa,gh,rhgf,Qad,Qbs,ec",
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
            "bc,ad,es,fr,gh,rsab,Qdg,Qcf,he",
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
            "bc,dr,se,fa,gh,rhgb,Qad,Qcs,ef",
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
            "bc,sr,de,fa,gh,regb,Qad,Qcs,hf",
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
            "cr,ds,be,fa,gh,rsgb,Qac,Qhd,ef",
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
            "cr,sd,be,fa,gh,regf,Qac,Qhs,db",
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
            "cr,bd,ef,ga,hs,dscb,Qae,Qfh,rg",
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
            "cr,bd,ef,ag,hs,rsab,Qge,Qfh,dc",
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
            "ac,bd,er,fs,gh,rsab,Qhe,Qdg,cf",
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
            "ac,bd,re,fs,gh,csfb,Qhr,Qdg,ea",
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
            "cr,de,af,gs,bh,rsab,Qfd,Qeg,hc",
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
            "cr,de,af,sg,bh,rhac,Qfd,Qes,gb",
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
            "cr,de,fs,ba,gh,rsgc,Qad,Qef,hb",
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
            "cr,de,sf,ba,gh,rhgb,Qad,Qes,fc",
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
            "cd,eb,af,gs,rh,dsac,Qfr,Qbg,he",
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
            "cd,be,af,sg,rh,dhab,Qfr,Qes,gc",
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
            "cd,es,rf,ba,gh,dsgb,Qar,Qhe,fc",
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
            "cd,se,rf,ba,gh,dfgc,Qar,Qhs,eb",
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
            "cd,er,ab,fs,gh,rsfc,Qhe,Qbg,da",
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
            "cd,re,ab,fs,gh,dsac,Qhr,Qbg,ef",
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
            "cb,dr,es,af,gh,rfae,Qhd,Qbg,sc",
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
            "bc,da,er,sf,gh,rhgb,Qae,Qcs,fd",
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
            "bc,ad,re,sf,gh,fhab,Qdr,Qcs,eg",
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
            "bc,dr,es,af,gh,rsab,Qhd,Qcg,fe",
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
            "cr,sd,ba,ef,gh,rheb,Qac,Qfs,dg",
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
            "rc,sd,ba,ef,gh,dhgb,Qar,Qfs,ce",
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
            "ar,cd,be,sf,gh,fhab,Qdc,Qes,rg",
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
            "ar,cd,be,sf,gh,rhgb,Qdc,Qes,fa",
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
            "ar,cd,es,fb,gh,rsac,Qhe,Qbg,df",
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
            "ar,cd,se,bf,gh,rhab,Qfc,Qds,eg",
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
            "ar,cb,ds,ef,gh,rfad,Qhe,Qbg,sc",
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
            "ar,bc,ds,ef,gh,rsab,Qhe,Qcg,fd",
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
            "ar,sc,de,bf,gh,ceab,Qfd,Qhs,rg",
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
            "ar,sc,de,bf,gh,regb,Qfd,Qhs,ca",
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
            "cr,ad,be,fs,gh,rsab,Qhf,Qeg,dc",
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
            "cr,de,fa,sb,gh,ehcd,Qag,Qbs,rf",
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
            "cr,de,af,sb,gh,rhad,Qfg,Qbs,ec",
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
            "cr,de,fa,bs,gh,hscb,Qad,Qeg,rf",
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
            "cr,de,af,bs,gh,rsab,Qfd,Qeg,hc",
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
            "cr,bd,ea,sf,gh,rhcb,Qag,Qds,fe",
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
            "ac,rd,es,bf,gh,csab,Qfr,Qhe,dg",
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
            "cd,be,fa,gr,hs,rscb,Qag,Qeh,df",
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
            "cd,be,af,rg,hs,dsab,Qfr,Qeh,gc",
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
            "ra,cd,se,fb,gh,dhfc,Qar,Qbs,eg",
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
            "ra,cd,se,fb,gh,ehgc,Qar,Qbs,df",
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
            "ra,cb,ds,ef,gh,fhgd,Qar,Qbe,sc",
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
            "ra,bc,ds,ef,gh,fsgb,Qar,Qce,hd",
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
            "bc,de,af,sr,gh,rhab,Qfd,Qes,cg",
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
            "cd,be,rs,af,gh,dsab,Qhr,Qeg,fc",
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
            "cd,ae,bs,fg,rh,dhab,Qer,Qgf,sc",
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
            "cd,ae,bs,fg,rh,dsac,Qer,Qgf,hb",
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
            "cd,bs,re,af,gh,dsab,Qhr,Qfg,ec",
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
            "cd,sb,re,fa,gh,dhgf,Qar,Qbs,ec",
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
            "cd,sb,re,fa,gh,degc,Qar,Qbs,hf",
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
            "cd,ea,rf,gs,hb,dshc,Qar,Qbg,fe",
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
            "cd,ea,sr,fg,hb,rghc,Qaf,Qbs,de",
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
            "cd,ae,fr,gs,hb,rsac,Qef,Qbg,dh",
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
            "cd,ae,sr,fg,hb,dgac,Qef,Qbs,rh",
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
            "cd,ae,fs,bg,hr,rsab,Qec,Qdh,gf",
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
            "cd,bs,er,fa,gh,rsgb,Qac,Qde,hf",
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
            "cb,ad,er,fs,gh,rsag,Qde,Qbf,hc",
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
            "cb,ad,es,fr,gh,rsae,Qdg,Qbf,hc",
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
            "cb,dr,se,fa,gh,rhgf,Qad,Qbs,ec",
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
            "cb,sr,de,fa,gh,regf,Qad,Qbs,hc",
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
            "bc,ad,er,sf,gh,rhab,Qde,Qcs,fg",
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
            "bc,dr,es,fa,gh,rsgb,Qad,Qce,hf",
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
            "bc,sr,de,fa,gh,rhgb,Qad,Qcs,ef",
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
            "cr,sd,be,fa,gh,regb,Qac,Qhs,df",
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
            "bs,cr,de,fa,gh,regb,Qad,Qhc,sf",
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
            "bs,cr,de,fa,gh,rsgf,Qad,Qhc,eb",
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
            "ar,cd,ef,gs,hb,rsae,Qdc,Qbg,fh",
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
            "ar,cd,eb,fs,gh,rsag,Qdc,Qbf,he",
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
            "ar,cd,be,sf,gh,rhab,Qdc,Qes,fg",
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
            "ar,cb,sd,ef,gh,rfag,Qhe,Qbs,dc",
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
            "ar,bc,ds,ef,gh,rsab,Qhe,Qcd,fg",
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
            "cr,ad,ef,bg,hs,rsab,Qfe,Qgh,dc",
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
            "ac,rd,sb,ef,gh,chae,Qfr,Qbs,dg",
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
            "cd,ea,fr,bs,gh,rscb,Qaf,Qhg,de",
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
            "cd,ae,rf,bs,gh,dsab,Qer,Qhg,fc",
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
            "ra,cd,be,fs,gh,dsgb,Qar,Qef,hc",
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
            "ra,cd,se,bf,gh,dfgb,Qar,Qhs,ec",
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
            "cd,ae,fr,sb,gh,rhac,Qef,Qbs,dg",
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
            "ca,rd,bs,ef,gh,fsgb,Qar,Qhe,dc",
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
            "ac,dr,bs,ef,gh,rsab,Qcd,Qhe,fg",
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
            "ac,dr,es,bf,gh,rsab,Qcd,Qhg,fe",
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
            "cr,bs,de,fa,gh,rsgb,Qac,Qed,hf",
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
