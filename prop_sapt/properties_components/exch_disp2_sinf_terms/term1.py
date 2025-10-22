import numpy as np
import opt_einsum as oe

from prop_sapt.molecule import Dimer


def get_exch_disp2_sinf_property_term1(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
):

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

    # < V R(X) | P R(V) >
    x2_exch_disp_sinf_term1 = (
        -4
        * oe.contract(
            "cd,ee,gg,dgce,eg", mol.E_ar, mol.A_bb, mol.D_ss, mol.t_rsab, vrx_bs
        )
        - 4
        * oe.contract(
            "cc,ee,gh,ehcg,ce", mol.B_aa, mol.C_rr, mol.F_bs, mol.t_rsab, vrx_ar
        )
        - 2
        * oe.contract(
            "cd,ee,dh,dhec,ed", mol.I_br, mol.B_aa, mol.G_rs, mol.t_rsab, vrx_ar
        )
        - 2
        * oe.contract(
            "cc,ef,ge,fegc,ce", mol.A_bb, mol.G_sr, mol.J_as, mol.t_rsab, vrx_bs
        )
        + 2
        * oe.contract(
            "cd,ee,gc,degc,ce", mol.I_br, mol.D_ss, mol.H_ab, mol.t_rsab, vrx_bs
        )
        + 2
        * oe.contract(
            "cc,ef,fh,chfe,fc", mol.C_rr, mol.H_ba, mol.J_as, mol.t_rsab, vrx_ar
        )
        + 4
        * oe.contract(
            "cd,df,gc,dfcg,cd", mol.E_ar, mol.G_rs, mol.H_ba, mol.t_rsab, vrx_ar
        )
        + 4
        * oe.contract(
            "cd,df,gc,fdgc,cd", mol.F_bs, mol.G_sr, mol.H_ab, mol.t_rsab, vrx_bs
        )
        - 8
        * oe.contract(
            "cd,ef,de,fc,decf,cfde",
            mol.E_ar,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            vrx_abrs,
        )
        - 8
        * oe.contract(
            "cd,ef,fc,de,cfde,decf",
            mol.E_ra,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            vrx_abrs,
        )
        - 4
        * oe.contract(
            "cd,df,gg,fc,dgcf,cfdg",
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.t_rsab,
            vrx_abrs,
        )
        - 4
        * oe.contract(
            "cd,ec,gg,de,cgde,decg",
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.t_rsab,
            vrx_abrs,
        )
        - 4
        * oe.contract(
            "cc,ef,ge,fg,cfge,gecf",
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            vrx_abrs,
        )
        - 4
        * oe.contract(
            "cc,ef,fh,he,cehf,hfce",
            mol.C_rr,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            vrx_abrs,
        )
        - 2
        * oe.contract(
            "cd,df,gc,fg,dfgc,gcdf",
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            vrx_abrs,
        )
        - 2
        * oe.contract(
            "cd,ec,dh,he,cehd,hdce",
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            vrx_abrs,
        )
        - 2
        * oe.contract(
            "cc,ee,gh,hg,hgec,echg",
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            vrx_abrs,
        )
        - 2
        * oe.contract(
            "cc,ee,gh,hg,cehg,hgce",
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            vrx_abrs,
        )
        + 2
        * oe.contract(
            "cd,dc,gg,ii,digc,gcdi",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.t_rsab,
            vrx_abrs,
        )
        + 2
        * oe.contract(
            "cc,ee,gh,hg,ehgc,gceh",
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "cd,ee,dh,hc,dhce,cedh",
            mol.E_ar,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "cd,ee,gc,dg,cgde,decg",
            mol.E_ra,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "cd,ee,gc,dg,dgec,ecdg",
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.G_rs,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "cd,ee,dh,hc,ched,edch",
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "cc,ee,gg,ii,giec,ecgi",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "cd,dc,gh,hg,dchg,hgdc",
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            vrx_abrs,
        )
        + 8
        * oe.contract(
            "cd,dc,gg,ii,dicg,cgdi",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.t_rsab,
            vrx_abrs,
        )
        + 8
        * oe.contract(
            "cc,ee,gh,hg,ehcg,cgeh",
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.t_rsab,
            vrx_abrs,
        )
        - 4
        * oe.contract(
            "cd,db,sf,gc,bs,dfcg,cd",
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "cd,re,ff,ea,ar,dfce,ef",
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "cd,eb,ff,se,bs,dfce,ef",
            mol.E_ar,
            mol.A_bb,
            mol.D_ss,
            mol.F_sb,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "cd,ee,sg,gb,bs,dgce,eg",
            mol.E_ar,
            mol.A_bb,
            mol.D_ss,
            mol.F_sb,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "ca,dd,rc,gh,ar,chdg,dc",
            mol.E_ra,
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "rc,ca,ee,gh,ar,ehcg,ce",
            mol.E_ra,
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "cb,sd,df,gc,bs,fdgc,cd",
            mol.A_bb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "cc,ee,sg,hb,bs,gehc,ce",
            mol.A_bb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "ca,rd,df,gc,ar,dfcg,cd",
            mol.B_aa,
            mol.C_rr,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "cc,ee,rg,ha,ar,egch,ce",
            mol.B_aa,
            mol.C_rr,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "rc,de,fd,ea,ar,cefd,de",
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "cc,ef,gb,sg,bs,cfge,gc",
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "ca,dc,ff,rh,ar,chfd,fc",
            mol.E_ra,
            mol.I_br,
            mol.B_aa,
            mol.G_rs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "rc,de,ca,eh,ar,ehcd,ce",
            mol.E_ra,
            mol.I_br,
            mol.B_aa,
            mol.G_rs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "cd,re,fc,ea,ar,defc,ce",
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cd,df,gb,sg,bs,dfgc,gd",
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "cb,sc,ef,fh,bs,chfe,fc",
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "rc,de,ca,gd,ar,edgc,cd",
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cb,dd,sf,fh,bs,fhdc,df",
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "cb,sc,ef,ge,bs,fegc,ce",
            mol.A_bb,
            mol.F_sb,
            mol.G_sr,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cc,ea,fg,rf,ar,gfec,cf",
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cc,eb,sf,ge,bs,fegc,ce",
            mol.A_bb,
            mol.F_sb,
            mol.G_sr,
            mol.J_as,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "rc,dd,fa,gf,ar,cdgf,fd",
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cc,se,fg,gb,bs,cegf,gc",
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "ca,rc,ef,fh,ar,chfe,fc",
            mol.E_ra,
            mol.C_rr,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "rc,dd,fa,ch,ar,dhcf,cd",
            mol.E_ra,
            mol.C_rr,
            mol.H_ba,
            mol.J_as,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "cd,db,ff,sh,bs,dhfc,fd",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "cd,rc,fa,gg,ar,dgfc,cg",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "cd,se,eb,gc,bs,degc,ce",
            mol.I_br,
            mol.D_ss,
            mol.F_sb,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "cd,ee,sc,hb,bs,dehc,ce",
            mol.I_br,
            mol.D_ss,
            mol.F_sb,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "cb,dd,fg,sf,bs,dgfc,fd",
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "cb,dd,sf,gc,bs,fdgc,cd",
            mol.A_bb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "cc,re,fg,ga,ar,egfc,cg",
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "cc,se,eg,hb,bs,gehc,ce",
            mol.A_bb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "ca,dd,rf,gc,ar,dfcg,cd",
            mol.B_aa,
            mol.C_rr,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "cc,re,eg,ha,ar,egch,ce",
            mol.B_aa,
            mol.C_rr,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,da,rf,gc,ar,dfcg,cd",
            mol.E_ar,
            mol.E_ra,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,rc,dg,ha,ar,dgch,cd",
            mol.E_ar,
            mol.E_ra,
            mol.G_rs,
            mol.H_ba,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,eb,dg,sc,bs,dgce,cd",
            mol.E_ar,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,ee,rg,ga,ar,dgce,eg",
            mol.E_ar,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cb,dd,fg,sc,bs,cgdf,dc",
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "rc,da,cf,fh,ar,hfdc,cf",
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cb,dd,ff,sh,bs,fhdc,df",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cc,ea,rf,gg,ar,fgec,cg",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cd,db,sf,gc,bs,fdgc,cd",
            mol.F_bs,
            mol.F_sb,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cd,sc,dg,hb,bs,gdhc,cd",
            mol.F_bs,
            mol.F_sb,
            mol.G_sr,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "sc,ce,fg,gb,bs,cegf,gc",
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,rc,fa,gf,ar,dcgf,fc",
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        - 8
        * oe.contract(
            "cd,ra,eb,dg,sc,dgce,Qar,Qbs,cd",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 8
        * oe.contract(
            "cd,da,sb,rf,gc,dfcg,Qar,Qbs,cd",
            mol.E_ar,
            mol.E_ra,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 8
        * oe.contract(
            "cd,rc,sb,dg,ha,dgch,Qar,Qbs,cd",
            mol.E_ar,
            mol.E_ra,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 8
        * oe.contract(
            "cd,db,rf,ga,sc,dfcg,Qar,Qbs,cd",
            mol.E_ar,
            mol.I_rb,
            mol.G_rs,
            mol.H_ba,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 8
        * oe.contract(
            "cd,ee,sb,rg,ga,dgce,Qar,Qbs,eg",
            mol.E_ar,
            mol.A_bb,
            mol.F_sb,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 8
        * oe.contract(
            "cd,eb,sf,re,fa,decf,Qar,Qbs,fe",
            mol.E_ar,
            mol.F_sb,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 8
        * oe.contract(
            "ra,cb,dd,fg,sc,cgdf,Qar,Qbs,dc",
            mol.E_ra,
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 8
        * oe.contract(
            "ra,cb,dd,ff,sh,fhdc,Qar,Qbs,df",
            mol.E_ra,
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 8
        * oe.contract(
            "ra,cd,db,sf,gc,fdgc,Qar,Qbs,cd",
            mol.E_ra,
            mol.F_bs,
            mol.F_sb,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 8
        * oe.contract(
            "ra,cd,sc,dg,hb,gdhc,Qar,Qbs,cd",
            mol.E_ra,
            mol.F_bs,
            mol.F_sb,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 8
        * oe.contract(
            "ra,sc,ce,fg,gb,cegf,Qar,Qbs,gc",
            mol.E_ra,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 8
        * oe.contract(
            "ca,rd,ef,sc,db,cfde,Qar,Qbs,dc",
            mol.E_ra,
            mol.E_ra,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 8
        * oe.contract(
            "rc,da,cf,sb,fh,hfdc,Qar,Qbs,cf",
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.F_sb,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 8
        * oe.contract(
            "rc,ce,sf,gb,ea,fegc,Qar,Qbs,ce",
            mol.I_rb,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 8
        * oe.contract(
            "cc,ea,rf,gg,sb,fgec,Qar,Qbs,cg",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 8
        * oe.contract(
            "sb,cd,rc,fa,gf,dcgf,Qar,Qbs,fc",
            mol.F_sb,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "cd,da,rb,sf,gc,dfcg,Qar,Qbs,cd",
            mol.E_ar,
            mol.E_ra,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "cd,rc,db,sg,ha,dgch,Qar,Qbs,cd",
            mol.E_ar,
            mol.E_ra,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "cd,rb,ee,sg,ga,dgce,Qar,Qbs,eg",
            mol.E_ar,
            mol.I_rb,
            mol.A_bb,
            mol.D_ss,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "cd,rb,ee,sg,ga,decg,Qar,Qbs,ge",
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.F_sb,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "cd,re,eb,gg,sa,dgce,Qar,Qbs,eg",
            mol.E_ar,
            mol.I_rb,
            mol.A_bb,
            mol.D_ss,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "cd,re,sf,fb,ea,dfce,Qar,Qbs,ef",
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.F_sb,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "ra,cd,db,ff,sh,dhfc,Qar,Qbs,fd",
            mol.E_ra,
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "ra,cd,se,eb,gc,degc,Qar,Qbs,ce",
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.F_sb,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "ra,cd,ee,sc,hb,dehc,Qar,Qbs,ce",
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.F_sb,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "ra,cb,dd,fg,sf,dgfc,Qar,Qbs,fd",
            mol.E_ra,
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "ra,cb,dd,sf,gc,fdgc,Qar,Qbs,cd",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "ra,cc,se,eg,hb,gehc,Qar,Qbs,ce",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "ca,rd,ec,sg,db,cgde,Qar,Qbs,dc",
            mol.E_ra,
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "ca,rc,ef,gb,sg,cfge,Qar,Qbs,gc",
            mol.E_ra,
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "ca,rc,sb,ef,fh,chfe,Qar,Qbs,fc",
            mol.E_ra,
            mol.C_rr,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "rc,dd,fg,cb,sa,dgcf,Qar,Qbs,cd",
            mol.E_ra,
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "rc,dd,sb,fa,ch,dhcf,Qar,Qbs,cd",
            mol.E_ra,
            mol.C_rr,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "cd,rc,fa,gg,sb,dgfc,Qar,Qbs,cg",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "cd,rc,sf,gb,fa,dfgc,Qar,Qbs,cf",
            mol.I_br,
            mol.I_rb,
            mol.D_ss,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "rb,cc,ea,ff,sh,hfec,Qar,Qbs,cf",
            mol.I_rb,
            mol.A_bb,
            mol.B_aa,
            mol.D_ss,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "rb,ca,dd,fg,sc,dgcf,Qar,Qbs,cd",
            mol.I_rb,
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "rb,sc,ce,fa,gf,ecgf,Qar,Qbs,fc",
            mol.I_rb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "cb,dd,rc,gh,sa,chdg,Qar,Qbs,dc",
            mol.I_rb,
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "cb,rc,ea,fg,sf,cgfe,Qar,Qbs,fc",
            mol.I_rb,
            mol.C_rr,
            mol.H_ba,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "rc,cb,ea,sf,fh,hfec,Qar,Qbs,cf",
            mol.I_rb,
            mol.A_bb,
            mol.B_aa,
            mol.D_ss,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "rc,dd,sf,ca,hb,fdhc,Qar,Qbs,cd",
            mol.I_rb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "cb,da,re,eg,sd,egdc,Qar,Qbs,de",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "cb,dd,ff,rh,sa,fhdc,Qar,Qbs,df",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "cc,re,sb,fg,ga,egfc,Qar,Qbs,cg",
            mol.A_bb,
            mol.C_rr,
            mol.F_sb,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "ca,dd,sb,rf,gc,dfcg,Qar,Qbs,cd",
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "cc,re,sb,eg,ha,egch,Qar,Qbs,ce",
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "rc,de,eb,gd,sa,cegd,Qar,Qbs,de",
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "rc,de,sd,gb,ea,cegd,Qar,Qbs,de",
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "rc,db,se,ea,gd,cdge,Qar,Qbs,ed",
            mol.C_rr,
            mol.F_sb,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 4
        * oe.contract(
            "rc,ce,fg,gb,sa,cegf,Qar,Qbs,gc",
            mol.C_rr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "cc,re,fa,gb,sg,cegf,Qar,Qbs,gc",
            mol.C_rr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "ca,dc,rf,gb,sg,cfgd,Qar,Qbs,gc",
            mol.E_ra,
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "ca,rb,sc,ef,fh,chfe,Qar,Qbs,fc",
            mol.E_ra,
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "ca,db,ee,sc,rh,ched,Qar,Qbs,ec",
            mol.E_ra,
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "ca,rc,se,fg,gb,cegf,Qar,Qbs,gc",
            mol.E_ra,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "rc,de,eg,cb,sa,egcd,Qar,Qbs,ce",
            mol.E_ra,
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "rc,db,sd,fa,ch,dhcf,Qar,Qbs,cd",
            mol.E_ra,
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "rc,db,ca,sf,fh,fhcd,Qar,Qbs,cf",
            mol.E_ra,
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "rc,dd,sf,ga,cb,dfcg,Qar,Qbs,cd",
            mol.E_ra,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "cd,rb,ea,dg,se,dgec,Qar,Qbs,ed",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "cd,db,ff,rh,sa,dhfc,Qar,Qbs,fd",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "cd,eb,re,gc,sa,degc,Qar,Qbs,ce",
            mol.I_br,
            mol.F_sb,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cd,sc,rf,gb,fa,dfgc,Qar,Qbs,cf",
            mol.I_br,
            mol.F_sb,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "rb,cc,se,fg,ga,egfc,Qar,Qbs,cg",
            mol.I_rb,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "rb,ca,dd,sf,gc,dfcg,Qar,Qbs,cd",
            mol.I_rb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "rb,cc,se,eg,ha,egch,Qar,Qbs,ce",
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "rb,sc,de,ca,gd,edgc,Qar,Qbs,cd",
            mol.I_rb,
            mol.F_sb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cb,da,sc,rf,gd,cfdg,Qar,Qbs,dc",
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "cb,dd,rc,sg,ha,cgdh,Qar,Qbs,dc",
            mol.I_rb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "rc,cb,ef,ge,sa,fegc,Qar,Qbs,ce",
            mol.I_rb,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "rc,db,se,ca,gd,edgc,Qar,Qbs,cd",
            mol.I_rb,
            mol.F_sb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cb,da,sc,fg,rf,gfdc,Qar,Qbs,cf",
            mol.A_bb,
            mol.B_aa,
            mol.F_sb,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cb,rd,ee,gc,sa,degc,Qar,Qbs,ce",
            mol.A_bb,
            mol.C_rr,
            mol.D_ss,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cb,sd,re,fc,ea,defc,Qar,Qbs,ce",
            mol.A_bb,
            mol.G_sr,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cc,ea,fb,sg,rf,gfec,Qar,Qbs,cf",
            mol.A_bb,
            mol.B_aa,
            mol.F_sb,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cc,re,sf,gb,fa,efgc,Qar,Qbs,cf",
            mol.A_bb,
            mol.C_rr,
            mol.D_ss,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "cc,ef,re,hb,sa,fehc,Qar,Qbs,ce",
            mol.A_bb,
            mol.G_sr,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "rc,sd,db,fa,gf,cdgf,Qar,Qbs,fd",
            mol.C_rr,
            mol.D_ss,
            mol.F_sb,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "rc,dd,sf,fa,hb,cdhf,Qar,Qbs,fd",
            mol.C_rr,
            mol.D_ss,
            mol.F_sb,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "ca,dc,rb,ff,sh,chfd,Qar,Qbs,fc",
            mol.E_ra,
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "ca,db,rc,fg,sf,cgfd,Qar,Qbs,fc",
            mol.E_ra,
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "rc,de,eb,ca,sh,ehcd,Qar,Qbs,ce",
            mol.E_ra,
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "rc,db,ee,ch,sa,ehcd,Qar,Qbs,ce",
            mol.E_ra,
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "cd,rb,ea,ff,sc,dfec,Qar,Qbs,cf",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "cd,rb,se,fc,ea,defc,Qar,Qbs,ce",
            mol.I_br,
            mol.I_rb,
            mol.D_ss,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "cd,rc,fa,sg,gb,dgfc,Qar,Qbs,cg",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "cd,rc,ff,hb,sa,dfhc,Qar,Qbs,cf",
            mol.I_br,
            mol.I_rb,
            mol.D_ss,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "rb,cc,ea,sf,fh,hfec,Qar,Qbs,cf",
            mol.I_rb,
            mol.A_bb,
            mol.B_aa,
            mol.D_ss,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "rb,cc,ea,fg,sf,cgfe,Qar,Qbs,fc",
            mol.I_rb,
            mol.C_rr,
            mol.H_ba,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "rb,cc,se,fa,gf,ecgf,Qar,Qbs,fc",
            mol.I_rb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "cb,rc,ef,fh,sa,chfe,Qar,Qbs,fc",
            mol.I_rb,
            mol.C_rr,
            mol.H_ba,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "rc,cb,ea,ff,sh,hfec,Qar,Qbs,cf",
            mol.I_rb,
            mol.A_bb,
            mol.B_aa,
            mol.D_ss,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "rc,sd,df,ca,hb,fdhc,Qar,Qbs,cd",
            mol.I_rb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "cb,da,ee,rg,sd,egdc,Qar,Qbs,de",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "cb,dd,rf,fh,sa,fhdc,Qar,Qbs,df",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "cb,rd,sc,fg,ga,dgfc,Qar,Qbs,cg",
            mol.A_bb,
            mol.C_rr,
            mol.F_sb,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "cc,re,fb,gf,sa,efgc,Qar,Qbs,cf",
            mol.A_bb,
            mol.C_rr,
            mol.F_sb,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "rc,ce,fa,gb,sg,cegf,Qar,Qbs,gc",
            mol.C_rr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 2
        * oe.contract(
            "cc,re,fg,gb,sa,cegf,Qar,Qbs,gc",
            mol.C_rr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,da,fb,rg,sc,dgcf,Qar,Qbs,cd",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,rc,fb,dh,sa,dhcf,Qar,Qbs,cd",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,rb,df,ga,sc,dfcg,Qar,Qbs,cd",
            mol.E_ar,
            mol.I_rb,
            mol.G_rs,
            mol.H_ba,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,db,rf,gc,sa,dfcg,Qar,Qbs,cd",
            mol.E_ar,
            mol.I_rb,
            mol.G_rs,
            mol.H_ba,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,eb,se,rg,ga,dgce,Qar,Qbs,eg",
            mol.E_ar,
            mol.A_bb,
            mol.F_sb,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cd,ee,gb,rg,sa,dgce,Qar,Qbs,eg",
            mol.E_ar,
            mol.A_bb,
            mol.F_sb,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "ra,cd,df,gb,sg,dfgc,Qar,Qbs,gd",
            mol.E_ra,
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "ra,cb,sc,ef,fh,chfe,Qar,Qbs,fc",
            mol.E_ra,
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "ra,cb,dd,sf,fh,fhdc,Qar,Qbs,df",
            mol.E_ra,
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "ra,cb,sc,ef,ge,fegc,Qar,Qbs,ce",
            mol.E_ra,
            mol.A_bb,
            mol.F_sb,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "ra,cc,eb,sf,ge,fegc,Qar,Qbs,ce",
            mol.E_ra,
            mol.A_bb,
            mol.F_sb,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "ra,cc,se,fg,gb,cegf,Qar,Qbs,gc",
            mol.E_ra,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "ca,rd,eb,sc,dh,chde,Qar,Qbs,dc",
            mol.E_ra,
            mol.E_ra,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "ca,dc,ff,sb,rh,chfd,Qar,Qbs,fc",
            mol.E_ra,
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "ca,rb,dd,fg,sc,cgdf,Qar,Qbs,dc",
            mol.E_ra,
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "ca,db,ee,rc,sh,ched,Qar,Qbs,ec",
            mol.E_ra,
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "ca,sc,re,fg,gb,cegf,Qar,Qbs,gc",
            mol.E_ra,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "rc,de,ca,sb,eh,ehcd,Qar,Qbs,ce",
            mol.E_ra,
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "rc,db,ca,fg,sd,dgcf,Qar,Qbs,cd",
            mol.E_ra,
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "rc,db,ca,ff,sh,fhcd,Qar,Qbs,cf",
            mol.E_ra,
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "rc,sd,df,ga,cb,dfcg,Qar,Qbs,cd",
            mol.E_ra,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,db,fa,rg,sf,dgfc,Qar,Qbs,fd",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cd,ea,fb,sc,rf,dfec,Qar,Qbs,cf",
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.F_sb,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cd,sb,re,fc,ea,defc,Qar,Qbs,ce",
            mol.I_br,
            mol.F_sb,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "rb,ca,de,sd,eh,hecd,Qar,Qbs,de",
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.F_sb,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "rb,ca,sd,df,gc,dfcg,Qar,Qbs,cd",
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "rb,cc,ee,sg,ha,egch,Qar,Qbs,ce",
            mol.I_rb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "rb,cd,se,fc,da,edfc,Qar,Qbs,cd",
            mol.I_rb,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cb,da,rc,sf,gd,cfdg,Qar,Qbs,dc",
            mol.I_rb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "cb,dd,sc,rg,ha,cgdh,Qar,Qbs,dc",
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 4
        * oe.contract(
            "rc,cb,se,fg,ga,egfc,Qar,Qbs,cg",
            mol.I_rb,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "rc,da,cf,fb,sh,hfdc,Qar,Qbs,cf",
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.F_sb,
            mol.G_sr,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "rc,ce,eg,hb,sa,gehc,Qar,Qbs,ce",
            mol.I_rb,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "rc,sb,de,ca,gd,edgc,Qar,Qbs,cd",
            mol.I_rb,
            mol.F_sb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cb,da,re,ff,sc,efdc,Qar,Qbs,cf",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cb,rd,se,fc,ea,defc,Qar,Qbs,ce",
            mol.A_bb,
            mol.C_rr,
            mol.D_ss,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cb,de,rd,gc,sa,edgc,Qar,Qbs,cd",
            mol.A_bb,
            mol.G_sr,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cc,ea,rf,sg,gb,fgec,Qar,Qbs,cg",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cc,ea,sb,fg,rf,gfec,Qar,Qbs,cf",
            mol.A_bb,
            mol.B_aa,
            mol.F_sb,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cc,re,ff,hb,sa,efhc,Qar,Qbs,cf",
            mol.A_bb,
            mol.C_rr,
            mol.D_ss,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cc,se,rf,gb,fa,efgc,Qar,Qbs,cf",
            mol.A_bb,
            mol.G_sr,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "rc,dd,sb,fa,gf,cdgf,Qar,Qbs,fd",
            mol.C_rr,
            mol.D_ss,
            mol.F_sb,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "cb,sd,rc,fa,gf,dcgf,Qar,Qbs,fc",
            mol.F_sb,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 4
        * oe.contract(
            "sc,de,rd,ca,hb,edhc,Qar,Qbs,cd",
            mol.F_sb,
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 8
        * oe.contract(
            "cd,ra,db,sf,gc,dfcg,Qar,Qbs,cd",
            mol.E_ar,
            mol.E_ra,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 8
        * oe.contract(
            "cd,ra,eb,ff,se,dfce,Qar,Qbs,ef",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 8
        * oe.contract(
            "cd,ra,ee,sg,gb,dgce,Qar,Qbs,eg",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 8
        * oe.contract(
            "cd,da,rc,gb,sh,dhcg,Qar,Qbs,cd",
            mol.E_ar,
            mol.E_ra,
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 8
        * oe.contract(
            "cd,re,eb,sg,ga,dgce,Qar,Qbs,eg",
            mol.E_ar,
            mol.I_rb,
            mol.A_bb,
            mol.D_ss,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 8
        * oe.contract(
            "cd,re,ff,sb,ea,dfce,Qar,Qbs,ef",
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.F_sb,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 8
        * oe.contract(
            "ra,cb,sd,df,gc,fdgc,Qar,Qbs,cd",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 8
        * oe.contract(
            "ra,cc,ee,sg,hb,gehc,Qar,Qbs,ce",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 8
        * oe.contract(
            "ra,cc,ef,gb,sg,cfge,Qar,Qbs,gc",
            mol.E_ra,
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 8
        * oe.contract(
            "ca,dd,rc,gh,sb,chdg,Qar,Qbs,dc",
            mol.E_ra,
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 8
        * oe.contract(
            "rc,ca,ee,gh,sb,ehcg,Qar,Qbs,ce",
            mol.E_ra,
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 8
        * oe.contract(
            "cb,da,rc,fg,sd,cgdf,Qar,Qbs,dc",
            mol.I_rb,
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 8
        * oe.contract(
            "ca,rd,ef,fb,se,dfce,Qar,Qbs,ef",
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.F_sb,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
        + 8
        * oe.contract(
            "ca,rd,sb,df,gc,dfcg,Qar,Qbs,cd",
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 8
        * oe.contract(
            "cc,ee,sb,rg,ha,egch,Qar,Qbs,ce",
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_A_ar,
        )
        + 8
        * oe.contract(
            "rc,de,sb,fd,ea,cefd,Qar,Qbs,de",
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            mol.Qar,
            mol.Qbs,
            xt_B_bs,
        )
    )

    return x2_exch_disp_sinf_term1
