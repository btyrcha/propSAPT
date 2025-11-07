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
            "ar,bc,ds,rsab,cd", mol.E_ar, mol.A_bb, mol.D_ss, mol.t_rsab, vrx_bs
        )
        - 4
        * oe.contract(
            "ac,dr,bs,rsab,cd", mol.B_aa, mol.C_rr, mol.F_bs, mol.t_rsab, vrx_ar
        )
        - 2
        * oe.contract(
            "bc,ad,rs,csab,dr", mol.I_br, mol.B_aa, mol.G_rs, mol.t_rsab, vrx_ar
        )
        - 2
        * oe.contract(
            "bc,sr,ad,rdab,cs", mol.A_bb, mol.G_sr, mol.J_as, mol.t_rsab, vrx_bs
        )
        + 2
        * oe.contract(
            "cr,ds,ab,rsac,bd", mol.I_br, mol.D_ss, mol.H_ab, mol.t_rsab, vrx_bs
        )
        + 2
        * oe.contract(
            "cr,ba,ds,rsdb,ac", mol.C_rr, mol.H_ba, mol.J_as, mol.t_rsab, vrx_ar
        )
        + 4
        * oe.contract(
            "ar,cs,bd,rsab,dc", mol.E_ar, mol.G_rs, mol.H_ba, mol.t_rsab, vrx_ar
        )
        + 4
        * oe.contract(
            "bs,cr,ad,rsab,dc", mol.F_bs, mol.G_sr, mol.H_ab, mol.t_rsab, vrx_bs
        )
        - 8
        * oe.contract(
            "ar,sb,cd,ef,rdae,fbcs",
            mol.E_ar,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.t_rsab,
            vrx_abrs,
        )
        - 8
        * oe.contract(
            "ra,bs,cd,ef,dseb,afrc",
            mol.E_ra,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.t_rsab,
            vrx_abrs,
        )
        - 4
        * oe.contract(
            "ar,cd,es,bf,rsab,fdce",
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.t_rsab,
            vrx_abrs,
        )
        - 4
        * oe.contract(
            "ra,cd,es,fb,dsfc,abre",
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.t_rsab,
            vrx_abrs,
        )
        - 4
        * oe.contract(
            "cr,bs,ad,ef,rsab,fdce",
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            vrx_abrs,
        )
        - 4
        * oe.contract(
            "cr,sb,da,ef,rfed,abcs",
            mol.C_rr,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            vrx_abrs,
        )
        - 2
        * oe.contract(
            "cd,rs,ab,ef,dsac,fbre",
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.t_rsab,
            vrx_abrs,
        )
        - 2
        * oe.contract(
            "cd,sr,ba,ef,rfeb,adcs",
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.t_rsab,
            vrx_abrs,
        )
        - 2
        * oe.contract(
            "bc,ad,sr,ef,rfab,dces",
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.t_rsab,
            vrx_abrs,
        )
        - 2
        * oe.contract(
            "cr,ds,ba,ef,rseb,afcd",
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            vrx_abrs,
        )
        + 2
        * oe.contract(
            "br,cd,ae,fs,rsab,edcf",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.t_rsab,
            vrx_abrs,
        )
        + 2
        * oe.contract(
            "bc,dr,as,ef,rsab,fcde",
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "ar,bc,ds,ef,rsab,fcde",
            mol.E_ar,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "ra,bc,sd,ef,dfeb,acrs",
            mol.E_ra,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "cd,ae,sb,rf,dfac,ebrs",
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.G_rs,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "cd,ae,bs,fr,rsab,edcf",
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "bc,ad,er,fs,rsab,dcef",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.t_rsab,
            vrx_abrs,
        )
        + 4
        * oe.contract(
            "sr,cd,ba,ef,rdeb,afcs",
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.t_rsab,
            vrx_abrs,
        )
        + 8
        * oe.contract(
            "ar,cd,be,fs,rsab,decf",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.t_rsab,
            vrx_abrs,
        )
        + 8
        * oe.contract(
            "ac,dr,bs,ef,rsab,cfde",
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.t_rsab,
            vrx_abrs,
        )
        - 4
        * oe.contract(
            "ar,cd,es,bf,fc,rsab,de",
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
            "ar,cd,se,bf,ds,reab,fc",
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
            "ar,cb,ds,ef,be,rsac,fd",
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
            "ar,bc,sd,ef,fs,rdab,ce",
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
            "cd,ea,fr,bs,ac,rseb,df",
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
            "cd,ae,rf,bs,dr,fsab,ec",
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
            "cb,sd,er,af,bs,rdac,fe",
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
            "bc,ds,er,af,fe,rsab,cd",
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
            "ca,rd,es,bf,ar,dscb,fe",
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
            "ac,dr,es,bf,fe,rsab,cd",
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
            "cr,bs,ad,ef,de,rsab,fc",
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        - 4
        * oe.contract(
            "rc,bs,ad,ef,fr,csab,de",
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        - 2
        * oe.contract(
            "rc,bd,ea,fs,ar,dseb,cf",
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
            "rc,bd,ae,fs,cf,dsab,er",
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
            "cd,rs,ab,ef,be,dsac,fr",
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
            "cd,rs,ab,ef,fr,dsac,be",
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
            "cd,sr,ba,ef,ds,rfeb,ac",
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
            "cd,sr,ba,ef,ac,rfeb,ds",
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
            "cb,ad,sr,ef,bs,rfac,de",
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
            "cb,sd,er,af,bs,rfac,de",
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
            "bc,da,sr,ef,ae,rfdb,cs",
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
            "bc,sd,er,af,de,rfab,cs",
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
            "cr,sd,ba,ef,fs,rdeb,ac",
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.omegaA_bs,
            mol.t_rsab,
            xt_A_ar,
        )
        - 2
        * oe.contract(
            "rc,ds,ba,ef,ar,cseb,fd",
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.omegaB_ar,
            mol.t_rsab,
            xt_B_bs,
        )
        + 2
        * oe.contract(
            "ca,dr,be,fs,ec,rsfb,ad",
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
            "ca,rd,be,fs,ar,dsfb,ec",
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
            "br,cd,ea,fs,ac,rseb,df",
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
            "br,cd,ae,sf,ds,rfab,ec",
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
            "cr,ds,eb,af,fe,rsac,bd",
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
            "cr,sd,eb,af,bs,rdac,fe",
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
            "cb,dr,as,ef,be,rsac,fd",
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
            "cb,ds,er,af,be,rsac,fd",
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
            "bc,rd,as,ef,fr,dsab,ce",
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
            "bc,sd,er,af,fs,rdab,ce",
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
            "ca,dr,es,bf,ae,rscb,fd",
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
            "ac,rd,es,bf,fr,dsab,ce",
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
            "ar,cd,es,bf,fc,rsab,de",
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
            "ar,cd,es,bf,de,rsab,fc",
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
            "ar,cb,ds,ef,be,rsac,fd",
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
            "ar,bc,ds,ef,fd,rsab,ce",
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
            "cd,ea,bs,fr,ac,rseb,df",
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
            "cd,ae,bs,fr,df,rsab,ec",
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
            "cb,ad,er,sf,bs,rfac,de",
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
            "bc,da,re,fs,ar,esdb,cf",
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
            "bs,cd,er,af,fc,rsab,de",
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
            "bs,cd,er,af,de,rsab,fc",
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
            "sr,cd,ba,ef,fs,rdeb,ac",
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
            "sr,cd,ba,ef,ac,rdeb,fs",
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
            "ar,cd,be,fs,gh,rsab,Qdc,Qeg,hf",
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
            "ar,cd,sb,ef,gh,rfag,Qhc,Qbs,de",
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
            "ar,cd,sb,ef,gh,rfag,Qde,Qbs,hc",
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
            "ar,cd,es,bf,gh,rsab,Qfe,Qdg,hc",
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
            "ar,bc,sd,ef,gh,rfab,Qhe,Qds,cg",
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
            "ar,cb,sd,ef,gh,rfag,Qhe,Qbs,dc",
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
            "ra,bs,cd,ef,gh,fsgb,Qar,Qhc,de",
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
            "ra,bs,cd,ef,gh,fsgb,Qar,Qde,hc",
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
            "ra,sc,de,bf,gh,cegb,Qar,Qhs,fd",
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
            "ca,rd,bs,ef,gh,fsgb,Qar,Qhe,dc",
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
            "rc,de,af,bs,gh,hsab,Qcr,Qeg,fd",
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
            "cd,be,af,gr,hs,rsab,Qdc,Qeh,fg",
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
            "cd,ae,bs,fg,hr,rsab,Qec,Qgf,dh",
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
            "cd,bs,er,af,gh,rsab,Qhc,Qfe,dg",
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
            "bc,ad,er,fs,gh,rsab,Qde,Qhg,cf",
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
            "sb,cr,de,fa,gh,regf,Qad,Qbs,hc",
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
            "ar,cd,ef,gs,bh,rsab,Qhc,Qfg,de",
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
            "ar,cd,ef,gs,bh,rsab,Qde,Qfg,hc",
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
            "ar,cd,be,fs,gh,rsab,Qhc,Qdf,eg",
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
            "ar,cd,be,fs,gh,rsab,Qhc,Qeg,df",
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
            "ar,cd,es,fb,gh,rsag,Qhc,Qbe,df",
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
            "ar,cd,es,fb,gh,rsag,Qhc,Qdf,be",
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
            "ra,cd,es,fb,gh,dsgc,Qar,Qbe,hf",
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
            "ra,cd,es,fb,gh,dsgc,Qar,Qhf,be",
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
            "ra,bc,ds,ef,gh,fsgb,Qar,Qhd,ce",
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
            "ra,bc,ds,ef,gh,fsgb,Qar,Qce,hd",
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
            "ca,rd,ef,gs,hb,fshe,Qar,Qbg,dc",
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
            "ca,bd,er,fs,gh,rsfb,Qac,Qdg,he",
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
            "ca,dr,bs,ef,gh,rseb,Qad,Qfg,hc",
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
            "ca,dr,bs,ef,gh,rseb,Qhc,Qfg,ad",
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
            "ca,dr,sb,ef,gh,rhge,Qad,Qbs,fc",
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
            "ca,dr,sb,ef,gh,rhge,Qfc,Qbs,ad",
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
            "rc,bd,ef,ag,hs,dsab,Qcr,Qfh,ge",
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
            "cr,de,af,gs,hb,rsac,Qfd,Qbh,eg",
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
            "cr,de,fs,ab,gh,rsac,Qhd,Qbf,eg",
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
            "cd,ae,fr,bs,gh,rsab,Qhf,Qdg,ec",
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
            "cd,ae,fr,bs,gh,rsab,Qec,Qdg,hf",
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
            "cd,er,ba,fs,gh,rsfb,Qae,Qdg,hc",
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
            "cd,be,af,gs,hr,rsab,Qfc,Qeg,dh",
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
            "cd,be,af,gs,hr,rsab,Qfc,Qdh,eg",
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
            "cd,es,fr,ba,gh,rsgb,Qac,Qde,hf",
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
            "cd,es,fr,ba,gh,rsgb,Qac,Qhf,de",
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
            "bc,ad,er,fs,gh,rsab,Qde,Qcg,hf",
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
            "bc,ad,er,fs,gh,rsab,Qhf,Qcg,de",
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
            "bc,dr,se,af,gh,rfab,Qhd,Qes,cg",
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
            "ac,dr,sb,ef,gh,rfag,Qhd,Qbs,ce",
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
            "ac,dr,sb,ef,gh,rfag,Qce,Qbs,hd",
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
            "cr,bs,de,af,gh,rsab,Qhc,Qfd,eg",
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
            "cr,bs,de,af,gh,rsab,Qhc,Qeg,fd",
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
            "cr,db,se,fa,gh,rhgf,Qac,Qbs,ed",
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
            "cr,ds,ba,ef,gh,rseb,Qhc,Qfg,ad",
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
            "cr,ds,ba,ef,gh,rseb,Qad,Qfg,hc",
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
            "ra,cd,es,fb,gh,dsfc,Qhr,Qbg,ae",
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
            "ra,cd,es,fb,gh,dsfc,Qae,Qbg,hr",
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
            "ra,cd,se,bf,gh,ehgb,Qfr,Qds,ac",
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
            "ra,cd,se,bf,gh,ehgb,Qac,Qds,fr",
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
            "ca,dr,es,bf,gh,rsgb,Qad,Qhe,fc",
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
            "ca,dr,es,bf,gh,rsgb,Qfc,Qhe,ad",
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
            "rc,bd,ae,sf,gh,fhab,Qer,Qds,cg",
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
            "rc,bd,ae,sf,gh,fhab,Qcg,Qds,er",
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
            "bc,de,af,rs,gh,csab,Qhr,Qeg,fd",
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
            "bc,de,af,rs,gh,csab,Qfd,Qeg,hr",
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
            "cd,sb,re,af,gh,deac,Qhr,Qfs,bg",
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
            "cd,sb,re,af,gh,deac,Qhr,Qbg,fs",
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
            "cd,ae,fr,gs,bh,rsab,Qhf,Qdg,ec",
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
            "cd,ae,fr,gs,bh,rsab,Qec,Qdg,hf",
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
            "cd,ae,sr,fg,bh,rgab,Qef,Qds,hc",
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
            "cd,ae,sr,fg,bh,rgab,Qhc,Qds,ef",
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
            "cd,be,sr,af,gh,rfab,Qhc,Qds,eg",
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
            "cd,be,sr,af,gh,rfab,Qhc,Qeg,ds",
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
            "cd,sb,er,fa,gh,rhgf,Qac,Qds,be",
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
            "cd,sb,er,fa,gh,rhgf,Qac,Qbe,ds",
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
            "bc,ad,se,fr,gh,rhab,Qdg,Qcs,ef",
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
            "bc,ad,se,fr,gh,rhab,Qdg,Qef,cs",
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
            "bc,dr,es,af,gh,rsab,Qhd,Qfe,cg",
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
            "bc,dr,es,af,gh,rsab,Qhd,Qcg,fe",
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
            "bc,sr,de,af,gh,reab,Qhd,Qcs,fg",
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
            "bc,sr,de,af,gh,reab,Qhd,Qfg,cs",
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
            "cr,ds,eb,fa,gh,rsgf,Qac,Qbd,he",
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
            "cr,ds,eb,fa,gh,rsgf,Qac,Qhe,bd",
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
            "ca,bd,er,fs,gh,rsfb,Qae,Qdg,hc",
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
            "ca,bd,er,fs,gh,rsfb,Qhc,Qdg,ae",
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
            "rc,bd,ef,ag,hs,dsab,Qgr,Qfh,ce",
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
            "rc,bd,ef,ag,hs,dsab,Qce,Qfh,gr",
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
            "cr,de,af,gs,hb,rsac,Qfd,Qbg,eh",
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
            "cr,de,af,gs,hb,rsac,Qfd,Qeh,bg",
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
            "cr,de,fs,ab,gh,rsac,Qhd,Qef,bg",
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
            "cr,de,fs,ab,gh,rsac,Qhd,Qbg,ef",
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
            "cd,er,ba,fs,gh,rsfb,Qhe,Qdg,ac",
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
            "cd,er,ba,fs,gh,rsfb,Qac,Qdg,he",
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
            "cd,be,af,gs,hr,rsab,Qfc,Qdg,eh",
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
            "cd,be,af,gs,hr,rsab,Qfc,Qeh,dg",
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
            "cd,es,fr,ba,gh,rsgb,Qac,Qhe,df",
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
            "cd,es,fr,ba,gh,rsgb,Qac,Qdf,he",
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
            "bc,ad,er,fs,gh,rsab,Qhe,Qcg,df",
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
            "bc,ad,er,fs,gh,rsab,Qdf,Qcg,he",
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
            "bc,dr,se,af,gh,rfab,Qhd,Qcs,eg",
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
            "bc,dr,se,af,gh,rfab,Qhd,Qeg,cs",
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
            "cr,ds,ba,ef,gh,rseb,Qac,Qfg,hd",
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
            "cr,ds,ba,ef,gh,rseb,Qhd,Qfg,ac",
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
            "ar,cd,be,fs,gh,rsab,Qhc,Qeg,df",
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
            "ar,cd,be,fs,gh,rsab,Qdf,Qeg,hc",
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
            "ar,cd,es,bf,gh,rsab,Qhe,Qdg,fc",
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
            "ar,cd,es,bf,gh,rsab,Qfc,Qdg,he",
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
            "ar,bc,sd,ef,gh,rfab,Qhe,Qcs,dg",
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
            "ar,bc,sd,ef,gh,rfab,Qhe,Qdg,cs",
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
            "ra,cd,es,fb,gh,dsfc,Qar,Qbg,he",
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
            "ra,cd,se,bf,gh,ehgb,Qar,Qds,fc",
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
            "ra,bc,sd,ef,gh,fhgb,Qar,Qcs,de",
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
            "ra,bc,sd,ef,gh,fhgb,Qar,Qde,cs",
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
            "ra,sc,de,bf,gh,cegb,Qfr,Qhs,ad",
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
            "ra,sc,de,bf,gh,cegb,Qad,Qhs,fr",
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
            "ca,rd,be,sf,gh,fhgb,Qar,Qes,dc",
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
            "ca,dr,es,bf,gh,rsgb,Qac,Qhe,fd",
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
            "rc,de,af,sb,gh,ehad,Qfr,Qbs,cg",
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
            "rc,de,af,sb,gh,ehad,Qcg,Qbs,fr",
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
            "rc,de,af,bs,gh,hsab,Qfr,Qeg,cd",
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
            "rc,de,af,bs,gh,hsab,Qcd,Qeg,fr",
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
            "rc,bd,ae,sf,gh,fhab,Qcr,Qds,eg",
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
            "cd,be,af,gr,hs,rsab,Qdg,Qeh,fc",
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
            "cd,be,af,gr,hs,rsab,Qfc,Qeh,dg",
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
            "bc,de,af,rs,gh,csab,Qfr,Qeg,hd",
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
            "cd,ae,fb,sg,rh,dhac,Qer,Qbs,gf",
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
            "cd,sb,re,af,gh,deac,Qhr,Qbs,fg",
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
            "cd,ae,fr,gs,bh,rsab,Qef,Qdg,hc",
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
            "cd,ae,fr,gs,bh,rsab,Qhc,Qdg,ef",
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
            "cd,ae,sr,fg,bh,rgab,Qhf,Qds,ec",
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
            "cd,ae,sr,fg,bh,rgab,Qec,Qds,hf",
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
            "cd,be,sr,af,gh,rfab,Qhc,Qes,dg",
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
            "cd,ae,bs,fg,hr,rsab,Qec,Qdf,gh",
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
            "cd,ae,bs,fg,hr,rsab,Qec,Qgh,df",
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
            "cd,bs,er,af,gh,rsab,Qhc,Qde,fg",
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
            "cd,bs,er,af,gh,rsab,Qhc,Qfg,de",
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
            "cd,sb,er,fa,gh,rhgf,Qac,Qbs,de",
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
            "bc,ad,er,fs,gh,rsab,Qde,Qhf,cg",
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
            "bc,ad,er,fs,gh,rsab,Qde,Qcg,hf",
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
            "bc,ad,se,fr,gh,rhab,Qdg,Qes,cf",
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
            "bc,dr,es,af,gh,rsab,Qhd,Qce,fg",
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
            "bc,dr,es,af,gh,rsab,Qhd,Qfg,ce",
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
            "bc,sr,de,af,gh,reab,Qhd,Qfs,cg",
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
            "bc,sr,de,af,gh,reab,Qhd,Qcg,fs",
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
            "cr,ds,eb,fa,gh,rsgf,Qac,Qbe,hd",
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
            "sb,cr,de,fa,gh,regf,Qad,Qhs,bc",
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
            "sb,cr,de,fa,gh,regf,Qad,Qbc,hs",
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
            "ar,cd,ef,gs,bh,rsab,Qdc,Qfg,he",
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
            "ar,cd,be,fs,gh,rsab,Qdc,Qhf,eg",
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
            "ar,cd,be,fs,gh,rsab,Qdc,Qeg,hf",
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
            "ar,cd,ef,bg,hs,rsab,Qde,Qgh,fc",
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
            "ar,cd,be,fs,gh,rsab,Qhc,Qef,dg",
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
            "ar,cd,es,fb,gh,rsag,Qhc,Qbf,de",
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
            "ra,bc,ds,ef,gh,fsgb,Qar,Qcd,he",
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
            "ra,bc,ds,ef,gh,fsgb,Qar,Qhe,cd",
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
            "ca,dr,bs,ef,gh,rseb,Qac,Qfg,hd",
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
            "cd,ae,fr,bs,gh,rsab,Qdf,Qhg,ec",
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
            "cd,ae,fr,bs,gh,rsab,Qec,Qhg,df",
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
            "cd,ae,fr,bs,gh,rsab,Qef,Qdg,hc",
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
            "ac,dr,bs,ef,gh,rsab,Qcd,Qfg,he",
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
            "ac,dr,sb,ef,gh,rfag,Qcd,Qbs,he",
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
            "ac,dr,sb,ef,gh,rfag,Qhe,Qbs,cd",
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
            "cr,bs,de,af,gh,rsab,Qhc,Qed,fg",
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
