import numpy as np
import opt_einsum as oe

from prop_sapt import Dimer

from .exch_disp_sinf_terms import (
    get_exch_disp_density_ra,
    get_exch_disp_density_sb,
)


def get_exch_disp_density(mol: Dimer, monomer: str) -> np.ndarray:

    theta_sinf_t_abrs = (
        -2
        * oe.contract("cd,eb,sf,bs->dfce", mol.E_ar, mol.A_bb, mol.D_ss, mol.omegaA_bs)
        - 2
        * oe.contract("ca,rd,ef,ar->dfce", mol.B_aa, mol.C_rr, mol.F_bs, mol.omegaB_ar)
        - oe.contract("cd,ea,rf,ar->dfec", mol.I_br, mol.B_aa, mol.G_rs, mol.omegaB_ar)
        - oe.contract("cb,sd,ef,bs->dfec", mol.A_bb, mol.G_sr, mol.J_as, mol.omegaA_bs)
        + oe.contract("cd,se,fb,bs->defc", mol.I_br, mol.D_ss, mol.H_ab, mol.omegaA_bs)
        + oe.contract("rc,da,ef,ar->cfed", mol.C_rr, mol.H_ba, mol.J_as, mol.omegaB_ar)
        + 2
        * oe.contract("cd,re,fa,ar->decf", mol.E_ar, mol.G_rs, mol.H_ba, mol.omegaB_ar)
        + 2
        * oe.contract("cd,se,fb,bs->edfc", mol.F_bs, mol.G_sr, mol.H_ab, mol.omegaA_bs)
        - 4
        * oe.contract(
            "cd,sb,re,fa,Qar,Qbs->decf",
            mol.E_ar,
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            mol.Qar,
            mol.Qbs,
        )
        - 4
        * oe.contract(
            "ra,cd,se,fb,Qar,Qbs->edfc",
            mol.E_ra,
            mol.F_bs,
            mol.G_sr,
            mol.H_ab,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "cd,rb,se,fa,Qar,Qbs->decf",
            mol.E_ar,
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "ra,cd,se,fb,Qar,Qbs->defc",
            mol.E_ra,
            mol.I_br,
            mol.D_ss,
            mol.H_ab,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "rc,de,fb,sa,Qar,Qbs->cefd",
            mol.C_rr,
            mol.F_bs,
            mol.H_ab,
            mol.J_sa,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "rc,sb,da,ef,Qar,Qbs->cfed",
            mol.C_rr,
            mol.F_sb,
            mol.H_ba,
            mol.J_as,
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cd,re,fb,sa,Qar,Qbs->defc",
            mol.I_br,
            mol.G_rs,
            mol.H_ab,
            mol.J_sa,
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rb,sc,da,ef,Qar,Qbs->cfed",
            mol.I_rb,
            mol.G_sr,
            mol.H_ba,
            mol.J_as,
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "cb,da,se,rf,Qar,Qbs->efdc",
            mol.A_bb,
            mol.B_aa,
            mol.G_sr,
            mol.G_rs,
            mol.Qar,
            mol.Qbs,
        )
        - oe.contract(
            "rc,sd,ea,fb,Qar,Qbs->cdfe",
            mol.C_rr,
            mol.D_ss,
            mol.H_ba,
            mol.H_ab,
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "cd,rb,ea,sf,Qar,Qbs->dfec",
            mol.I_br,
            mol.I_rb,
            mol.B_aa,
            mol.D_ss,
            mol.Qar,
            mol.Qbs,
        )
        + oe.contract(
            "cb,rd,ef,sa,Qar,Qbs->dfec",
            mol.A_bb,
            mol.C_rr,
            mol.J_as,
            mol.J_sa,
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cd,eb,rf,sa,Qar,Qbs->dfce",
            mol.E_ar,
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "ra,cb,sd,ef,Qar,Qbs->dfec",
            mol.E_ra,
            mol.A_bb,
            mol.G_sr,
            mol.J_as,
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cd,ea,sb,rf,Qar,Qbs->dfec",
            mol.I_br,
            mol.B_aa,
            mol.F_sb,
            mol.G_rs,
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "rb,ca,de,sf,Qar,Qbs->fecd",
            mol.I_rb,
            mol.B_aa,
            mol.F_bs,
            mol.G_sr,
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "cb,da,re,sf,Qar,Qbs->efdc",
            mol.A_bb,
            mol.B_aa,
            mol.C_rr,
            mol.D_ss,
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "sc,rd,ea,fb,Qar,Qbs->cdfe",
            mol.G_sr,
            mol.G_rs,
            mol.H_ba,
            mol.H_ab,
            mol.Qar,
            mol.Qbs,
        )
        + 4
        * oe.contract(
            "cd,ra,eb,sf,Qar,Qbs->dfce",
            mol.E_ar,
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            mol.Qar,
            mol.Qbs,
        )
        + 4
        * oe.contract(
            "ca,rd,ef,sb,Qar,Qbs->dfce",
            mol.B_aa,
            mol.C_rr,
            mol.F_bs,
            mol.F_sb,
            mol.Qar,
            mol.Qbs,
        )
    )
    # right now theta_t_abrs is rsab
    # but we make it abrs here
    theta_sinf_t_abrs = oe.contract("rsab,rsab->abrs", theta_sinf_t_abrs, mol.e_rsab)

    rho_MO_exch_disp = np.zeros((mol.nmo, mol.nmo))

    if monomer == "A":

        rho_MO_exch_disp_ra = get_exch_disp_density_ra(mol, theta_sinf_t_abrs)
        rho_MO_exch_disp[mol.slices["r"], mol.slices["a"]] = rho_MO_exch_disp_ra
        rho_MO_exch_disp[mol.slices["a"], mol.slices["r"]] = rho_MO_exch_disp_ra.T

        rho_MO_exch_disp[mol.slices["a"], mol.slices["a"]] = -oe.contract(
            "rsAb,abrs->aA", mol.t_rsab, theta_sinf_t_abrs
        )
        rho_MO_exch_disp[mol.slices["r"], mol.slices["r"]] = oe.contract(
            "Rsab,abrs->Rr", mol.t_rsab, theta_sinf_t_abrs
        )

    if monomer == "B":

        rho_MO_exch_disp_sb = get_exch_disp_density_sb(mol)
        rho_MO_exch_disp[mol.slices["s"], mol.slices["b"]] = rho_MO_exch_disp_sb
        rho_MO_exch_disp[mol.slices["b"], mol.slices["s"]] = rho_MO_exch_disp_sb.T

        rho_MO_exch_disp[mol.slices["b"], mol.slices["b"]] = -oe.contract(
            "rsaB,abrs->bB", mol.t_rsab, theta_sinf_t_abrs
        )
        rho_MO_exch_disp[mol.slices["s"], mol.slices["s"]] = oe.contract(
            "rSab,abrs->Ss", mol.t_rsab, theta_sinf_t_abrs
        )

    return rho_MO_exch_disp


def get_exch_disp_s2_density(mol: Dimer, monomer: str) -> np.ndarray:

    s_ab = mol.s("ab")
    s_ba = s_ab.T
    s_as = mol.s("as")
    s_sa = s_as.T
    s_rs = mol.s("rs")
    s_sr = s_rs.T
    s_rb = mol.s("rb")
    s_br = s_rb.T

    theta_t_abrs = (
        -2 * oe.contract("br,ab,cs->rsac", s_br, s_ab, mol.omegaA_bs)
        + oe.contract("br,ac,cs->rsab", s_br, s_ab, mol.omegaA_bs)
        - oe.contract("sr,ac,bs->rcab", s_sr, s_as, mol.omegaA_bs)
        - 2 * oe.contract("as,ba,cr->rscb", s_as, s_ba, mol.omegaB_ar)
        + oe.contract("as,bc,cr->rsab", s_as, s_ba, mol.omegaB_ar)
        - oe.contract("br,cs,ac->rsab", s_br, s_rs, mol.omegaB_ar)
        - oe.contract("ba,cd,Qar,Qds->rscb", s_ba, s_ab, mol.Qar, mol.Qbs)
        + 2 * oe.contract("ba,cb,Qar,Qes->rsce", s_ba, s_ab, mol.Qar, mol.Qbs)
        + 2 * oe.contract("ba,ac,Qdr,Qcs->rsdb", s_ba, s_ab, mol.Qar, mol.Qbs)
        - 2 * oe.contract("br,cb,Qac,Qds->rsad", s_br, s_rb, mol.Qar, mol.Qbs)
        + oe.contract("br,cd,Qac,Qds->rsab", s_br, s_rb, mol.Qar, mol.Qbs)
        - oe.contract("sr,cd,Qac,Qbs->rdab", s_sr, s_rs, mol.Qar, mol.Qbs)
        - 2 * oe.contract("as,ca,Qdr,Qbc->rsdb", s_as, s_sa, mol.Qar, mol.Qbs)
        + oe.contract("as,cd,Qdr,Qbc->rsab", s_as, s_sa, mol.Qar, mol.Qbs)
    )  # right now theta_t_abrs is rsab
    # but we make it abrs here
    theta_t_abrs = oe.contract("rsab,rsab->abrs", theta_t_abrs, mol.e_rsab)

    rho_MO_exch_disp = np.zeros((mol.nmo, mol.nmo))

    if monomer == "A":

        rho_MO_exch_disp_ra = 0.5 * (
            # < R(X) | V P2 R(V) >
            -oe.contract("ba,cd,rscb,ds->ra", s_ba, s_ab, mol.t_rsab, mol.omegaA_bs)
            + 2
            * oe.contract("ba,cb,rsce,es->ra", s_ba, s_ab, mol.t_rsab, mol.omegaA_bs)
            + 2
            * oe.contract("bc,ce,rsab,es->ra", s_ba, s_ab, mol.t_rsab, mol.omegaA_bs)
            - 2
            * oe.contract("bc,rb,csae,es->ra", s_br, s_rb, mol.t_rsab, mol.omegaA_bs)
            + oe.contract("bc,rd,csab,ds->ra", s_br, s_rb, mol.t_rsab, mol.omegaA_bs)
            - oe.contract("sc,rd,cdab,bs->ra", s_sr, s_rs, mol.t_rsab, mol.omegaA_bs)
            - 2
            * oe.contract("cs,dc,rsab,bd->ra", s_as, s_sa, mol.t_rsab, mol.omegaA_bs)
            + oe.contract("cs,da,rscb,bd->ra", s_as, s_sa, mol.t_rsab, mol.omegaA_bs)
            - oe.contract("cs,bd,rscb,da->ra", s_as, s_ba, mol.t_rsab, mol.omegaB_aa)
            + 2
            * oe.contract("cs,bc,rseb,ea->ra", s_as, s_ba, mol.t_rsab, mol.omegaB_aa)
            + oe.contract("bc,rs,cseb,ea->ra", s_br, s_rs, mol.t_rsab, mol.omegaB_aa)
            - 2
            * oe.contract("rs,ba,csdb,dc->ra", s_rs, s_ba, mol.t_rsab, mol.omegaB_ar)
            - 2
            * oe.contract("cs,bd,rsab,dc->ra", s_rs, s_ba, mol.t_rsab, mol.omegaB_ar)
            + oe.contract("rs,bc,dsab,cd->ra", s_rs, s_ba, mol.t_rsab, mol.omegaB_ar)
            + oe.contract("cs,ba,rsdb,dc->ra", s_rs, s_ba, mol.t_rsab, mol.omegaB_ar)
            - 2
            * oe.contract("cs,bc,esab,re->ra", s_as, s_ba, mol.t_rsab, mol.omegaB_rr)
            + oe.contract("cs,ba,dscb,rd->ra", s_as, s_ba, mol.t_rsab, mol.omegaB_rr)
            - oe.contract("bc,ds,csab,rd->ra", s_br, s_rs, mol.t_rsab, mol.omegaB_rr)
            - 2
            * oe.contract(
                "bc,db,rsdf,Qca,Qfs->ra", s_ba, s_ab, mol.t_rsab, mol.Qaa, mol.Qbs
            )
            - 2
            * oe.contract(
                "bc,ce,rsfb,Qfa,Qes->ra", s_ba, s_ab, mol.t_rsab, mol.Qaa, mol.Qbs
            )
            + oe.contract(
                "bc,de,rsdb,Qca,Qes->ra", s_ba, s_ab, mol.t_rsab, mol.Qaa, mol.Qbs
            )
            - oe.contract(
                "bc,rd,csfb,Qfa,Qds->ra", s_br, s_rb, mol.t_rsab, mol.Qaa, mol.Qbs
            )
            + 2
            * oe.contract(
                "bc,rb,csef,Qea,Qfs->ra", s_br, s_rb, mol.t_rsab, mol.Qaa, mol.Qbs
            )
            + oe.contract(
                "sc,rd,cdgb,Qga,Qbs->ra", s_sr, s_rs, mol.t_rsab, mol.Qaa, mol.Qbs
            )
            - oe.contract(
                "cs,de,rscb,Qea,Qbd->ra", s_as, s_sa, mol.t_rsab, mol.Qaa, mol.Qbs
            )
            + 2
            * oe.contract(
                "cs,dc,rsfb,Qfa,Qbd->ra", s_as, s_sa, mol.t_rsab, mol.Qaa, mol.Qbs
            )
            - 4
            * oe.contract(
                "bc,db,rsae,Qcd,Qes->ra", s_ba, s_rb, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - oe.contract(
                "ba,cd,rseb,Qec,Qds->ra", s_ba, s_rb, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - oe.contract(
                "bc,rd,esab,Qce,Qds->ra", s_ba, s_rb, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "ba,cb,rsde,Qdc,Qes->ra", s_ba, s_rb, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "ba,rc,dseb,Qed,Qcs->ra", s_ba, s_rb, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "bc,rb,dsae,Qcd,Qes->ra", s_ba, s_rb, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "bc,de,rsab,Qcd,Qes->ra", s_ba, s_rb, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "rs,ca,dseb,Qed,Qbc->ra", s_rs, s_sa, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "cs,de,rsab,Qec,Qbd->ra", s_rs, s_sa, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + oe.contract(
                "rs,cd,esab,Qde,Qbc->ra", s_rs, s_sa, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + oe.contract(
                "cs,da,rseb,Qec,Qbd->ra", s_rs, s_sa, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 4
            * oe.contract(
                "bc,db,csdg,Qar,Qgs->ra", s_br, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "bc,de,csdb,Qar,Qes->ra", s_br, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "sc,de,cedb,Qar,Qbs->ra", s_sr, s_as, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - oe.contract(
                "ba,cd,escb,Qre,Qds->ra", s_ba, s_ab, mol.t_rsab, mol.Qrr, mol.Qbs
            )
            + 2
            * oe.contract(
                "ba,cb,dscf,Qrd,Qfs->ra", s_ba, s_ab, mol.t_rsab, mol.Qrr, mol.Qbs
            )
            + 2
            * oe.contract(
                "bc,ce,fsab,Qrf,Qes->ra", s_ba, s_ab, mol.t_rsab, mol.Qrr, mol.Qbs
            )
            - 2
            * oe.contract(
                "bc,db,csaf,Qrd,Qfs->ra", s_br, s_rb, mol.t_rsab, mol.Qrr, mol.Qbs
            )
            + oe.contract(
                "bc,de,csab,Qrd,Qes->ra", s_br, s_rb, mol.t_rsab, mol.Qrr, mol.Qbs
            )
            - oe.contract(
                "sc,de,ceab,Qrd,Qbs->ra", s_sr, s_rs, mol.t_rsab, mol.Qrr, mol.Qbs
            )
            - 2
            * oe.contract(
                "cs,dc,fsab,Qrf,Qbd->ra", s_as, s_sa, mol.t_rsab, mol.Qrr, mol.Qbs
            )
            + oe.contract(
                "cs,da,escb,Qre,Qbd->ra", s_as, s_sa, mol.t_rsab, mol.Qrr, mol.Qbs
            )
            # < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
            + oe.contract("QRr,Qbs,abRs->ra", mol.Qrr, mol.Qbs, theta_t_abrs)
            - oe.contract("QaA,Qbs,Abrs->ra", mol.Qaa, mol.Qbs, theta_t_abrs)
            # < V P2 R(X) R(V) >
            - 2
            * oe.contract("br,as,csdb,dc->ra", s_br, s_as, mol.t_rsab, mol.omegaB_ar)
            + oe.contract("br,cs,dscb,ad->ra", s_br, s_as, mol.t_rsab, mol.omegaB_ar)
            + oe.contract("bc,as,cseb,er->ra", s_br, s_as, mol.t_rsab, mol.omegaB_ar)
            - 4
            * oe.contract(
                "bc,db,csdg,Qar,Qgs->ra", s_br, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - oe.contract(
                "br,cd,escb,Qae,Qds->ra", s_br, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - oe.contract(
                "bc,ad,csfb,Qfr,Qds->ra", s_br, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "br,cb,dscf,Qad,Qfs->ra", s_br, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "br,ac,dseb,Qed,Qcs->ra", s_br, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "bc,ab,csef,Qer,Qfs->ra", s_br, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "bc,de,csdb,Qar,Qes->ra", s_br, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "sr,ac,dcfb,Qfd,Qbs->ra", s_sr, s_as, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "sc,de,cedb,Qar,Qbs->ra", s_sr, s_as, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + oe.contract(
                "sr,cd,edcb,Qae,Qbs->ra", s_sr, s_as, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + oe.contract(
                "sc,ad,cdgb,Qgr,Qbs->ra", s_sr, s_as, mol.t_rsab, mol.Qar, mol.Qbs
            )
        )

        rho_MO_exch_disp_ra = mol.cpscf("A", perturbation=rho_MO_exch_disp_ra.T)

        rho_MO_exch_disp[mol.slices["r"], mol.slices["a"]] = rho_MO_exch_disp_ra
        rho_MO_exch_disp[mol.slices["a"], mol.slices["r"]] = rho_MO_exch_disp_ra.T

        rho_MO_exch_disp[mol.slices["a"], mol.slices["a"]] = -oe.contract(
            "rsAb,abrs->aA", mol.t_rsab, theta_t_abrs
        )
        rho_MO_exch_disp[mol.slices["r"], mol.slices["r"]] = oe.contract(
            "Rsab,abrs->Rr", mol.t_rsab, theta_t_abrs
        )

    if monomer == "B":

        rho_MO_exch_disp_sb = 0.5 * (
            # < R(X) | V P2 R(V) >
            -oe.contract("cr,ad,rsac,db->sb", s_br, s_ab, mol.t_rsab, mol.omegaA_bb)
            + 2
            * oe.contract("cr,ac,rsae,eb->sb", s_br, s_ab, mol.t_rsab, mol.omegaA_bb)
            + oe.contract("sr,ac,rcae,eb->sb", s_sr, s_as, mol.t_rsab, mol.omegaA_bb)
            - 2
            * oe.contract("sr,ab,rcad,dc->sb", s_sr, s_ab, mol.t_rsab, mol.omegaA_bs)
            - 2
            * oe.contract("cr,ad,rsab,dc->sb", s_sr, s_ab, mol.t_rsab, mol.omegaA_bs)
            + oe.contract("sr,ac,rdab,cd->sb", s_sr, s_ab, mol.t_rsab, mol.omegaA_bs)
            + oe.contract("cr,ab,rsad,dc->sb", s_sr, s_ab, mol.t_rsab, mol.omegaA_bs)
            - 2
            * oe.contract("cr,ac,reab,se->sb", s_br, s_ab, mol.t_rsab, mol.omegaA_ss)
            + oe.contract("cr,ab,rdac,sd->sb", s_br, s_ab, mol.t_rsab, mol.omegaA_ss)
            - oe.contract("cr,ad,rdab,sc->sb", s_sr, s_as, mol.t_rsab, mol.omegaA_ss)
            - oe.contract("ca,db,rsdc,ar->sb", s_ba, s_ab, mol.t_rsab, mol.omegaB_ar)
            + 2
            * oe.contract("ca,ab,rsdc,dr->sb", s_ba, s_ab, mol.t_rsab, mol.omegaB_ar)
            + 2
            * oe.contract("ca,dc,rsdb,ar->sb", s_ba, s_ab, mol.t_rsab, mol.omegaB_ar)
            - 2
            * oe.contract("cr,dc,rsab,ad->sb", s_br, s_rb, mol.t_rsab, mol.omegaB_ar)
            + oe.contract("cr,db,rsac,ad->sb", s_br, s_rb, mol.t_rsab, mol.omegaB_ar)
            - oe.contract("sr,cd,rdab,ac->sb", s_sr, s_rs, mol.t_rsab, mol.omegaB_ar)
            - 2
            * oe.contract("ac,sa,rceb,er->sb", s_as, s_sa, mol.t_rsab, mol.omegaB_ar)
            + oe.contract("ac,sd,rcab,dr->sb", s_as, s_sa, mol.t_rsab, mol.omegaB_ar)
            - 2
            * oe.contract(
                "ca,dc,rsdg,Qar,Qgb->sb", s_ba, s_ab, mol.t_rsab, mol.Qar, mol.Qbb
            )
            - 2
            * oe.contract(
                "ca,ad,rsec,Qer,Qdb->sb", s_ba, s_ab, mol.t_rsab, mol.Qar, mol.Qbb
            )
            + oe.contract(
                "ca,de,rsdc,Qar,Qeb->sb", s_ba, s_ab, mol.t_rsab, mol.Qar, mol.Qbb
            )
            - oe.contract(
                "cr,de,rsac,Qad,Qeb->sb", s_br, s_rb, mol.t_rsab, mol.Qar, mol.Qbb
            )
            + 2
            * oe.contract(
                "cr,dc,rsaf,Qad,Qfb->sb", s_br, s_rb, mol.t_rsab, mol.Qar, mol.Qbb
            )
            + oe.contract(
                "sr,cd,rdaf,Qac,Qfb->sb", s_sr, s_rs, mol.t_rsab, mol.Qar, mol.Qbb
            )
            - oe.contract(
                "ac,sd,rcaf,Qdr,Qfb->sb", s_as, s_sa, mol.t_rsab, mol.Qar, mol.Qbb
            )
            + 2
            * oe.contract(
                "ac,sa,rcef,Qer,Qfb->sb", s_as, s_sa, mol.t_rsab, mol.Qar, mol.Qbb
            )
            - 4
            * oe.contract(
                "ca,ad,rseb,Qer,Qdc->sb", s_sa, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - oe.contract(
                "sa,cd,recb,Qar,Qde->sb", s_sa, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - oe.contract(
                "ca,db,rsdf,Qar,Qfc->sb", s_sa, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "sa,cb,rdcf,Qar,Qfd->sb", s_sa, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "sa,ac,rdeb,Qer,Qcd->sb", s_sa, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "ca,ab,rsde,Qdr,Qec->sb", s_sa, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "ca,de,rsdb,Qar,Qec->sb", s_sa, s_ab, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "sr,cb,rdae,Qac,Qed->sb", s_sr, s_rb, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "cr,de,rsab,Qad,Qec->sb", s_sr, s_rb, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + oe.contract(
                "sr,cd,reab,Qac,Qde->sb", s_sr, s_rb, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + oe.contract(
                "cr,db,rsae,Qad,Qec->sb", s_sr, s_rb, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 4
            * oe.contract(
                "ac,da,rcfd,Qfr,Qbs->sb", s_as, s_ba, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "ac,de,rcad,Qer,Qbs->sb", s_as, s_ba, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "cr,de,reac,Qad,Qbs->sb", s_br, s_rs, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - oe.contract(
                "ca,db,redc,Qar,Qse->sb", s_ba, s_ab, mol.t_rsab, mol.Qar, mol.Qss
            )
            + 2
            * oe.contract(
                "ca,ab,rdec,Qer,Qsd->sb", s_ba, s_ab, mol.t_rsab, mol.Qar, mol.Qss
            )
            + 2
            * oe.contract(
                "ca,dc,rfdb,Qar,Qsf->sb", s_ba, s_ab, mol.t_rsab, mol.Qar, mol.Qss
            )
            - 2
            * oe.contract(
                "cr,dc,rfab,Qad,Qsf->sb", s_br, s_rb, mol.t_rsab, mol.Qar, mol.Qss
            )
            + oe.contract(
                "cr,db,reac,Qad,Qse->sb", s_br, s_rb, mol.t_rsab, mol.Qar, mol.Qss
            )
            - oe.contract(
                "cr,de,reab,Qad,Qsc->sb", s_sr, s_rs, mol.t_rsab, mol.Qar, mol.Qss
            )
            - 2
            * oe.contract(
                "ac,da,rcfb,Qfr,Qsd->sb", s_as, s_sa, mol.t_rsab, mol.Qar, mol.Qss
            )
            + oe.contract(
                "ac,de,rcab,Qer,Qsd->sb", s_as, s_sa, mol.t_rsab, mol.Qar, mol.Qss
            )
            # < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
            + oe.contract("Qar,QSs,abrS->sb", mol.Qar, mol.Qss, theta_t_abrs)
            - oe.contract("Qar,QbB,aBrs->sb", mol.Qar, mol.Qbb, theta_t_abrs)
            # < V P2 R(X) R(V) >
            - 2
            * oe.contract("br,as,rcad,dc->sb", s_br, s_as, mol.t_rsab, mol.omegaA_bs)
            + oe.contract("br,ac,rcae,es->sb", s_br, s_as, mol.t_rsab, mol.omegaA_bs)
            + oe.contract("cr,as,rdac,bd->sb", s_br, s_as, mol.t_rsab, mol.omegaA_bs)
            - 4
            * oe.contract(
                "ac,da,rcfd,Qfr,Qbs->sb", s_as, s_ba, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - oe.contract(
                "as,cd,reac,Qdr,Qbe->sb", s_as, s_ba, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - oe.contract(
                "ac,bd,rcaf,Qdr,Qfs->sb", s_as, s_ba, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "as,ca,rdec,Qer,Qbd->sb", s_as, s_ba, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "as,bc,rdae,Qcr,Qed->sb", s_as, s_ba, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "ac,ba,rcef,Qer,Qfs->sb", s_as, s_ba, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "ac,de,rcad,Qer,Qbs->sb", s_as, s_ba, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "br,cs,rdae,Qac,Qed->sb", s_br, s_rs, mol.t_rsab, mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "cr,de,reac,Qad,Qbs->sb", s_br, s_rs, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + oe.contract(
                "br,cd,rdaf,Qac,Qfs->sb", s_br, s_rs, mol.t_rsab, mol.Qar, mol.Qbs
            )
            + oe.contract(
                "cr,ds,reac,Qad,Qbe->sb", s_br, s_rs, mol.t_rsab, mol.Qar, mol.Qbs
            )
        )

        rho_MO_exch_disp_sb = mol.cpscf("B", perturbation=rho_MO_exch_disp_sb.T)

        rho_MO_exch_disp[mol.slices["s"], mol.slices["b"]] = rho_MO_exch_disp_sb
        rho_MO_exch_disp[mol.slices["b"], mol.slices["s"]] = rho_MO_exch_disp_sb.T

        rho_MO_exch_disp[mol.slices["b"], mol.slices["b"]] = -oe.contract(
            "rsaB,abrs->bB", mol.t_rsab, theta_t_abrs
        )
        rho_MO_exch_disp[mol.slices["s"], mol.slices["s"]] = oe.contract(
            "rSab,abrs->Ss", mol.t_rsab, theta_t_abrs
        )

    return rho_MO_exch_disp
