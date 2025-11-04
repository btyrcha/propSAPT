import numpy as np
import opt_einsum as oe

from prop_sapt import Dimer

from .exch_ind_sinf_terms import (
    get_exch_ind_density_aa,
    get_exch_ind_density_ra,
    get_exch_ind_density_rr,
    get_exch_ind_density_bb,
    get_exch_ind_density_sb,
    get_exch_ind_density_ss,
)


def get_exch_ind_density(mol: Dimer, monomer: str) -> np.ndarray:

    rho_MO_exch_ind = np.zeros((mol.nmo, mol.nmo))

    if monomer == "A":

        rho_MO_exch_ind_ra = get_exch_ind_density_ra(mol)
        rho_MO_exch_ind[mol.slices["r"], mol.slices["a"]] = rho_MO_exch_ind_ra
        rho_MO_exch_ind[mol.slices["a"], mol.slices["r"]] = rho_MO_exch_ind_ra.T

        rho_MO_exch_ind[mol.slices["a"], mol.slices["a"]] = get_exch_ind_density_aa(mol)
        rho_MO_exch_ind[mol.slices["r"], mol.slices["r"]] = get_exch_ind_density_rr(mol)

    if monomer == "B":

        rho_MO_exch_ind_sb = get_exch_ind_density_sb(mol)
        rho_MO_exch_ind[mol.slices["s"], mol.slices["b"]] = rho_MO_exch_ind_sb
        rho_MO_exch_ind[mol.slices["b"], mol.slices["s"]] = rho_MO_exch_ind_sb.T

        rho_MO_exch_ind[mol.slices["b"], mol.slices["b"]] = get_exch_ind_density_bb(mol)
        rho_MO_exch_ind[mol.slices["s"], mol.slices["s"]] = get_exch_ind_density_ss(mol)

    return rho_MO_exch_ind


def get_exch_ind_s2_density(mol: Dimer, monomer: str) -> np.ndarray:

    s_ab = mol.s("ab")
    s_ba = s_ab.T
    s_as = mol.s("as")
    s_sa = s_as.T
    s_rs = mol.s("rs")
    s_sr = s_rs.T
    s_rb = mol.s("rb")
    s_br = s_rb.T

    theta_t_ra = (
        -oe.contract("sr,ab,bs->ra", s_sr, s_ab, mol.omegaA_bs)
        + oe.contract("ba,cb,ar->rc", s_ba, s_ab, mol.omegaB_ar)
        - oe.contract("br,cb,ac->ra", s_br, s_rb, mol.omegaB_ar)
        - 2 * oe.contract("sa,ab,Qcr,Qbs->rc", s_sa, s_ab, mol.Qar, mol.Qbs)
        + oe.contract("sa,cb,Qar,Qbs->rc", s_sa, s_ab, mol.Qar, mol.Qbs)
        - oe.contract("sr,cb,Qac,Qbs->ra", s_sr, s_rb, mol.Qar, mol.Qbs)
    )
    theta_t_ra = mol.cpscf("A", perturbation=theta_t_ra.T)

    theta_t_sb = (
        +oe.contract("ba,ac,cs->sb", s_ba, s_ab, mol.omegaA_bs)
        - oe.contract("as,ca,bc->sb", s_as, s_sa, mol.omegaA_bs)
        - oe.contract("rs,ba,ar->sb", s_rs, s_ba, mol.omegaB_ar)
        - 2 * oe.contract("ba,rb,Qar,Qcs->sc", s_ba, s_rb, mol.Qar, mol.Qbs)
        + oe.contract("ba,rc,Qar,Qcs->sb", s_ba, s_rb, mol.Qar, mol.Qbs)
        - oe.contract("rs,ca,Qar,Qbc->sb", s_rs, s_sa, mol.Qar, mol.Qbs)
    )
    theta_t_sb = mol.cpscf("B", perturbation=theta_t_sb.T)

    rho_MO_exch_ind = np.zeros((mol.nmo, mol.nmo))

    if monomer == "A":

        rho_MO_exch_ind_ra = 0.5 * (
            # < R(X) | V P2 R(V) >
            oe.contract(
                "sa,cb,rc,bs->ra", s_sa, s_ab, mol.get_cpscf_ra(), mol.omegaA_bs
            )
            - oe.contract(
                "sc,rb,ca,bs->ra", s_sr, s_rb, mol.get_cpscf_ra(), mol.omegaA_bs
            )
            + oe.contract(
                "ba,rc,sb,cs->ra", s_ba, s_rb, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            - oe.contract(
                "rs,ca,sb,bc->ra", s_rs, s_sa, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            - oe.contract(
                "bc,db,rd,ca->ra", s_ba, s_ab, mol.get_cpscf_ra(), mol.omegaB_aa
            )
            + oe.contract(
                "bc,rb,ce,ea->ra", s_br, s_rb, mol.get_cpscf_ra(), mol.omegaB_aa
            )
            + oe.contract(
                "rs,bc,sb,ca->ra", s_rs, s_ba, mol.get_cpscf_sb(), mol.omegaB_aa
            )
            + oe.contract(
                "ba,cb,rd,dc->ra", s_ba, s_rb, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            + oe.contract(
                "bc,rb,da,cd->ra", s_ba, s_rb, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            + oe.contract(
                "ba,cb,dc,rd->ra", s_ba, s_ab, mol.get_cpscf_ra(), mol.omegaB_rr
            )
            - oe.contract(
                "bc,db,ca,rd->ra", s_br, s_rb, mol.get_cpscf_ra(), mol.omegaB_rr
            )
            - oe.contract(
                "cs,ba,sb,rc->ra", s_rs, s_ba, mol.get_cpscf_sb(), mol.omegaB_rr
            )
            - oe.contract(
                "sc,db,rd,Qca,Qbs->ra", s_sa, s_ab, mol.get_cpscf_ra(), mol.Qaa, mol.Qbs
            )
            + 2
            * oe.contract(
                "sc,cb,re,Qea,Qbs->ra", s_sa, s_ab, mol.get_cpscf_ra(), mol.Qaa, mol.Qbs
            )
            + oe.contract(
                "sc,rb,ce,Qea,Qbs->ra", s_sr, s_rb, mol.get_cpscf_ra(), mol.Qaa, mol.Qbs
            )
            - oe.contract(
                "bc,rd,sb,Qca,Qds->ra", s_ba, s_rb, mol.get_cpscf_sb(), mol.Qaa, mol.Qbs
            )
            + 2
            * oe.contract(
                "bc,rb,sd,Qca,Qds->ra", s_ba, s_rb, mol.get_cpscf_sb(), mol.Qaa, mol.Qbs
            )
            + oe.contract(
                "rs,cd,sb,Qda,Qbc->ra", s_rs, s_sa, mol.get_cpscf_sb(), mol.Qaa, mol.Qbs
            )
            - 2
            * oe.contract(
                "sa,rb,cd,Qdc,Qbs->ra", s_sa, s_rb, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "sa,cb,rd,Qdc,Qbs->ra", s_sa, s_rb, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "sc,rb,da,Qcd,Qbs->ra", s_sa, s_rb, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "sc,db,cd,Qar,Qbs->ra", s_sr, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "bc,ce,sb,Qar,Qes->ra", s_ba, s_ab, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "cs,dc,sb,Qar,Qbd->ra", s_as, s_sa, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "sc,cb,ea,Qre,Qbs->ra", s_sa, s_ab, mol.get_cpscf_ra(), mol.Qrr, mol.Qbs
            )
            + oe.contract(
                "sa,cb,dc,Qrd,Qbs->ra", s_sa, s_ab, mol.get_cpscf_ra(), mol.Qrr, mol.Qbs
            )
            - oe.contract(
                "sc,db,ca,Qrd,Qbs->ra", s_sr, s_rb, mol.get_cpscf_ra(), mol.Qrr, mol.Qbs
            )
            - 2
            * oe.contract(
                "ba,cb,sd,Qrc,Qds->ra", s_ba, s_rb, mol.get_cpscf_sb(), mol.Qrr, mol.Qbs
            )
            + oe.contract(
                "ba,cd,sb,Qrc,Qds->ra", s_ba, s_rb, mol.get_cpscf_sb(), mol.Qrr, mol.Qbs
            )
            - oe.contract(
                "cs,da,sb,Qrc,Qbd->ra", s_rs, s_sa, mol.get_cpscf_sb(), mol.Qrr, mol.Qbs
            )
            # < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
            + oe.contract("rR,Ra->ra", mol.omegaB_rr, theta_t_ra)
            - oe.contract("Aa,rA->ra", mol.omegaB_aa, theta_t_ra)
            + 2 * oe.contract("Qar,Qbs,sb->ra", mol.Qar, mol.Qbs, theta_t_sb)
            # < V P2 R(X) R(V) >
            + oe.contract(
                "br,ac,sb,cs->ra", s_br, s_ab, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            - oe.contract(
                "sr,ac,cb,bs->ra", s_sr, s_as, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            + oe.contract(
                "br,cb,dc,ad->ra", s_br, s_ab, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            + oe.contract(
                "bc,ab,ce,er->ra", s_br, s_ab, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            + oe.contract(
                "as,bc,sb,cr->ra", s_as, s_ba, mol.get_cpscf_sb(), mol.omegaB_ar
            )
            - oe.contract(
                "br,cs,sb,ac->ra", s_br, s_rs, mol.get_cpscf_sb(), mol.omegaB_ar
            )
            - 2
            * oe.contract(
                "sr,ab,cd,Qdc,Qbs->ra", s_sr, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "sc,db,cd,Qar,Qbs->ra", s_sr, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "sr,cb,dc,Qad,Qbs->ra", s_sr, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "sc,ab,ce,Qer,Qbs->ra", s_sr, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            - oe.contract(
                "bc,ad,sb,Qcr,Qds->ra", s_ba, s_ab, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "bc,ab,sd,Qcr,Qds->ra", s_ba, s_ab, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "bc,ce,sb,Qar,Qes->ra", s_ba, s_ab, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "br,cb,sd,Qac,Qds->ra", s_br, s_rb, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "br,cd,sb,Qac,Qds->ra", s_br, s_rb, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            - oe.contract(
                "sr,cd,db,Qac,Qbs->ra", s_sr, s_rs, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "cs,dc,sb,Qar,Qbd->ra", s_as, s_sa, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "as,cd,sb,Qdr,Qbc->ra", s_as, s_sa, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
        )

        rho_MO_exch_ind_ra = mol.cpscf(
            "A", perturbation=rho_MO_exch_ind_ra.T
        ) + oe.contract("rsab,bs->ra", mol.t_rsab, theta_t_sb.T)

        rho_MO_exch_ind[mol.slices["r"], mol.slices["a"]] = rho_MO_exch_ind_ra
        rho_MO_exch_ind[mol.slices["a"], mol.slices["r"]] = rho_MO_exch_ind_ra.T

        rho_MO_exch_ind[mol.slices["a"], mol.slices["a"]] = -oe.contract(
            "rA,ar->aA", mol.get_cpscf_ra(), theta_t_ra.T
        )
        rho_MO_exch_ind[mol.slices["r"], mol.slices["r"]] = oe.contract(
            "ra,aR->rR", mol.get_cpscf_ra(), theta_t_ra.T
        )

    if monomer == "B":

        rho_MO_exch_ind_sb = 0.5 * (
            # < R(X) | V P2 R(V) >
            +oe.contract(
                "sr,ac,ra,cb->sb", s_sr, s_ab, mol.get_cpscf_ra(), mol.omegaA_bb
            )
            - oe.contract(
                "ca,ad,sc,db->sb", s_ba, s_ab, mol.get_cpscf_sb(), mol.omegaA_bb
            )
            + oe.contract(
                "ac,sa,ce,eb->sb", s_as, s_sa, mol.get_cpscf_sb(), mol.omegaA_bb
            )
            + oe.contract(
                "sa,ac,db,cd->sb", s_sa, s_ab, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            + oe.contract(
                "ca,ab,sd,dc->sb", s_sa, s_ab, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            - oe.contract(
                "cr,ab,ra,sc->sb", s_sr, s_ab, mol.get_cpscf_ra(), mol.omegaA_ss
            )
            + oe.contract(
                "ca,ab,dc,sd->sb", s_ba, s_ab, mol.get_cpscf_sb(), mol.omegaA_ss
            )
            - oe.contract(
                "ac,da,cb,sd->sb", s_as, s_sa, mol.get_cpscf_sb(), mol.omegaA_ss
            )
            + oe.contract(
                "sa,cb,rc,ar->sb", s_sa, s_ab, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            - oe.contract(
                "sr,cb,ra,ac->sb", s_sr, s_rb, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            + oe.contract(
                "ca,rb,sc,ar->sb", s_ba, s_rb, mol.get_cpscf_sb(), mol.omegaB_ar
            )
            - oe.contract(
                "rc,sa,cb,ar->sb", s_rs, s_sa, mol.get_cpscf_sb(), mol.omegaB_ar
            )
            - oe.contract(
                "sa,cd,rc,Qar,Qdb->sb", s_sa, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qbb
            )
            + 2
            * oe.contract(
                "sa,ac,rd,Qdr,Qcb->sb", s_sa, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qbb
            )
            + oe.contract(
                "sr,cd,ra,Qac,Qdb->sb", s_sr, s_rb, mol.get_cpscf_ra(), mol.Qar, mol.Qbb
            )
            - oe.contract(
                "ca,rd,sc,Qar,Qdb->sb", s_ba, s_rb, mol.get_cpscf_sb(), mol.Qar, mol.Qbb
            )
            + 2
            * oe.contract(
                "ca,rc,se,Qar,Qeb->sb", s_ba, s_rb, mol.get_cpscf_sb(), mol.Qar, mol.Qbb
            )
            + oe.contract(
                "rc,sa,ce,Qar,Qeb->sb", s_rs, s_sa, mol.get_cpscf_sb(), mol.Qar, mol.Qbb
            )
            - 2
            * oe.contract(
                "sa,rb,cd,Qar,Qdc->sb", s_sa, s_rb, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "sa,rc,db,Qar,Qcd->sb", s_sa, s_rb, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "ca,rb,sd,Qar,Qdc->sb", s_sa, s_rb, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "ca,dc,rd,Qar,Qbs->sb", s_ba, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "cr,dc,ra,Qad,Qbs->sb", s_br, s_rb, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "rc,da,cd,Qar,Qbs->sb", s_rs, s_ba, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "ca,ab,rd,Qdr,Qsc->sb", s_sa, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qss
            )
            + oe.contract(
                "ca,db,rd,Qar,Qsc->sb", s_sa, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qss
            )
            - oe.contract(
                "cr,db,ra,Qad,Qsc->sb", s_sr, s_rb, mol.get_cpscf_ra(), mol.Qar, mol.Qss
            )
            - 2
            * oe.contract(
                "ca,rc,eb,Qar,Qse->sb", s_ba, s_rb, mol.get_cpscf_sb(), mol.Qar, mol.Qss
            )
            + oe.contract(
                "ca,rb,dc,Qar,Qsd->sb", s_ba, s_rb, mol.get_cpscf_sb(), mol.Qar, mol.Qss
            )
            - oe.contract(
                "rc,da,cb,Qar,Qsd->sb", s_rs, s_sa, mol.get_cpscf_sb(), mol.Qar, mol.Qss
            )
            # < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
            + oe.contract("sS,Sb->sb", mol.omegaA_ss, theta_t_sb)
            - oe.contract("Bb,sB->sb", mol.omegaA_bb, theta_t_sb)
            + 2 * oe.contract("Qar,Qbs,ra->sb", mol.Qar, mol.Qbs, theta_t_ra)
            # < V P2 R(X) R(V) >
            + oe.contract(
                "br,ac,ra,cs->sb", s_br, s_ab, mol.get_cpscf_ra(), mol.omegaA_bs
            )
            - oe.contract(
                "cr,as,ra,bc->sb", s_sr, s_as, mol.get_cpscf_ra(), mol.omegaA_bs
            )
            + oe.contract(
                "as,ca,dc,bd->sb", s_as, s_ba, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            + oe.contract(
                "ac,ba,ce,es->sb", s_as, s_ba, mol.get_cpscf_sb(), mol.omegaA_bs
            )
            + oe.contract(
                "as,bc,ra,cr->sb", s_as, s_ba, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            - oe.contract(
                "br,cs,ra,ac->sb", s_br, s_rs, mol.get_cpscf_ra(), mol.omegaB_ar
            )
            - oe.contract(
                "ba,cd,rc,Qar,Qds->sb", s_ba, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "ba,ac,rd,Qdr,Qcs->sb", s_ba, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            + 2
            * oe.contract(
                "ca,dc,rd,Qar,Qbs->sb", s_ba, s_ab, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "cr,dc,ra,Qad,Qbs->sb", s_br, s_rb, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "br,cd,ra,Qac,Qds->sb", s_br, s_rb, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            - oe.contract(
                "cr,ds,ra,Qad,Qbc->sb", s_sr, s_rs, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "as,ca,rd,Qdr,Qbc->sb", s_as, s_sa, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "as,cd,ra,Qdr,Qbc->sb", s_as, s_sa, mol.get_cpscf_ra(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "rs,ba,cd,Qar,Qdc->sb", s_rs, s_ba, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            - 2
            * oe.contract(
                "rc,da,cd,Qar,Qbs->sb", s_rs, s_ba, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "rs,ca,dc,Qar,Qbd->sb", s_rs, s_ba, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
            + oe.contract(
                "rc,ba,ce,Qar,Qes->sb", s_rs, s_ba, mol.get_cpscf_sb(), mol.Qar, mol.Qbs
            )
        )

        rho_MO_exch_ind_sb = mol.cpscf(
            "B", perturbation=rho_MO_exch_ind_sb.T
        ) + oe.contract("rsab,ar->sb", mol.t_rsab, theta_t_ra.T)

        rho_MO_exch_ind[mol.slices["s"], mol.slices["b"]] = rho_MO_exch_ind_sb
        rho_MO_exch_ind[mol.slices["b"], mol.slices["s"]] = rho_MO_exch_ind_sb.T

        rho_MO_exch_ind[mol.slices["b"], mol.slices["b"]] = -oe.contract(
            "sB,bs->bB", mol.get_cpscf_sb(), theta_t_sb.T
        )
        rho_MO_exch_ind[mol.slices["s"], mol.slices["s"]] = oe.contract(
            "sb,bS->sS", mol.get_cpscf_sb(), theta_t_sb.T
        )

    return rho_MO_exch_ind
