from collections.abc import Generator

import numpy as np
import opt_einsum as oe

from ..molecule import Dimer

from ..cpscf_batching import DensityBuilder, solve_now, theta_amplitudes
from .exch_ind_sinf_terms import (
    get_exch_ind_perturbation_ra,
    get_exch_ind_perturbation_sb,
)


def _build_exch_ind_theta(mol: Dimer) -> Generator:
    """Builder of the theta amplitudes of the S^inf exchange-induction density of
    monomers A ("ra") and B ("sb"), solved for together; both enter the density of
    each monomer."""

    theta_sinf_t_ra = (
        -oe.contract("sc,db,bs->cd", mol.G_sr, mol.H_ab, mol.omegaA_bs)
        + oe.contract("ca,rd,ar->dc", mol.B_aa, mol.C_rr, mol.omegaB_ar)
        - 2
        * oe.contract(
            "ca,rd,sb,Qar,Qbs->dc", mol.B_aa, mol.C_rr, mol.F_sb, mol.Qar, mol.Qbs
        )
        - oe.contract(
            "rb,ca,sd,Qar,Qbs->dc", mol.I_rb, mol.B_aa, mol.G_sr, mol.Qar, mol.Qbs
        )
        + oe.contract(
            "rc,db,sa,Qar,Qbs->cd", mol.C_rr, mol.H_ab, mol.J_sa, mol.Qar, mol.Qbs
        )
        + 2
        * oe.contract(
            "ra,sc,db,Qar,Qbs->cd", mol.E_ra, mol.G_sr, mol.H_ab, mol.Qar, mol.Qbs
        )
    )
    theta_sinf_t_sb = (
        -oe.contract("rc,da,ar->cd", mol.G_rs, mol.H_ba, mol.omegaB_ar)
        + oe.contract("cb,sd,bs->dc", mol.A_bb, mol.D_ss, mol.omegaA_bs)
        - 2
        * oe.contract(
            "ra,cb,sd,Qar,Qbs->dc", mol.E_ra, mol.A_bb, mol.D_ss, mol.Qar, mol.Qbs
        )
        - oe.contract(
            "cb,rd,sa,Qar,Qbs->dc", mol.A_bb, mol.G_rs, mol.J_sa, mol.Qar, mol.Qbs
        )
        + oe.contract(
            "rb,sc,da,Qar,Qbs->cd", mol.I_rb, mol.D_ss, mol.H_ba, mol.Qar, mol.Qbs
        )
        + 2
        * oe.contract(
            "sb,rc,da,Qar,Qbs->cd", mol.F_sb, mol.G_rs, mol.H_ba, mol.Qar, mol.Qbs
        )
    )
    theta_sinf_t_ra, theta_sinf_t_sb = yield [
        ("A", theta_sinf_t_ra.T),
        ("B", theta_sinf_t_sb.T),
    ]

    return theta_sinf_t_ra, theta_sinf_t_sb


def exch_ind_density_builder(mol: Dimer, monomer: str) -> DensityBuilder:
    """Builder of the S^inf exchange-induction density matrix (MO) of `monomer`,
    induction included, see `get_exch_ind_density`."""

    theta_sinf_t_ra, theta_sinf_t_sb = yield from theta_amplitudes(
        mol, "exch-ind", _build_exch_ind_theta
    )

    rho_MO_exch_ind = np.zeros((mol.nmo, mol.nmo))

    if monomer == "A":

        rho_MO_exch_ind_ra = get_exch_ind_perturbation_ra(
            mol,
            theta_sinf_t_ra,
            theta_sinf_t_sb,
        )
        (rho_MO_exch_ind_ra,) = yield [("A", rho_MO_exch_ind_ra.T)]
        rho_MO_exch_ind[mol.slices["r"], mol.slices["a"]] = rho_MO_exch_ind_ra
        rho_MO_exch_ind[mol.slices["a"], mol.slices["r"]] = rho_MO_exch_ind_ra.T

        # theta and the CPSCF amplitudes enter the field derivative in one order,
        # which fixes only the symmetric part of the oo and vv blocks:
        # take both orders with weight 1/2
        rho_aa = -oe.contract("rA,ar->aA", mol.get_cpscf_ra(), theta_sinf_t_ra.T)
        rho_rr = oe.contract("ra,aR->rR", mol.get_cpscf_ra(), theta_sinf_t_ra.T)
        rho_MO_exch_ind[mol.slices["a"], mol.slices["a"]] = 0.5 * (rho_aa + rho_aa.T)
        rho_MO_exch_ind[mol.slices["r"], mol.slices["r"]] = 0.5 * (rho_rr + rho_rr.T)

    if monomer == "B":

        rho_MO_exch_ind_sb = get_exch_ind_perturbation_sb(
            mol,
            theta_sinf_t_ra,
            theta_sinf_t_sb,
        )
        (rho_MO_exch_ind_sb,) = yield [("B", rho_MO_exch_ind_sb.T)]
        rho_MO_exch_ind[mol.slices["s"], mol.slices["b"]] = rho_MO_exch_ind_sb
        rho_MO_exch_ind[mol.slices["b"], mol.slices["s"]] = rho_MO_exch_ind_sb.T

        # symmetric part, as for monomer A
        rho_bb = -oe.contract("sB,bs->bB", mol.get_cpscf_sb(), theta_sinf_t_sb.T)
        rho_ss = oe.contract("sb,bS->sS", mol.get_cpscf_sb(), theta_sinf_t_sb.T)
        rho_MO_exch_ind[mol.slices["b"], mol.slices["b"]] = 0.5 * (rho_bb + rho_bb.T)
        rho_MO_exch_ind[mol.slices["s"], mol.slices["s"]] = 0.5 * (rho_ss + rho_ss.T)

    return rho_MO_exch_ind


def get_exch_ind_density(mol: Dimer, monomer: str) -> np.ndarray:
    """S^inf exchange-induction density matrix (MO) of `monomer`, induction included.

    The theta amplitudes of both monomers are kept on `mol` for the other monomer, see
    `Dimer.get_theta_amplitudes`. The CPSCF equations are solved one request at a time;
    `calc_density_matrix` solves those of all components in batches.
    """

    return solve_now(mol, exch_ind_density_builder(mol, monomer))


def _build_exch_ind_s2_theta(mol: Dimer) -> Generator:
    """Builder of the theta amplitudes of the S^2 exchange-induction density of
    monomers A ("ra") and B ("sb"), solved for together; both enter the density of
    each monomer."""

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
    theta_t_sb = (
        +oe.contract("ba,ac,cs->sb", s_ba, s_ab, mol.omegaA_bs)
        - oe.contract("as,ca,bc->sb", s_as, s_sa, mol.omegaA_bs)
        - oe.contract("rs,ba,ar->sb", s_rs, s_ba, mol.omegaB_ar)
        - 2 * oe.contract("ba,rb,Qar,Qcs->sc", s_ba, s_rb, mol.Qar, mol.Qbs)
        + oe.contract("ba,rc,Qar,Qcs->sb", s_ba, s_rb, mol.Qar, mol.Qbs)
        - oe.contract("rs,ca,Qar,Qbc->sb", s_rs, s_sa, mol.Qar, mol.Qbs)
    )
    theta_t_ra, theta_t_sb = yield [
        ("A", theta_t_ra.T),
        ("B", theta_t_sb.T),
    ]

    return theta_t_ra, theta_t_sb


def exch_ind_s2_density_builder(mol: Dimer, monomer: str) -> DensityBuilder:
    """Builder of the S^2 exchange-induction density matrix (MO) of `monomer`, see
    `get_exch_ind_s2_density`."""

    s_ab = mol.s("ab")
    s_ba = s_ab.T
    s_as = mol.s("as")
    s_sa = s_as.T
    s_rs = mol.s("rs")
    s_sr = s_rs.T
    s_rb = mol.s("rb")
    s_br = s_rb.T

    theta_t_ra, theta_t_sb = yield from theta_amplitudes(
        mol, "exch-ind_S2", _build_exch_ind_s2_theta
    )

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
            + 2 * oe.contract("Qar,Qbs,bs->ra", mol.Qar, mol.Qbs, theta_t_sb.T)
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
            # X_A -> X_A + [W_A, C_A - C_A^\dagger]_1
            + 4
            * oe.contract(
                "AR,xA,Qra,QRx->ra", theta_t_ra.T, mol.get_cpscf_ra(), mol.Qra, mol.Qrr
            )
            - oe.contract(
                "AR,xA,Qrx,QRa->ra", theta_t_ra.T, mol.get_cpscf_ra(), mol.Qrr, mol.Qra
            )
            - oe.contract(
                "AR,xA,Qax,QRr->ra", theta_t_ra.T, mol.get_cpscf_ra(), mol.Qar, mol.Qrr
            )
            - 4
            * oe.contract(
                "AR,Ry,Qra,QyA->ra", theta_t_ra.T, mol.get_cpscf_ra(), mol.Qra, mol.Qaa
            )
            + oe.contract(
                "AR,Ry,QrA,Qya->ra", theta_t_ra.T, mol.get_cpscf_ra(), mol.Qra, mol.Qaa
            )
            + oe.contract(
                "AR,Ry,QaA,Qyr->ra", theta_t_ra.T, mol.get_cpscf_ra(), mol.Qaa, mol.Qar
            )
            # two-electron Hessian derivative part
            - 4
            * oe.contract(
                "Qda,Qsc,sc,ar->rd", mol.Qaa, mol.Qra, mol.get_cpscf_ra(), theta_t_ra.T
            )
            + 4
            * oe.contract(
                "Qrt,Qsc,sc,ar->ta", mol.Qrr, mol.Qra, mol.get_cpscf_ra(), theta_t_ra.T
            )
            - 4
            * oe.contract(
                "Qra,Qdc,sc,ar->sd", mol.Qra, mol.Qaa, mol.get_cpscf_ra(), theta_t_ra.T
            )
            + 4
            * oe.contract(
                "Qra,Qst,sc,ar->tc", mol.Qra, mol.Qrr, mol.get_cpscf_ra(), theta_t_ra.T
            )
            + oe.contract(
                "Qds,Qac,sc,ar->rd", mol.Qar, mol.Qaa, mol.get_cpscf_ra(), theta_t_ra.T
            )
            + oe.contract(
                "Qrd,Qac,sc,ar->sd", mol.Qra, mol.Qaa, mol.get_cpscf_ra(), theta_t_ra.T
            )
            - oe.contract(
                "Qrs,Qtc,sc,ar->ta", mol.Qrr, mol.Qra, mol.get_cpscf_ra(), theta_t_ra.T
            )
            - oe.contract(
                "Qrs,Qat,sc,ar->tc", mol.Qrr, mol.Qar, mol.get_cpscf_ra(), theta_t_ra.T
            )
            + oe.contract(
                "Qdc,Qas,sc,ar->rd", mol.Qaa, mol.Qar, mol.get_cpscf_ra(), theta_t_ra.T
            )
            - oe.contract(
                "Qrt,Qas,sc,ar->tc", mol.Qrr, mol.Qar, mol.get_cpscf_ra(), theta_t_ra.T
            )
            - oe.contract(
                "Qrc,Qts,sc,ar->ta", mol.Qra, mol.Qrr, mol.get_cpscf_ra(), theta_t_ra.T
            )
            + oe.contract(
                "Qrc,Qad,sc,ar->sd", mol.Qra, mol.Qaa, mol.get_cpscf_ra(), theta_t_ra.T
            )
        )

        (rho_MO_exch_ind_ra,) = yield [("A", rho_MO_exch_ind_ra.T)]

        rho_MO_exch_ind[mol.slices["r"], mol.slices["a"]] = rho_MO_exch_ind_ra
        rho_MO_exch_ind[mol.slices["a"], mol.slices["r"]] = rho_MO_exch_ind_ra.T

        # theta and the CPSCF amplitudes enter the field derivative in one order,
        # which fixes only the symmetric part of the oo and vv blocks:
        # take both orders with weight 1/2
        rho_aa = -oe.contract("rA,ar->aA", mol.get_cpscf_ra(), theta_t_ra.T)
        rho_rr = oe.contract("ra,aR->rR", mol.get_cpscf_ra(), theta_t_ra.T)
        rho_MO_exch_ind[mol.slices["a"], mol.slices["a"]] = 0.5 * (rho_aa + rho_aa.T)
        rho_MO_exch_ind[mol.slices["r"], mol.slices["r"]] = 0.5 * (rho_rr + rho_rr.T)

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
            + 2 * oe.contract("Qar,Qbs,ar->sb", mol.Qar, mol.Qbs, theta_t_ra.T)
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
            # X_B -> X_B + [W_B, C_B - C_B^\dagger]_1
            + 4
            * oe.contract(
                "AR,xA,Qra,QRx->ra", theta_t_sb.T, mol.get_cpscf_sb(), mol.Qsb, mol.Qss
            )
            - oe.contract(
                "AR,xA,Qrx,QRa->ra", theta_t_sb.T, mol.get_cpscf_sb(), mol.Qss, mol.Qsb
            )
            - oe.contract(
                "AR,xA,Qax,QRr->ra", theta_t_sb.T, mol.get_cpscf_sb(), mol.Qbs, mol.Qss
            )
            - 4
            * oe.contract(
                "AR,Ry,Qra,QyA->ra", theta_t_sb.T, mol.get_cpscf_sb(), mol.Qsb, mol.Qbb
            )
            + oe.contract(
                "AR,Ry,QrA,Qya->ra", theta_t_sb.T, mol.get_cpscf_sb(), mol.Qsb, mol.Qbb
            )
            + oe.contract(
                "AR,Ry,QaA,Qyr->ra", theta_t_sb.T, mol.get_cpscf_sb(), mol.Qbb, mol.Qbs
            )
            # two-electron Hessian derivative part
            - 4
            * oe.contract(
                "Qdb,Quc,uc,bs->sd", mol.Qbb, mol.Qsb, mol.get_cpscf_sb(), theta_t_sb.T
            )
            + 4
            * oe.contract(
                "Qst,Quc,uc,bs->tb", mol.Qss, mol.Qsb, mol.get_cpscf_sb(), theta_t_sb.T
            )
            - 4
            * oe.contract(
                "Qsb,Qdc,uc,bs->ud", mol.Qsb, mol.Qbb, mol.get_cpscf_sb(), theta_t_sb.T
            )
            + 4
            * oe.contract(
                "Qsb,Qut,uc,bs->tc", mol.Qsb, mol.Qss, mol.get_cpscf_sb(), theta_t_sb.T
            )
            + oe.contract(
                "Qdu,Qbc,uc,bs->sd", mol.Qbs, mol.Qbb, mol.get_cpscf_sb(), theta_t_sb.T
            )
            + oe.contract(
                "Qsd,Qbc,uc,bs->ud", mol.Qsb, mol.Qbb, mol.get_cpscf_sb(), theta_t_sb.T
            )
            - oe.contract(
                "Qsu,Qtc,uc,bs->tb", mol.Qss, mol.Qsb, mol.get_cpscf_sb(), theta_t_sb.T
            )
            - oe.contract(
                "Qsu,Qbt,uc,bs->tc", mol.Qss, mol.Qbs, mol.get_cpscf_sb(), theta_t_sb.T
            )
            + oe.contract(
                "Qdc,Qbu,uc,bs->sd", mol.Qbb, mol.Qbs, mol.get_cpscf_sb(), theta_t_sb.T
            )
            - oe.contract(
                "Qst,Qbu,uc,bs->tc", mol.Qss, mol.Qbs, mol.get_cpscf_sb(), theta_t_sb.T
            )
            - oe.contract(
                "Qsc,Qtu,uc,bs->tb", mol.Qsb, mol.Qss, mol.get_cpscf_sb(), theta_t_sb.T
            )
            + oe.contract(
                "Qsc,Qbd,uc,bs->ud", mol.Qsb, mol.Qbb, mol.get_cpscf_sb(), theta_t_sb.T
            )
        )

        (rho_MO_exch_ind_sb,) = yield [("B", rho_MO_exch_ind_sb.T)]

        rho_MO_exch_ind[mol.slices["s"], mol.slices["b"]] = rho_MO_exch_ind_sb
        rho_MO_exch_ind[mol.slices["b"], mol.slices["s"]] = rho_MO_exch_ind_sb.T

        # symmetric part, as for monomer A
        rho_bb = -oe.contract("sB,bs->bB", mol.get_cpscf_sb(), theta_t_sb.T)
        rho_ss = oe.contract("sb,bS->sS", mol.get_cpscf_sb(), theta_t_sb.T)
        rho_MO_exch_ind[mol.slices["b"], mol.slices["b"]] = 0.5 * (rho_bb + rho_bb.T)
        rho_MO_exch_ind[mol.slices["s"], mol.slices["s"]] = 0.5 * (rho_ss + rho_ss.T)

    return rho_MO_exch_ind


def get_exch_ind_s2_density(mol: Dimer, monomer: str) -> np.ndarray:
    """S^2 exchange-induction density matrix (MO) of `monomer`.

    The theta amplitudes of both monomers are kept on `mol` for the other monomer, see
    `Dimer.get_theta_amplitudes`. The CPSCF equations are solved one request at a time;
    `calc_density_matrix` solves those of all components in batches.
    """

    return solve_now(mol, exch_ind_s2_density_builder(mol, monomer))
