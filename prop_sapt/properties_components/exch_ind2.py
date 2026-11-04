from collections.abc import Generator

import numpy as np
import opt_einsum as oe

from ..molecule import Dimer


def get_u_ra_amplitudes(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_rr: np.ndarray,
    coupled: bool = True,
) -> Generator:
    """
    Field derivative of the induction amplitudes of monomer A.

    A builder step, see `prop_sapt.cpscf_batching`: the coupled derivative yields its
    CPSCF request. Use as `u_ra = yield from get_u_ra_amplitudes(...)`.

    The monomers relax in the field either way: `xt_A_ra` and `xt_B_sb` are their
    CPSCF responses to the property operator, and the Fock operator changes with
    them. What `coupled` chooses is the amplitudes being differentiated.

    Args:
        mol (Dimer): A dimer system.
        xt_A_ra (np.ndarray): Response of monomer A to the property operator.
        xt_B_sb (np.ndarray): Response of monomer B to the property operator.
        prop_A_aa (np.ndarray): Property operator in the occupied MOs of A.
        prop_A_rr (np.ndarray): Property operator in the virtual MOs of A.
        coupled (bool, optional): True for the CPSCF amplitudes,
            `mol.get_cpscf_ra()`, whose derivative solves the CPSCF equations again.
            False for the uncoupled ones, `mol.tB_ra`, which see the electron
            interaction only through the Fock operator: their derivative drops the
            derivative of the two-electron orbital Hessian, and the orbital energy
            denominators replace the CPSCF solve. Defaults to True.

    Returns:
        np.ndarray: The derivative, in the shape of the amplitudes (ra).
    """

    xt_B_bs = xt_B_sb.T

    t_ra = mol.get_cpscf_ra() if coupled else mol.tB_ra

    # X_A -> X_A + [W_A, C_A - C_A^\dagger]_1
    gx_A_aa = (
        prop_A_aa
        + 4 * oe.contract("ra,Qar,Qcd->cd", xt_A_ra, mol.Qar, mol.Qaa)
        - oe.contract("ra,Qcr,Qad->cd", xt_A_ra, mol.Qar, mol.Qaa)
        - oe.contract("ra,Qdr,Qac->cd", xt_A_ra, mol.Qar, mol.Qaa)
    )
    gx_A_rr = (
        prop_A_rr
        + 4 * oe.contract("ra,Qar,Qcd->cd", xt_A_ra, mol.Qar, mol.Qrr)
        - oe.contract("ra,Qcr,Qad->cd", xt_A_ra, mol.Qrr, mol.Qar)
        - oe.contract("ra,Qdr,Qac->cd", xt_A_ra, mol.Qrr, mol.Qar)
    )

    u_ra = (
        oe.contract("rR,Ra->ra", gx_A_rr, t_ra)
        - oe.contract("Aa,rA->ra", gx_A_aa, t_ra)
        + 2 * oe.contract("bs,Qra,Qsb->ra", xt_B_bs, mol.Qra, mol.Qsb)
        + oe.contract("Ra,rR->ra", xt_A_ra, mol.omegaB_rr)
        - oe.contract("rA,Aa->ra", xt_A_ra, mol.omegaB_aa)
        + 2 * oe.contract("sb,Qar,Qbs->ra", xt_B_sb, mol.Qar, mol.Qbs)
    )

    if not coupled:
        # uncoupled amplitudes: no two-electron Hessian, denominators instead of CPSCF
        return u_ra * mol.eps_ra

    # two-electron Hessian derivative acting on C_A^(1):
    #   sum_RA t_RA d/dF [4(ra|RA) - (rR|aA) - (rA|aR)]
    u_ra += (
        -4 * oe.contract("rd,Qda,Qsc,sc->ra", xt_A_ra, mol.Qaa, mol.Qra, t_ra)
        + 4 * oe.contract("ta,Qrt,Qsc,sc->ra", xt_A_ra, mol.Qrr, mol.Qra, t_ra)
        - 4 * oe.contract("sd,Qra,Qdc,sc->ra", xt_A_ra, mol.Qra, mol.Qaa, t_ra)
        + 4 * oe.contract("tc,Qra,Qst,sc->ra", xt_A_ra, mol.Qra, mol.Qrr, t_ra)
        + oe.contract("rd,Qds,Qac,sc->ra", xt_A_ra, mol.Qar, mol.Qaa, t_ra)
        + oe.contract("sd,Qrd,Qac,sc->ra", xt_A_ra, mol.Qra, mol.Qaa, t_ra)
        - oe.contract("ta,Qrs,Qtc,sc->ra", xt_A_ra, mol.Qrr, mol.Qra, t_ra)
        - oe.contract("tc,Qrs,Qat,sc->ra", xt_A_ra, mol.Qrr, mol.Qar, t_ra)
        + oe.contract("rd,Qdc,Qas,sc->ra", xt_A_ra, mol.Qaa, mol.Qar, t_ra)
        - oe.contract("tc,Qrt,Qas,sc->ra", xt_A_ra, mol.Qrr, mol.Qar, t_ra)
        - oe.contract("ta,Qrc,Qts,sc->ra", xt_A_ra, mol.Qra, mol.Qrr, t_ra)
        + oe.contract("sd,Qrc,Qad,sc->ra", xt_A_ra, mol.Qra, mol.Qaa, t_ra)
    )

    (u_ra,) = yield [("A", u_ra.T)]

    return u_ra


def get_u_sb_amplitudes(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_B_bb: np.ndarray,
    prop_B_ss: np.ndarray,
    coupled: bool = True,
) -> Generator:
    """
    Field derivative of the induction amplitudes of monomer B.

    A builder step, see `prop_sapt.cpscf_batching`: the coupled derivative yields its
    CPSCF request. Use as `u_sb = yield from get_u_sb_amplitudes(...)`.

    The monomer B counterpart of `get_u_ra_amplitudes`; `coupled` chooses between
    the CPSCF amplitudes, `mol.get_cpscf_sb()`, and the uncoupled ones, `mol.tA_sb`.

    Args:
        mol (Dimer): A dimer system.
        xt_A_ra (np.ndarray): Response of monomer A to the property operator.
        xt_B_sb (np.ndarray): Response of monomer B to the property operator.
        prop_B_bb (np.ndarray): Property operator in the occupied MOs of B.
        prop_B_ss (np.ndarray): Property operator in the virtual MOs of B.
        coupled (bool, optional): True for the CPSCF amplitudes, False for the
            uncoupled ones. Defaults to True.

    Returns:
        np.ndarray: The derivative, in the shape of the amplitudes (sb).
    """

    xt_A_ar = xt_A_ra.T

    t_sb = mol.get_cpscf_sb() if coupled else mol.tA_sb

    # X_B -> X_B + [W_B, C_B - C_B^\dagger]_1
    gx_B_bb = (
        prop_B_bb
        + 4 * oe.contract("sb,Qbs,Qcd->cd", xt_B_sb, mol.Qbs, mol.Qbb)
        - oe.contract("sb,Qcs,Qbd->cd", xt_B_sb, mol.Qbs, mol.Qbb)
        - oe.contract("sb,Qds,Qbc->cd", xt_B_sb, mol.Qbs, mol.Qbb)
    )
    gx_B_ss = (
        prop_B_ss
        + 4 * oe.contract("sb,Qbs,Qcd->cd", xt_B_sb, mol.Qbs, mol.Qss)
        - oe.contract("sb,Qcs,Qbd->cd", xt_B_sb, mol.Qss, mol.Qbs)
        - oe.contract("sb,Qds,Qbc->cd", xt_B_sb, mol.Qss, mol.Qbs)
    )

    u_sb = (
        oe.contract("sS,Sb->sb", gx_B_ss, t_sb)
        - oe.contract("Bb,sB->sb", gx_B_bb, t_sb)
        + 2 * oe.contract("ar,Qra,Qsb->sb", xt_A_ar, mol.Qra, mol.Qsb)
        + oe.contract("Sb,sS->sb", xt_B_sb, mol.omegaA_ss)
        - oe.contract("sB,Bb->sb", xt_B_sb, mol.omegaA_bb)
        + 2 * oe.contract("ra,Qar,Qbs->sb", xt_A_ra, mol.Qar, mol.Qbs)
    )

    if not coupled:
        # uncoupled amplitudes: no two-electron Hessian, denominators instead of CPSCF
        return u_sb * mol.eps_sb

    # two-electron Hessian derivative acting on C_B^(1):
    #   sum_SB t_SB d/dF [4(sb|SB) - (sS|bB) - (sB|bS)]
    u_sb += (
        -4 * oe.contract("sd,Qdb,Quc,uc->sb", xt_B_sb, mol.Qbb, mol.Qsb, t_sb)
        + 4 * oe.contract("tb,Qst,Quc,uc->sb", xt_B_sb, mol.Qss, mol.Qsb, t_sb)
        - 4 * oe.contract("ud,Qsb,Qdc,uc->sb", xt_B_sb, mol.Qsb, mol.Qbb, t_sb)
        + 4 * oe.contract("tc,Qsb,Qut,uc->sb", xt_B_sb, mol.Qsb, mol.Qss, t_sb)
        + oe.contract("sd,Qdu,Qbc,uc->sb", xt_B_sb, mol.Qbs, mol.Qbb, t_sb)
        + oe.contract("ud,Qsd,Qbc,uc->sb", xt_B_sb, mol.Qsb, mol.Qbb, t_sb)
        - oe.contract("tb,Qsu,Qtc,uc->sb", xt_B_sb, mol.Qss, mol.Qsb, t_sb)
        - oe.contract("tc,Qsu,Qbt,uc->sb", xt_B_sb, mol.Qss, mol.Qbs, t_sb)
        + oe.contract("sd,Qdc,Qbu,uc->sb", xt_B_sb, mol.Qbb, mol.Qbs, t_sb)
        - oe.contract("tc,Qst,Qbu,uc->sb", xt_B_sb, mol.Qss, mol.Qbs, t_sb)
        - oe.contract("tb,Qsc,Qtu,uc->sb", xt_B_sb, mol.Qsb, mol.Qss, t_sb)
        + oe.contract("ud,Qsc,Qbd,uc->sb", xt_B_sb, mol.Qsb, mol.Qbb, t_sb)
    )

    (u_sb,) = yield [("B", u_sb.T)]

    return u_sb


def calc_exch_ind2_resp_s2_property(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    u_ra: np.ndarray,
    u_sb: np.ndarray,
    t_ra: np.ndarray | None = None,
    t_sb: np.ndarray | None = None,
) -> float:
    """
    Field derivative of the S^2 exchange-induction energy.

    The energy is linear in the induction amplitudes, so the same expression serves
    both kinds: the coupled ones with their derivatives from `get_u_ra_amplitudes`
    and `get_u_sb_amplitudes`, or the uncoupled ones with the derivatives the same
    functions give for `coupled=False`.

    Args:
        mol (Dimer): A dimer system.
        xt_A_ra (np.ndarray): Response of monomer A to the property operator.
        xt_B_sb (np.ndarray): Response of monomer B to the property operator.
        u_ra (np.ndarray): Field derivative of `t_ra`.
        u_sb (np.ndarray): Field derivative of `t_sb`.
        t_ra (np.ndarray, optional): Induction amplitudes of monomer A. Defaults to
            the CPSCF ones, `mol.get_cpscf_ra()`.
        t_sb (np.ndarray, optional): Induction amplitudes of monomer B. Defaults to
            the CPSCF ones, `mol.get_cpscf_sb()`.

    Returns:
        float: The property contribution.
    """

    if t_ra is None:
        t_ra = mol.get_cpscf_ra()
    if t_sb is None:
        t_sb = mol.get_cpscf_sb()

    xt_A_ar = xt_A_ra.T
    xt_B_bs = xt_B_sb.T

    s_ab = mol.s("ab")
    s_ba = s_ab.T
    s_as = mol.s("as")
    s_sa = s_as.T
    s_rs = mol.s("rs")
    s_sr = s_rs.T
    s_rb = mol.s("rb")
    s_br = s_rb.T

    x2_exch_ind_resp_s2 = np.array(
        [
            # < R(X) | V P2 R(V) >
            +2 * oe.contract("bs,sr,ac,ra,cb", xt_B_bs, s_sr, s_ab, t_ra, mol.omegaA_bb)
            - 2
            * oe.contract("bs,ca,ad,sc,db", xt_B_bs, s_ba, s_ab, t_sb, mol.omegaA_bb)
            + 2
            * oe.contract("bs,ac,sa,ce,eb", xt_B_bs, s_as, s_sa, t_sb, mol.omegaA_bb)
            + 2
            * oe.contract("ar,sa,cb,rc,bs", xt_A_ar, s_sa, s_ab, t_ra, mol.omegaA_bs)
            - 2
            * oe.contract("ar,sc,rb,ca,bs", xt_A_ar, s_sr, s_rb, t_ra, mol.omegaA_bs)
            + 2
            * oe.contract("bs,sa,ac,db,cd", xt_B_bs, s_sa, s_ab, t_sb, mol.omegaA_bs)
            + 2
            * oe.contract("bs,ca,ab,sd,dc", xt_B_bs, s_sa, s_ab, t_sb, mol.omegaA_bs)
            + 2
            * oe.contract("ar,ba,rc,sb,cs", xt_A_ar, s_ba, s_rb, t_sb, mol.omegaA_bs)
            - 2
            * oe.contract("ar,rs,ca,sb,bc", xt_A_ar, s_rs, s_sa, t_sb, mol.omegaA_bs)
            - 2
            * oe.contract("bs,cr,ab,ra,sc", xt_B_bs, s_sr, s_ab, t_ra, mol.omegaA_ss)
            + 2
            * oe.contract("bs,ca,ab,dc,sd", xt_B_bs, s_ba, s_ab, t_sb, mol.omegaA_ss)
            - 2
            * oe.contract("bs,ac,da,cb,sd", xt_B_bs, s_as, s_sa, t_sb, mol.omegaA_ss)
            - 2
            * oe.contract("ar,bc,db,rd,ca", xt_A_ar, s_ba, s_ab, t_ra, mol.omegaB_aa)
            + 2
            * oe.contract("ar,bc,rb,ce,ea", xt_A_ar, s_br, s_rb, t_ra, mol.omegaB_aa)
            + 2
            * oe.contract("ar,rs,bc,sb,ca", xt_A_ar, s_rs, s_ba, t_sb, mol.omegaB_aa)
            + 2
            * oe.contract("bs,sa,cb,rc,ar", xt_B_bs, s_sa, s_ab, t_ra, mol.omegaB_ar)
            + 2
            * oe.contract("ar,ba,cb,rd,dc", xt_A_ar, s_ba, s_rb, t_ra, mol.omegaB_ar)
            + 2
            * oe.contract("ar,bc,rb,da,cd", xt_A_ar, s_ba, s_rb, t_ra, mol.omegaB_ar)
            - 2
            * oe.contract("bs,sr,cb,ra,ac", xt_B_bs, s_sr, s_rb, t_ra, mol.omegaB_ar)
            + 2
            * oe.contract("bs,ca,rb,sc,ar", xt_B_bs, s_ba, s_rb, t_sb, mol.omegaB_ar)
            - 2
            * oe.contract("bs,rc,sa,cb,ar", xt_B_bs, s_rs, s_sa, t_sb, mol.omegaB_ar)
            + 2
            * oe.contract("ar,ba,cb,dc,rd", xt_A_ar, s_ba, s_ab, t_ra, mol.omegaB_rr)
            - 2
            * oe.contract("ar,bc,db,ca,rd", xt_A_ar, s_br, s_rb, t_ra, mol.omegaB_rr)
            - 2
            * oe.contract("ar,cs,ba,sb,rc", xt_A_ar, s_rs, s_ba, t_sb, mol.omegaB_rr)
            - 2
            * oe.contract(
                "ar,sc,db,rd,Qca,Qbs",
                xt_A_ar,
                s_sa,
                s_ab,
                t_ra,
                mol.Qaa,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,sc,cb,re,Qea,Qbs",
                xt_A_ar,
                s_sa,
                s_ab,
                t_ra,
                mol.Qaa,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,sc,rb,ce,Qea,Qbs",
                xt_A_ar,
                s_sr,
                s_rb,
                t_ra,
                mol.Qaa,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,bc,rd,sb,Qca,Qds",
                xt_A_ar,
                s_ba,
                s_rb,
                t_sb,
                mol.Qaa,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,bc,rb,sd,Qca,Qds",
                xt_A_ar,
                s_ba,
                s_rb,
                t_sb,
                mol.Qaa,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,rs,cd,sb,Qda,Qbc",
                xt_A_ar,
                s_rs,
                s_sa,
                t_sb,
                mol.Qaa,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "bs,sa,cd,rc,Qar,Qdb",
                xt_B_bs,
                s_sa,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qbb,
            )
            + 4
            * oe.contract(
                "bs,sa,ac,rd,Qdr,Qcb",
                xt_B_bs,
                s_sa,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qbb,
            )
            + 2
            * oe.contract(
                "bs,sr,cd,ra,Qac,Qdb",
                xt_B_bs,
                s_sr,
                s_rb,
                t_ra,
                mol.Qar,
                mol.Qbb,
            )
            - 2
            * oe.contract(
                "bs,ca,rd,sc,Qar,Qdb",
                xt_B_bs,
                s_ba,
                s_rb,
                t_sb,
                mol.Qar,
                mol.Qbb,
            )
            + 4
            * oe.contract(
                "bs,ca,rc,se,Qar,Qeb",
                xt_B_bs,
                s_ba,
                s_rb,
                t_sb,
                mol.Qar,
                mol.Qbb,
            )
            + 2
            * oe.contract(
                "bs,rc,sa,ce,Qar,Qeb",
                xt_B_bs,
                s_rs,
                s_sa,
                t_sb,
                mol.Qar,
                mol.Qbb,
            )
            - 4
            * oe.contract(
                "ar,sa,rb,cd,Qdc,Qbs",
                xt_A_ar,
                s_sa,
                s_rb,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,sa,cb,rd,Qdc,Qbs",
                xt_A_ar,
                s_sa,
                s_rb,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,sc,rb,da,Qcd,Qbs",
                xt_A_ar,
                s_sa,
                s_rb,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,sa,rb,cd,Qar,Qdc",
                xt_B_bs,
                s_sa,
                s_rb,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "bs,sa,rc,db,Qar,Qcd",
                xt_B_bs,
                s_sa,
                s_rb,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "bs,ca,rb,sd,Qar,Qdc",
                xt_B_bs,
                s_sa,
                s_rb,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "bs,ca,dc,rd,Qar,Qbs",
                xt_B_bs,
                s_ba,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,cr,dc,ra,Qad,Qbs",
                xt_B_bs,
                s_br,
                s_rb,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,rc,da,cd,Qar,Qbs",
                xt_B_bs,
                s_rs,
                s_ba,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "bs,ca,ab,rd,Qdr,Qsc",
                xt_B_bs,
                s_sa,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qss,
            )
            + 2
            * oe.contract(
                "bs,ca,db,rd,Qar,Qsc",
                xt_B_bs,
                s_sa,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qss,
            )
            - 2
            * oe.contract(
                "bs,cr,db,ra,Qad,Qsc",
                xt_B_bs,
                s_sr,
                s_rb,
                t_ra,
                mol.Qar,
                mol.Qss,
            )
            - 4
            * oe.contract(
                "bs,ca,rc,eb,Qar,Qse",
                xt_B_bs,
                s_ba,
                s_rb,
                t_sb,
                mol.Qar,
                mol.Qss,
            )
            + 2
            * oe.contract(
                "bs,ca,rb,dc,Qar,Qsd",
                xt_B_bs,
                s_ba,
                s_rb,
                t_sb,
                mol.Qar,
                mol.Qss,
            )
            - 2
            * oe.contract(
                "bs,rc,da,cb,Qar,Qsd",
                xt_B_bs,
                s_rs,
                s_sa,
                t_sb,
                mol.Qar,
                mol.Qss,
            )
            - 4
            * oe.contract(
                "ar,sc,db,cd,Qar,Qbs",
                xt_A_ar,
                s_sr,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ar,bc,ce,sb,Qar,Qes",
                xt_A_ar,
                s_ba,
                s_ab,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,cs,dc,sb,Qar,Qbd",
                xt_A_ar,
                s_as,
                s_sa,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,sc,cb,ea,Qre,Qbs",
                xt_A_ar,
                s_sa,
                s_ab,
                t_ra,
                mol.Qrr,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,sa,cb,dc,Qrd,Qbs",
                xt_A_ar,
                s_sa,
                s_ab,
                t_ra,
                mol.Qrr,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,sc,db,ca,Qrd,Qbs",
                xt_A_ar,
                s_sr,
                s_rb,
                t_ra,
                mol.Qrr,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ar,ba,cb,sd,Qrc,Qds",
                xt_A_ar,
                s_ba,
                s_rb,
                t_sb,
                mol.Qrr,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ar,ba,cd,sb,Qrc,Qds",
                xt_A_ar,
                s_ba,
                s_rb,
                t_sb,
                mol.Qrr,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ar,cs,da,sb,Qrc,Qbd",
                xt_A_ar,
                s_rs,
                s_sa,
                t_sb,
                mol.Qrr,
                mol.Qbs,
            )
            # < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
            - 2 * oe.contract("sr,ab,ra,bs", s_sr, s_ab, u_ra, mol.omegaA_bs)
            + 2 * oe.contract("ba,cb,rc,ar", s_ba, s_ab, u_ra, mol.omegaB_ar)
            - 2 * oe.contract("br,cb,ra,ac", s_br, s_rb, u_ra, mol.omegaB_ar)
            - 4 * oe.contract("sa,ab,rc,Qcr,Qbs", s_sa, s_ab, u_ra, mol.Qar, mol.Qbs)
            + 2 * oe.contract("sa,cb,rc,Qar,Qbs", s_sa, s_ab, u_ra, mol.Qar, mol.Qbs)
            - 2 * oe.contract("sr,cb,ra,Qac,Qbs", s_sr, s_rb, u_ra, mol.Qar, mol.Qbs)
            + 2 * oe.contract("ba,ac,sb,cs", s_ba, s_ab, u_sb, mol.omegaA_bs)
            - 2 * oe.contract("as,ca,sb,bc", s_as, s_sa, u_sb, mol.omegaA_bs)
            - 2 * oe.contract("rs,ba,sb,ar", s_rs, s_ba, u_sb, mol.omegaB_ar)
            - 4 * oe.contract("ba,rb,sc,Qar,Qcs", s_ba, s_rb, u_sb, mol.Qar, mol.Qbs)
            + 2 * oe.contract("ba,rc,sb,Qar,Qcs", s_ba, s_rb, u_sb, mol.Qar, mol.Qbs)
            - 2 * oe.contract("rs,ca,sb,Qar,Qbc", s_rs, s_sa, u_sb, mol.Qar, mol.Qbs)
            # < V P2 R(X) R(V) >
            + 2
            * oe.contract("sb,br,ac,ra,cs", xt_B_sb, s_br, s_ab, t_ra, mol.omegaA_bs)
            - 2
            * oe.contract("sb,cr,as,ra,bc", xt_B_sb, s_sr, s_as, t_ra, mol.omegaA_bs)
            + 2
            * oe.contract("ra,br,ac,sb,cs", xt_A_ra, s_br, s_ab, t_sb, mol.omegaA_bs)
            - 2
            * oe.contract("ra,sr,ac,cb,bs", xt_A_ra, s_sr, s_as, t_sb, mol.omegaA_bs)
            + 2
            * oe.contract("sb,as,ca,dc,bd", xt_B_sb, s_as, s_ba, t_sb, mol.omegaA_bs)
            + 2
            * oe.contract("sb,ac,ba,ce,es", xt_B_sb, s_as, s_ba, t_sb, mol.omegaA_bs)
            + 2
            * oe.contract("ra,br,cb,dc,ad", xt_A_ra, s_br, s_ab, t_ra, mol.omegaB_ar)
            + 2
            * oe.contract("ra,bc,ab,ce,er", xt_A_ra, s_br, s_ab, t_ra, mol.omegaB_ar)
            + 2
            * oe.contract("sb,as,bc,ra,cr", xt_B_sb, s_as, s_ba, t_ra, mol.omegaB_ar)
            - 2
            * oe.contract("sb,br,cs,ra,ac", xt_B_sb, s_br, s_rs, t_ra, mol.omegaB_ar)
            + 2
            * oe.contract("ra,as,bc,sb,cr", xt_A_ra, s_as, s_ba, t_sb, mol.omegaB_ar)
            - 2
            * oe.contract("ra,br,cs,sb,ac", xt_A_ra, s_br, s_rs, t_sb, mol.omegaB_ar)
            - 2
            * oe.contract(
                "sb,ba,cd,rc,Qar,Qds",
                xt_B_sb,
                s_ba,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "sb,ba,ac,rd,Qdr,Qcs",
                xt_B_sb,
                s_ba,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "sb,ca,dc,rd,Qar,Qbs",
                xt_B_sb,
                s_ba,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ra,sr,ab,cd,Qdc,Qbs",
                xt_A_ra,
                s_sr,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ra,sc,db,cd,Qar,Qbs",
                xt_A_ra,
                s_sr,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ra,sr,cb,dc,Qad,Qbs",
                xt_A_ra,
                s_sr,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ra,sc,ab,ce,Qer,Qbs",
                xt_A_ra,
                s_sr,
                s_ab,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,cr,dc,ra,Qad,Qbs",
                xt_B_sb,
                s_br,
                s_rb,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "sb,br,cd,ra,Qac,Qds",
                xt_B_sb,
                s_br,
                s_rb,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "sb,cr,ds,ra,Qad,Qbc",
                xt_B_sb,
                s_sr,
                s_rs,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,as,ca,rd,Qdr,Qbc",
                xt_B_sb,
                s_as,
                s_sa,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "sb,as,cd,ra,Qdr,Qbc",
                xt_B_sb,
                s_as,
                s_sa,
                t_ra,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ra,bc,ad,sb,Qcr,Qds",
                xt_A_ra,
                s_ba,
                s_ab,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ra,bc,ab,sd,Qcr,Qds",
                xt_A_ra,
                s_ba,
                s_ab,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            + 4
            * oe.contract(
                "ra,bc,ce,sb,Qar,Qes",
                xt_A_ra,
                s_ba,
                s_ab,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,rs,ba,cd,Qar,Qdc",
                xt_B_sb,
                s_rs,
                s_ba,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,rc,da,cd,Qar,Qbs",
                xt_B_sb,
                s_rs,
                s_ba,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "sb,rs,ca,dc,Qar,Qbd",
                xt_B_sb,
                s_rs,
                s_ba,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "sb,rc,ba,ce,Qar,Qes",
                xt_B_sb,
                s_rs,
                s_ba,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ra,br,cb,sd,Qac,Qds",
                xt_A_ra,
                s_br,
                s_rb,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ra,br,cd,sb,Qac,Qds",
                xt_A_ra,
                s_br,
                s_rb,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            - 2
            * oe.contract(
                "ra,sr,cd,db,Qac,Qbs",
                xt_A_ra,
                s_sr,
                s_rs,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "ra,cs,dc,sb,Qar,Qbd",
                xt_A_ra,
                s_as,
                s_sa,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "ra,as,cd,sb,Qdr,Qbc",
                xt_A_ra,
                s_as,
                s_sa,
                t_sb,
                mol.Qar,
                mol.Qbs,
            )
        ]
    )

    return x2_exch_ind_resp_s2.item()
