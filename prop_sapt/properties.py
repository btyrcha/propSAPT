"""
Calculations of SAPT one-electron properties.
"""

from collections.abc import Generator

import psi4
import numpy as np
import pandas as pd
import opt_einsum as oe

from .cpscf_batching import solve_batched, solve_now
from .molecule import Dimer
from .utils import trace_memory_peak, CalcTimer

from .properties_components import (
    calc_exch_ind2_resp_s2_property,
    calc_exch_ind2_resp_sinf_property,
    calc_exch_disp2_s2_property,
    calc_exch_disp2_sinf_property,
)
from .properties_components.exch_ind2 import (
    get_u_ra_amplitudes,
    get_u_sb_amplitudes,
)

RESPONSE_VALUES = (True, False, "both")


def _check_response(response) -> None:
    """Rejects a `response` value other than True, False or "both"."""

    if response not in RESPONSE_VALUES:
        raise ValueError(
            f"Invalid value for 'response': {response!r}. Must be True, False or \"both\"."
        )


def perform_property_contractions(
    dimer: Dimer, property_matrix: np.ndarray, response: bool | str = True
) -> pd.DataFrame:
    """
    Calculates property contributions. Returns pandas.DataFrame with values of separate
    contributions and the summed-up total.

    The property is the field derivative of the energy of `calc_sapt_energy` with the
    same `response`. The monomers relax in the field in every mode (the first-order,
    dispersion and exchange-dispersion terms do not depend on it); `response` chooses
    the induction amplitudes of the second-order induction terms:

    - True: the coupled (CPSCF) ones, `x2_ind,r`, `x2_exch-ind,r_S2` and
      `x2_exch-ind,r`. The uncoupled `x2_ind` is reported too.
    - False: the uncoupled ones, `x2_ind`, `x2_exch-ind_S2` and `x2_exch-ind`. No
      CPSCF equations are solved for the induction amplitudes or their derivatives.
    - "both": both sets.

    The totals `x_induced` and `x_induced_S2` take the coupled terms unless
    `response` is False, as `calc_sapt_energy` does.

    Args:
        dimer (Dimer): A dimer system.
        property_matrix (np.ndarray): A matrix representation of the operator corresponding
            to a property to calculate, in AO basis.
        response (bool | str, optional): True, False or "both", see above. Defaults to
            True.

    Returns:
        pd.DataFrame: DataFrame with results.
    """

    _check_response(response)

    return solve_now(dimer, _property_builder(dimer, property_matrix, response))


def _property_builder(
    mol: Dimer, property_matrix: np.ndarray, response: bool | str
) -> Generator:
    """Builder of the results of `perform_property_contractions`, see
    `prop_sapt.cpscf_batching`."""

    coupled = response in (True, "both")
    uncoupled = response in (False, "both")

    # prepare results DataFrame, one row that scalars are broadcast into
    results = pd.DataFrame(index=pd.RangeIndex(1))

    # variables for each axis
    prop_A_aa = mol.transform_ao_to_mo(property_matrix, "aa")
    prop_A_ar = mol.transform_ao_to_mo(property_matrix, "ar")
    prop_A_rr = mol.transform_ao_to_mo(property_matrix, "rr")

    prop_B_bb = mol.transform_ao_to_mo(property_matrix, "bb")
    prop_B_bs = mol.transform_ao_to_mo(property_matrix, "bs")
    prop_B_ss = mol.transform_ao_to_mo(property_matrix, "ss")

    # relaxed amplitudes with dipole moment
    xt_A_ra, xt_B_sb = yield [("A", prop_A_ar), ("B", prop_B_bs)]

    # SCF monomers

    results["x0_A"] = np.array([2 * oe.contract("aa", prop_A_aa)])
    results["x0_B"] = np.array([2 * oe.contract("bb", prop_B_bb)])

    ### First-order

    results["x1_pol,r_A"] = np.array([4 * oe.contract("ra,ar", xt_A_ra, mol.omegaB_ar)])
    results["x1_pol,r_B"] = np.array([4 * oe.contract("sb,bs", xt_B_sb, mol.omegaA_bs)])

    results["x1_pol,r"] = results["x1_pol,r_A"] + results["x1_pol,r_B"]

    results["x1_exch,r_S2_A"] = np.array(
        [
            2 * oe.contract("ra,ar", xt_A_ra, mol.omega_exchB_ar_S2)
            + 2 * oe.contract("ra,ra", mol.omega_exchB_ra_S2, xt_A_ra)
        ]
    )

    results["x1_exch,r_S2_B"] = np.array(
        [
            2 * oe.contract("sb,bs", xt_B_sb, mol.omega_exchA_bs_S2)
            + 2 * oe.contract("sb,sb", mol.omega_exchA_sb_S2, xt_B_sb)
        ]
    )

    results["x1_exch,r_S2"] = results["x1_exch,r_S2_A"] + results["x1_exch,r_S2_B"]

    results["x1_exch,r_A"] = np.array(
        [
            2 * oe.contract("ra,ar", xt_A_ra, mol.omega_exchB_ar)
            + 2 * oe.contract("ra,ra", mol.omega_exchB_ra, xt_A_ra)
        ]
    )

    results["x1_exch,r_B"] = np.array(
        [
            2 * oe.contract("sb,bs", xt_B_sb, mol.omega_exchA_bs)
            + 2 * oe.contract("sb,sb", mol.omega_exchA_sb, xt_B_sb)
        ]
    )

    results["x1_exch,r"] = results["x1_exch,r_A"] + results["x1_exch,r_B"]

    ### Second-order

    ## Induction
    results["x2_ind_A"] = np.array(
        [
            +16 * oe.contract("ra,Qar,Qbs,sb", xt_A_ra, mol.Qar, mol.Qbs, mol.tA_sb)
            - 4 * oe.contract("ra,ac,rc", mol.tB_ra, mol.omegaB_aa, xt_A_ra)
            + 4 * oe.contract("ra,cr,ca", mol.tB_ra, mol.omegaB_rr, xt_A_ra)
            - 2 * oe.contract("ac,rc,ra", prop_A_aa, mol.tB_ra, mol.tB_ra)
            + 2 * oe.contract("cr,ca,ra", prop_A_rr, mol.tB_ra, mol.tB_ra)
            # X_A -> X_A + [W_A, C_A - C_A^\dagger]_1
            + 4
            * oe.contract(
                "ra,cd,de,Qer,Qac", xt_A_ra, mol.tB_ra.T, mol.tB_ra, mol.Qar, mol.Qaa
            )
            + 8
            * oe.contract(
                "ra,cd,ec,Qde,Qar", xt_A_ra, mol.tB_ra.T, mol.tB_ra, mol.Qrr, mol.Qar
            )
            - 4
            * oe.contract(
                "ra,cd,ec,Qdr,Qae", xt_A_ra, mol.tB_ra.T, mol.tB_ra, mol.Qrr, mol.Qar
            )
            - 8
            * oe.contract(
                "ra,cd,de,Qar,Qec", xt_A_ra, mol.tB_ra.T, mol.tB_ra, mol.Qar, mol.Qaa
            )
        ]
    )

    results["x2_ind_B"] = np.array(
        [
            +16 * oe.contract("sb,Qar,Qbs,ra", xt_B_sb, mol.Qar, mol.Qbs, mol.tB_ra)
            - 4 * oe.contract("sb,bc,sc", mol.tA_sb, mol.omegaA_bb, xt_B_sb)
            + 4 * oe.contract("sb,cs,cb", mol.tA_sb, mol.omegaA_ss, xt_B_sb)
            - 2 * oe.contract("bc,sc,sb", prop_B_bb, mol.tA_sb, mol.tA_sb)
            + 2 * oe.contract("cs,cb,sb", prop_B_ss, mol.tA_sb, mol.tA_sb)
            # X_B -> X_B + [W_B, C_B - C_B^\dagger]_1
            + 4
            * oe.contract(
                "sb,cd,de,Qes,Qbc", xt_B_sb, mol.tA_sb.T, mol.tA_sb, mol.Qbs, mol.Qbb
            )
            + 8
            * oe.contract(
                "sb,cd,ec,Qde,Qbs", xt_B_sb, mol.tA_sb.T, mol.tA_sb, mol.Qss, mol.Qbs
            )
            - 4
            * oe.contract(
                "sb,cd,ec,Qds,Qbe", xt_B_sb, mol.tA_sb.T, mol.tA_sb, mol.Qss, mol.Qbs
            )
            - 8
            * oe.contract(
                "sb,cd,de,Qbs,Qec", xt_B_sb, mol.tA_sb.T, mol.tA_sb, mol.Qbs, mol.Qbb
            )
        ]
    )

    results["x2_ind"] = results["x2_ind_A"] + results["x2_ind_B"]

    ## Induction response
    if coupled:
        results["x2_ind,r_A"], results["x2_ind,r_B"] = _calc_ind2_r_terms(
            mol, xt_A_ra, xt_B_sb, prop_A_aa, prop_A_rr, prop_B_bb, prop_B_ss
        )
        results["x2_ind,r"] = results["x2_ind,r_A"] + results["x2_ind,r_B"]

    ## Exchange-Induction S^2 and S^inf
    prop_blocks = (prop_A_aa, prop_A_rr, prop_B_bb, prop_B_ss)
    if coupled:
        yield from _add_exch_ind2_terms(
            results, mol, xt_A_ra, xt_B_sb, *prop_blocks, coupled=True
        )
    if uncoupled:
        yield from _add_exch_ind2_terms(
            results, mol, xt_A_ra, xt_B_sb, *prop_blocks, coupled=False
        )

    ## Dispersion
    results["x2_disp_A"] = np.array(
        [
            # 2 Re <R(X)|V R(V)>
            # NOTE some response not coupled
            8 * oe.contract("rsab,Qcr,Qbs,ca", mol.t_rsab, mol.Qrr, mol.Qbs, xt_A_ra)
            - 8 * oe.contract("rsab,Qac,Qbs,rc", mol.t_rsab, mol.Qaa, mol.Qbs, xt_A_ra)
            # <R(V)|X R(V)>
            # NOTE not coupled no response
            - 4 * oe.contract("ac,rscb,rsab", prop_A_aa, mol.t_rsab, mol.t_rsab)
            + 4 * oe.contract("cr,csab,rsab", prop_A_rr, mol.t_rsab, mol.t_rsab)
            # X_A -> X_A + [W_A, C_A - C_A^\dagger]_1
            + 8
            * oe.contract(
                "ra,dscb,dseb,Qer,Qac",
                xt_A_ra,
                mol.t_rsab,
                mol.t_rsab,
                mol.Qar,
                mol.Qaa,
            )
            + 16
            * oe.contract(
                "ra,dscb,escb,Qar,Qde",
                xt_A_ra,
                mol.t_rsab,
                mol.t_rsab,
                mol.Qar,
                mol.Qrr,
            )
            - 8
            * oe.contract(
                "ra,dscb,escb,Qdr,Qae",
                xt_A_ra,
                mol.t_rsab,
                mol.t_rsab,
                mol.Qrr,
                mol.Qar,
            )
            - 16
            * oe.contract(
                "ra,dscb,dseb,Qar,Qec",
                xt_A_ra,
                mol.t_rsab,
                mol.t_rsab,
                mol.Qar,
                mol.Qaa,
            )
        ]
    )

    results["x2_disp_B"] = np.array(
        [
            # 2 Re <R(X)|V R(V)>
            # NOTE some response not coupled
            8 * oe.contract("rsab,Qar,Qcs,cb", mol.t_rsab, mol.Qar, mol.Qss, xt_B_sb)
            - 8 * oe.contract("rsab,Qar,Qbc,sc", mol.t_rsab, mol.Qar, mol.Qbb, xt_B_sb)
            # <R(V)|X R(V)>
            # NOTE not coupled no response
            - 4 * oe.contract("bc,rsac,rsab", prop_B_bb, mol.t_rsab, mol.t_rsab)
            + 4 * oe.contract("cs,rcab,rsab", prop_B_ss, mol.t_rsab, mol.t_rsab)
            # X_B -> X_B + [W_B, C_B - C_B^\dagger]_1
            + 8
            * oe.contract(
                "sb,rdac,rdae,Qes,Qbc",
                xt_B_sb,
                mol.t_rsab,
                mol.t_rsab,
                mol.Qbs,
                mol.Qbb,
            )
            + 16
            * oe.contract(
                "sb,rdac,reac,Qbs,Qde",
                xt_B_sb,
                mol.t_rsab,
                mol.t_rsab,
                mol.Qbs,
                mol.Qss,
            )
            - 8
            * oe.contract(
                "sb,rdac,reac,Qds,Qbe",
                xt_B_sb,
                mol.t_rsab,
                mol.t_rsab,
                mol.Qss,
                mol.Qbs,
            )
            - 16
            * oe.contract(
                "sb,rdac,rdae,Qbs,Qec",
                xt_B_sb,
                mol.t_rsab,
                mol.t_rsab,
                mol.Qbs,
                mol.Qbb,
            )
        ]
    )

    results["x2_disp"] = results["x2_disp_A"] + results["x2_disp_B"]

    ## Exchange-Dispersion S^2
    results["x2_exch-disp_S2"] = calc_exch_disp2_s2_property(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        prop_A_aa=prop_A_aa,
        prop_A_rr=prop_A_rr,
        prop_B_bb=prop_B_bb,
        prop_B_ss=prop_B_ss,
    )

    ## Exchange-Dispersion S^inf
    results["x2_exch-disp"] = (
        calc_exch_disp2_sinf_property(
            mol=mol,
            xt_A_ra=xt_A_ra,
            xt_B_sb=xt_B_sb,
            prop_A_aa=prop_A_aa,
            prop_A_rr=prop_A_rr,
            prop_B_bb=prop_B_bb,
            prop_B_ss=prop_B_ss,
        )
        - results["x2_disp"]
    )

    ### Summing up contributions

    # sum up first-order contributions
    results["x1_induced"] = results["x1_pol,r"] + results["x1_exch,r"]

    # induction with the coupled amplitudes unless only the uncoupled ones were made,
    # as in calc_sapt_energy
    ind = ",r" if coupled else ""

    # sum up all contributions
    results["x_induced"] = (
        results["x1_pol,r"]
        + results["x1_exch,r"]
        + results[f"x2_ind{ind}"]
        + results[f"x2_exch-ind{ind}"]
        + results["x2_disp"]
        + results["x2_exch-disp"]
    )
    # sum up all contributions with S^2 exchange terms
    results["x_induced_S2"] = (
        results["x1_pol,r"]
        + results["x1_exch,r"]
        + results[f"x2_ind{ind}"]
        + results[f"x2_exch-ind{ind}_S2"]
        + results["x2_disp"]
        + results["x2_exch-disp_S2"]
    )

    return results


def _calc_ind2_r_terms(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_rr: np.ndarray,
    prop_B_bb: np.ndarray,
    prop_B_ss: np.ndarray,
) -> tuple[float, float]:
    """
    The second-order induction terms with the coupled (CPSCF) induction amplitudes,
    `x2_ind,r_A` and `x2_ind,r_B`.

    Returns:
        tuple[float, float]: The terms of monomers A and B.
    """

    x2_ind_r_A = np.array(
        [
            +16
            * oe.contract(
                "ra,Qar,Qbs,sb", xt_A_ra, mol.Qar, mol.Qbs, mol.get_cpscf_sb()
            )
            - 4 * oe.contract("ra,ac,rc", mol.get_cpscf_ra(), mol.omegaB_aa, xt_A_ra)
            + 4 * oe.contract("ra,cr,ca", mol.get_cpscf_ra(), mol.omegaB_rr, xt_A_ra)
            - 2
            * oe.contract("ac,rc,ra", prop_A_aa, mol.get_cpscf_ra(), mol.get_cpscf_ra())
            + 2
            * oe.contract("cr,ca,ra", prop_A_rr, mol.get_cpscf_ra(), mol.get_cpscf_ra())
            # 2 Re <C^{(1)}_A | W_A C^{[1]}_A C^{(1)}_A > + Re <C^{(1)}_A C^{(1)}_A | W_A C^{[1]}_A >
            + 4
            * oe.contract(
                "ra,cr,de,Qad,Qec",
                xt_A_ra,
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qaa,
            )
            + 4
            * oe.contract(
                "ra,cd,de,Qer,Qac",
                xt_A_ra,
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qaa,
            )
            + 16
            * oe.contract(
                "ra,ac,de,Qcr,Qed",
                xt_A_ra,
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qrr,
                mol.Qar,
            )
            + 8
            * oe.contract(
                "ra,cd,ec,Qde,Qar",
                xt_A_ra,
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qrr,
                mol.Qar,
            )
            - 4
            * oe.contract(
                "ra,ac,de,Qcd,Qer",
                xt_A_ra,
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qrr,
                mol.Qar,
            )
            - 4
            * oe.contract(
                "ra,cd,ec,Qdr,Qae",
                xt_A_ra,
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qrr,
                mol.Qar,
            )
            - 16
            * oe.contract(
                "ra,cr,de,Qed,Qac",
                xt_A_ra,
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qaa,
            )
            - 8
            * oe.contract(
                "ra,cd,de,Qar,Qec",
                xt_A_ra,
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qaa,
            )
            + 4
            * oe.contract(
                "ra,cr,de,Qae,Qcd",
                xt_A_ra,
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qaa,
                mol.Qar,
            )
            - 4
            * oe.contract(
                "ra,ac,de,Qec,Qrd",
                xt_A_ra,
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qrr,
            )
        ]
    )

    x2_ind_r_B = np.array(
        [
            +16
            * oe.contract(
                "sb,Qar,Qbs,ra", xt_B_sb, mol.Qar, mol.Qbs, mol.get_cpscf_ra()
            )
            - 4 * oe.contract("sb,bc,sc", mol.get_cpscf_sb(), mol.omegaA_bb, xt_B_sb)
            + 4 * oe.contract("sb,cs,cb", mol.get_cpscf_sb(), mol.omegaA_ss, xt_B_sb)
            - 2
            * oe.contract("bc,sc,sb", prop_B_bb, mol.get_cpscf_sb(), mol.get_cpscf_sb())
            + 2
            * oe.contract("cs,cb,sb", prop_B_ss, mol.get_cpscf_sb(), mol.get_cpscf_sb())
            # 2 Re <C^{(1)}_B | W_B C^{[1]}_B C^{(1)}_B > + Re <C^{(1)}_B C^{(1)}_B | W_B C^{[1]}_B >
            + 4
            * oe.contract(
                "sb,cs,de,Qbd,Qec",
                xt_B_sb,
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbs,
                mol.Qbb,
            )
            + 4
            * oe.contract(
                "sb,cd,de,Qes,Qbc",
                xt_B_sb,
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbs,
                mol.Qbb,
            )
            + 16
            * oe.contract(
                "sb,bc,de,Qcs,Qed",
                xt_B_sb,
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qss,
                mol.Qbs,
            )
            + 8
            * oe.contract(
                "sb,cd,ec,Qde,Qbs",
                xt_B_sb,
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qss,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,bc,de,Qcd,Qes",
                xt_B_sb,
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qss,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,cd,ec,Qds,Qbe",
                xt_B_sb,
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qss,
                mol.Qbs,
            )
            - 16
            * oe.contract(
                "sb,cs,de,Qed,Qbc",
                xt_B_sb,
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbs,
                mol.Qbb,
            )
            - 8
            * oe.contract(
                "sb,cd,de,Qbs,Qec",
                xt_B_sb,
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbs,
                mol.Qbb,
            )
            + 4
            * oe.contract(
                "sb,cs,de,Qbe,Qcd",
                xt_B_sb,
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbb,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "sb,bc,de,Qec,Qsd",
                xt_B_sb,
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbs,
                mol.Qss,
            )
        ]
    )

    return x2_ind_r_A.item(), x2_ind_r_B.item()


def _add_exch_ind2_terms(
    results: pd.DataFrame,
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_rr: np.ndarray,
    prop_B_bb: np.ndarray,
    prop_B_ss: np.ndarray,
    coupled: bool,
) -> Generator:
    """
    Adds the S^2 and S^inf exchange-induction terms to `results`: `x2_exch-ind,r_S2`
    and `x2_exch-ind,r` with the coupled (CPSCF) induction amplitudes, or
    `x2_exch-ind_S2` and `x2_exch-ind` with the uncoupled ones. The induction term
    made with the same amplitudes (`x2_ind,r` or `x2_ind`) must already be there.

    A builder step, see `prop_sapt.cpscf_batching`: the coupled amplitude derivatives
    yield their CPSCF requests.
    """

    if coupled:
        t_ra, t_sb, tag = mol.get_cpscf_ra(), mol.get_cpscf_sb(), ",r"
    else:
        t_ra, t_sb, tag = mol.tB_ra, mol.tA_sb, ""

    u_ra = yield from get_u_ra_amplitudes(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        prop_A_aa=prop_A_aa,
        prop_A_rr=prop_A_rr,
        coupled=coupled,
    )
    u_sb = yield from get_u_sb_amplitudes(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        prop_B_bb=prop_B_bb,
        prop_B_ss=prop_B_ss,
        coupled=coupled,
    )
    amplitudes = dict(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        u_ra=u_ra,
        u_sb=u_sb,
        t_ra=t_ra,
        t_sb=t_sb,
    )

    results[f"x2_exch-ind{tag}_S2"] = calc_exch_ind2_resp_s2_property(**amplitudes)

    # the S^inf expression holds the induction term as well
    results[f"x2_exch-ind{tag}"] = (
        calc_exch_ind2_resp_sinf_property(**amplitudes) - results[f"x2_ind{tag}"]
    )


def calc_induced_dipole(dimer: Dimer, response: bool | str = True) -> pd.DataFrame:
    """
    Calculate interaction-induced dipole moment along X, Y and Z axes.

    The CPSCF equations of the three axes are solved together, a batch per monomer at a
    time, which costs much less than solving them axis by axis.

    Args:
        dimer (Dimer): A dimer system.
        response (bool | str, optional): Induction amplitudes, True (coupled), False
            (uncoupled) or "both"; see `perform_property_contractions`. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with results.
    """

    _check_response(response)

    # grab nuclear dipole moment
    nuc_dipole = dimer.dimer.nuclear_dipole()
    nuc_dipole = np.array([nuc_dipole[0], nuc_dipole[1], nuc_dipole[2]])

    # grab AO dipole moment matrices
    d_vec = dimer.mints.ao_dipole()
    d_vec = [np.asarray(elem) for elem in d_vec]

    # the CPSCF equations of the three axes are solved together, see `solve_batched`
    names = {axis: f"Property calculation along {axis}" for axis in "XYZ"}
    solved = solve_batched(
        dimer,
        {
            names[axis]: _property_builder(dimer, d_i, response)
            for d_i, axis in zip(d_vec, "XYZ")
        },
    )

    results = pd.concat(
        [solved[names[axis]].assign(axis=axis) for axis in "XYZ"], ignore_index=True
    )

    del d_vec

    results.set_index("axis", inplace=True)
    results["nuc"] = nuc_dipole

    return results


@trace_memory_peak
def calc_property(
    dimer: Dimer, prop: str | np.ndarray, response: bool | str = True, **kwargs
) -> pd.DataFrame:
    """
    Calculation of interaction-induced property.

    The property is the field derivative of the energy `calc_sapt_energy` gives for
    the same `response`.

    Known issue: the S^inf exchange-dispersion term `x2_exch-disp` is not yet the field
    derivative of `EXCH-DISP2`. For the dipole of He-H2 in aug-cc-pVTZ it is 7.2e-5 a.u.
    off the finite-field derivative (about 18%), and the total `x_induced` carries that
    error. Every term of `x_induced_S2` agrees with its finite-field derivative. See
    "Known limitations" in the usage guide.

    Args:
        dimer (Dimer): A dimer system.
        prop (str | np.ndarray): Property to calculate. Either name of already implemented
            property, e.g. dipole moment (prop="dipole") or an array with matrix representation,
            in AO basis, of the operator corresponding to a property to calculate.
        response (bool | str, optional): Induction amplitudes of the second-order
            induction terms. True: the coupled (CPSCF) ones, `x2_ind,r`,
            `x2_exch-ind,r_S2` and `x2_exch-ind,r` (plus the uncoupled `x2_ind`).
            False: the uncoupled ones, `x2_ind`, `x2_exch-ind_S2` and `x2_exch-ind`,
            with no CPSCF equations solved for them. "both": both sets. The totals take
            the coupled terms unless `response` is False. The monomers relax in the
            field in every mode, so the other terms do not depend on it. Defaults to
            True.

    Keyword Args:
        results (str): Path of the CSV file the results are saved to. Defaults to
            "results.csv".

    Returns:
        pd.DataFrame: DataFrame with results.
    """

    try:
        _check_response(response)
    except ValueError:
        psi4.core.clean()
        raise

    with CalcTimer("Interaction-induced property calculations"):
        ### Results output file
        results_fname = kwargs.get("results", "results.csv")

        # array check first: comparing an array with a string is elementwise
        if isinstance(prop, np.ndarray):
            ### Property matrix is given
            results = perform_property_contractions(dimer, prop, response=response)

        elif isinstance(prop, str) and prop == "dipole":
            ### Dipole moment calculations
            results = calc_induced_dipole(dimer, response=response)

        else:
            psi4.core.clean()
            raise ValueError(
                f"Property {prop} is not implemented. "
                "Please provide a valid property name or a property matrix."
            )

        ### Results saving to file
        results.to_csv(results_fname)

    psi4.core.clean()

    return results
