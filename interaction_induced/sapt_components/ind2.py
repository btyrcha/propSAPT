import pandas as pd
import opt_einsum as oe

from .exch_ind2 import (
    calc_exch_ind2_a_energy,
    calc_exch_ind2_b_energy,
    calc_exch_ind2_s2_a_energy,
    calc_exch_ind2_s2_b_energy,
)

from ..molecule import Dimer


def calc_ind2_energy(dimer: Dimer) -> pd.Series:
    """Calculate the second-order SAPT induction and exchange-induction energies.
    IMPORTANT: This function does not include orbital relaxation.

    Args:
        dimer (Dimer): A Dimer object containing the molecular information.

    Returns:
        pd.Series: A pandas Series containing the calculated induction energies.
    """

    # Calculate the second-order induction energies
    ind2_a = 2 * oe.contract("ra,ar", dimer.tB_ra, dimer.omegaB_ar)
    ind2_b = 2 * oe.contract("sb,bs", dimer.tA_sb, dimer.omegaA_bs)

    # Store the results in a pandas Series
    ind_results = pd.Series()

    ind_results["IND2_A"] = ind2_a
    ind_results["IND2_B"] = ind2_b
    ind_results["IND2"] = ind_results["IND2_A"] + ind_results["IND2_B"]

    # Calculate the exchange-induction energy with Sinfinity
    ind_results["EXCH-IND2_A"] = calc_exch_ind2_a_energy(dimer, coupled=False)
    ind_results["EXCH-IND2_B"] = calc_exch_ind2_b_energy(dimer, coupled=False)
    ind_results["EXCH-IND2"] = ind_results["EXCH-IND2_A"] + ind_results["EXCH-IND2_B"]

    # Calculate the exchange-induction energy with S^2
    ind_results["EXCH-IND2(S^2)_A"] = calc_exch_ind2_s2_a_energy(dimer, coupled=False)
    ind_results["EXCH-IND2(S^2)_B"] = calc_exch_ind2_s2_b_energy(dimer, coupled=False)
    ind_results["EXCH-IND2(S^2)"] = (
        ind_results["EXCH-IND2(S^2)_A"] + ind_results["EXCH-IND2(S^2)_B"]
    )

    return ind_results


def calc_ind2_r_energy(dimer: Dimer) -> pd.Series:
    """Calculate the second-order SAPT induction and exchange-induction energies
    including the orbital relaxation effects.

    Args:
        dimer (Dimer): A Dimer object containing the molecular information.

    Returns:
        pd.Series: A pandas Series containing the calculated induction energies.
    """

    # CPSCF calculations for the induction terms
    _, ind2_r_a = dimer.cpscf("A", ind=True)
    _, ind2_r_b = dimer.cpscf("B", ind=True)

    # Store the results in a pandas Series
    ind_results = pd.Series()

    ind_results["IND2,R_A"] = ind2_r_a
    ind_results["IND2,R_B"] = ind2_r_b
    ind_results["IND2,R"] = ind_results["IND2,R_A"] + ind_results["IND2,R_B"]

    # Calculate the exchange-induction energy with Sinfinity
    ind_results["EXCH-IND2,R_A"] = calc_exch_ind2_a_energy(dimer)
    ind_results["EXCH-IND2,R_B"] = calc_exch_ind2_b_energy(dimer)
    ind_results["EXCH-IND2,R"] = (
        ind_results["EXCH-IND2,R_A"] + ind_results["EXCH-IND2,R_B"]
    )

    # Calculate the exchange-induction energy with S^2
    ind_results["EXCH-IND2,R(S^2)_A"] = calc_exch_ind2_s2_a_energy(dimer)
    ind_results["EXCH-IND2,R(S^2)_B"] = calc_exch_ind2_s2_b_energy(dimer)
    ind_results["EXCH-IND2,R(S^2)"] = (
        ind_results["EXCH-IND2,R(S^2)_A"] + ind_results["EXCH-IND2,R(S^2)_B"]
    )

    return ind_results
