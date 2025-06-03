"""
SAPT Driver Module
This module provides a function to calculate the SAPT energy components for a dimer system.
"""

import pandas as pd

from .molecule import Dimer
from .utils import trace_memory_peak

from .sapt_components.elst1 import calc_elst1_energy
from .sapt_components.exch1 import calc_exch1_energy, calc_exch1_s2_energy
from .sapt_components.ind2 import calc_ind2_energy, calc_ind2_r_energy
from .sapt_components.disp2 import calc_disp2_energy


@trace_memory_peak
def calc_sapt_energy(dimer: Dimer, **kwargs) -> pd.Series:
    """Calculate SAPT energy for a given dimer.

    This function computes the SAPT energy components for a dimer system,
    for now only SAPT0 of SAPT(DFT) levels (with uncoupled dispersion) are implemented.
    The results are stored in a pandas Series and saved
    to a CSV file specified in the `kwargs`.

    The SAPT energy components are calculated as follows:
    - ELST1: First-order electrostatic energy
    - EXCH1: First-order exchange energy
    - EXCH1(S^2): First-order exchange energy with S^2 correction
    - IND2,R: Second-order induction energy (response)
    - EXCH-IND2,R: Exchange-induction energy (response)
    - DISP2: Second-order dispersion energy
    - EXCH-DISP2: Exchange-dispersion energy


    Args:
        dimer (Dimer): A dimer system for which to calculate the SAPT energy.

    Kwargs:
        results (str): Path to save the results CSV file. Defaults to "results.csv".

    Returns:
        pd.Series: A pandas Series containing the SAPT energy components.
    """

    # TODO: Add printout for each term
    # TODO: Time the calculations

    results = pd.Series()

    # First-order terms
    results["ELST1"] = calc_elst1_energy(dimer)
    results["EXCH1"] = calc_exch1_energy(dimer)
    results["EXCH1(S^2)"] = calc_exch1_s2_energy(dimer)

    # Second-order induction terms
    ind2_results = calc_ind2_energy(dimer)
    results = pd.concat([results, ind2_results])

    ind2_resp_results = calc_ind2_r_energy(dimer)
    results = pd.concat([results, ind2_resp_results])

    # Second-order dispersion terms
    disp2_results = calc_disp2_energy(dimer)
    results = pd.concat([results, disp2_results])

    # Calculate total energy
    results["TOTAL"] = (
        results["ELST1"]
        + results["EXCH1"]
        + results["IND2,R"]
        + results["EXCH-IND2,R"]
        + results["DISP2"]
        + results["EXCH-DISP2"]
    )

    # Save results to file
    results_fname = kwargs.get("results", "results.csv")
    results.to_csv(results_fname)

    return results
