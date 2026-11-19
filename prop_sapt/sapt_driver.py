"""
SAPT Driver Module
This module provides a function to calculate the SAPT energy components for a dimer system.
"""

import logging

import psi4
import pandas as pd

from .molecule import Dimer
from .utils import trace_memory_peak, CalcTimer, energy_printer

from .sapt_components import (
    calc_elst1_energy,
    calc_exch1_energy,
    calc_exch1_s2_energy,
    calc_ind2_energy,
    calc_ind2_r_energy,
    calc_disp2_energy,
)

logger = logging.getLogger(__name__)


def print_sapt_summary(results: pd.Series, **kwargs):
    """Prints a summary of the SAPT energy components.

    Args:
        results (pd.Series):  A pandas Series containing the SAPT energy components.
    """

    line_width = 93

    # Header
    logger.info(
        "            |------------------------------------| \n"
        "            |         SAPT Summary Table         | \n"
        "            |------------------------------------| "
    )

    # Table header
    logger.info(
        "=" * line_width
        + "\n Term                        Value (mEh)           Value (kcal/mol)"
        + "           Value (kJ/mol)\n"
        + "=" * line_width
    )

    # Electrostatics
    logger.info(f"Electrostatics\n{'-' * line_width}")
    energy_printer("ELST1", results["ELST1"], output="psi4")

    # Exchange
    logger.info(f"Exchange\n{'-' * line_width}")
    energy_printer("EXCH1", results["EXCH1"], output="psi4")
    energy_printer("EXCH1(S^2)", results["EXCH1(S^2)"], output="psi4")

    # Induction
    logger.info(f"Induction\n{'-' * line_width}")
    if kwargs.get("response") is True:
        energy_printer("IND2,R_A", results["IND2,R_A"], output="psi4")
        energy_printer("IND2,R_B", results["IND2,R_B"], output="psi4")
        energy_printer("IND2,R", results["IND2,R"], output="psi4")
        energy_printer("EXCH-IND2,R_A", results["EXCH-IND2,R_A"], output="psi4")
        energy_printer("EXCH-IND2,R_B", results["EXCH-IND2,R_B"], output="psi4")
        energy_printer("EXCH-IND2,R", results["EXCH-IND2,R"], output="psi4")
    else:
        energy_printer("IND2_A", results["IND2_A"], output="psi4")
        energy_printer("IND2_B", results["IND2_B"], output="psi4")
        energy_printer("IND2", results["IND2"], output="psi4")
        energy_printer("EXCH-IND2_A", results["EXCH-IND2_A"], output="psi4")
        energy_printer("EXCH-IND2_B", results["EXCH-IND2_B"], output="psi4")
        energy_printer("EXCH-IND2", results["EXCH-IND2"], output="psi4")

    # Dispersion
    logger.info(f"Dispersion\n{'-' * line_width}")
    energy_printer("DISP2", results["DISP2"], output="psi4")
    energy_printer("EXCH-DISP2", results["EXCH-DISP2"], output="psi4")

    # Total energy
    energy_printer("TOTAL", results["TOTAL"], output="psi4")
    logger.info("=" * line_width)


@trace_memory_peak
def calc_sapt_energy(dimer: Dimer, **kwargs) -> pd.Series:
    """Calculate SAPT energy for a given dimer.

    This function computes the SAPT energy components for a dimer system: SAPT0 for an
    RHF reference, SAPT(DFT) for an RKS one. The dispersion is the uncoupled one in both
    cases. With RHF that is the SAPT0 dispersion. With RKS it is not the coupled
    dispersion that Psi4's SAPT(DFT) uses by default, so SAPT(DFT) dispersion and total
    energies differ from Psi4's. Coupled dispersion is not implemented.
    The results are returned as a pandas Series and, unless `results=False`, saved to a
    CSV file, "sapt_results.csv" by default.

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

    Keyword Args:
        results (str | bool): Path of the CSV file the results are saved to. Defaults to
            "sapt_results.csv", which `True` also gives; `False` (or None) saves nothing.
        response (bool | str): Induction amplitudes: True, the coupled (CPSCF) ones;
            False, the uncoupled ones; "both". Defaults to True.

    Returns:
        pd.Series: A pandas Series containing the SAPT energy components.
    """

    # Print header
    logger.info("*" * 80)
    logger.info(
        "        |-------------------------------------|        \n"
        "        |        Second-Quantized SAPT        |        \n"
        "        |-------------------------------------|        "
    )
    psi4.core.tstart()

    with CalcTimer("SAPT energy calculations"):

        pd_results_series = pd.Series(dtype=float)

        # First-order terms
        pd_results_series["ELST1"] = calc_elst1_energy(dimer)
        pd_results_series["EXCH1"] = calc_exch1_energy(dimer)
        pd_results_series["EXCH1(S^2)"] = calc_exch1_s2_energy(dimer)

        # Second-order induction terms
        if kwargs.get("response") is None:
            kwargs["response"] = True  # Default to True if not specified

        if kwargs.get("response") is True:
            ind2_results = calc_ind2_r_energy(dimer)
            pd_results_series = pd.concat([pd_results_series, ind2_results])

        elif kwargs.get("response") is False:
            ind2_results = calc_ind2_energy(dimer)
            pd_results_series = pd.concat([pd_results_series, ind2_results])

        elif kwargs.get("response") == "both":
            in2_nonresp_results = calc_ind2_energy(dimer)
            ind2_resp_results = calc_ind2_r_energy(dimer)
            pd_results_series = pd.concat(
                [pd_results_series, in2_nonresp_results, ind2_resp_results]
            )

        else:
            raise ValueError(
                "Invalid value for 'response'. Must be True, False or 'both'."
            )

        # Second-order dispersion terms
        disp2_results = calc_disp2_energy(dimer)
        pd_results_series = pd.concat([pd_results_series, disp2_results])

        # Calculate total energy
        if kwargs.get("response") is True or kwargs.get("response") == "both":
            ind_key = "IND2,R"
            exch_ind_key = "EXCH-IND2,R"
        else:
            ind_key = "IND2"
            exch_ind_key = "EXCH-IND2"

        pd_results_series["TOTAL"] = (
            pd_results_series["ELST1"]
            + pd_results_series["EXCH1"]
            + pd_results_series[ind_key]
            + pd_results_series[exch_ind_key]
            + pd_results_series["DISP2"]
            + pd_results_series["EXCH-DISP2"]
        )

        pd_results_series["TOTAL(S^2)"] = (
            pd_results_series["ELST1"]
            + pd_results_series["EXCH1"]
            + pd_results_series[ind_key]
            + pd_results_series[exch_ind_key + "(S^2)"]
            + pd_results_series["DISP2"]
            + pd_results_series["EXCH-DISP2(S^2)"]
        )

    # Print results
    print_sapt_summary(pd_results_series, response=kwargs.get("response"))

    # Save results to file, unless results=False
    results_fname = kwargs.get("results", True)
    if results_fname is True:
        results_fname = "sapt_results.csv"
    if results_fname:
        pd_results_series.to_csv(results_fname)

    psi4.core.clean()

    return pd_results_series
