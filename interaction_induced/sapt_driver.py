import pandas as pd

from .molecule import Dimer

from .sapt_components.elst1 import calc_elst1_energy
from .sapt_components.exch1 import calc_exch1_energy, calc_exch1_s2_energy
from .sapt_components.ind2 import calc_ind2_energy, calc_ind2_r_energy
from .sapt_components.disp2 import calc_disp2_energy


def calc_sapt_energy(dimer: Dimer) -> pd.Series:

    # TODO: Add docstring
    # TODO: Add printout for each term

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

    return results
