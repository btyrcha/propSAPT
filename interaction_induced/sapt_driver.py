import pandas as pd

from .molecule import Dimer

from .sapt_components.elst1 import calc_elst1_energy
from .sapt_components.exch1 import calc_exch1_energy, calc_exch1_s2_energy
from .sapt_components.ind2 import calc_ind2_energy, calc_ind2_r_energy
from .sapt_components.exch_ind2 import (
    calc_exch_ind2_energy,
    calc_exch_ind2_r_energy,
    calc_exch_ind2_s2_energy,
    calc_exch_ind2_r_s2_energy,
)
from .sapt_components.disp2 import calc_disp2_energy
from .sapt_components.exch_disp2 import (
    calc_exch_disp2_energy,
    calc_exch_disp2_s2_energy,
)


def calc_sapt_energy(dimer: Dimer) -> pd.Series:

    results = pd.Series(
        index=[
            "ELST1",
            "EXCH1",
            "EXCH1(S^2)",
            "IND2",
            "IND2,R",
            "EXCH-IND2",
            "EXCH-IND2,R",
            "EXCH-IND2(S^2)",
            "EXCH-IND2,R(S^2)",
            "DISP2",
            "EXCH-DISP2",
            "EXCH-DISP2(S^2)",
            "TOTAL",
        ]
    )

    results["ELST1"] = calc_elst1_energy(dimer)
    results["EXCH1"] = calc_exch1_energy(dimer)
    results["EXCH1(S^2)"] = calc_exch1_s2_energy(dimer)
    results["IND2"] = calc_ind2_energy(dimer)
    results["IND2,R"] = calc_ind2_r_energy(dimer)
    results["EXCH-IND2"] = calc_exch_ind2_energy(dimer)
    results["EXCH-IND2,R"] = calc_exch_ind2_r_energy(dimer)
    results["EXCH-IND2(S^2)"] = calc_exch_ind2_s2_energy(dimer)
    results["EXCH-IND2,R(S^2)"] = calc_exch_ind2_r_s2_energy(dimer)
    results["DISP2"] = calc_disp2_energy(dimer)
    results["EXCH-DISP2"] = calc_exch_disp2_energy(dimer)
    results["EXCH-DISP2(S^2)"] = calc_exch_disp2_s2_energy(dimer)

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
