import pandas as pd
import opt_einsum as oe

from .exch_disp2 import calc_exch_disp2_energy, calc_exch_disp2_s2_energy

from ..molecule import Dimer
from ..utils import CalcTimer


def calc_disp2_energy(mol: Dimer) -> pd.Series:
    """Calculate the second-order SAPT dispersion energy.
    IMPORTANT: This function calculates the dispersion energy in the uncoupled approximation.

    Args:
        mol (Dimer): A Dimer object containing the molecular information.

    Returns:
        pd.Series: A pandas Series containing the calculated dispersion energy.
    """

    with CalcTimer("Second-order dispersion energy calculation"):

        # Calculate the second-order dispersion energy
        disp2 = 4 * oe.contract("rsab,Qar,Qbs", mol.t_rsab, mol.Qar, mol.Qbs)

        # Combine the results into a pandas Series
        results = pd.Series(dtype=float)
        results["DISP2"] = disp2

    with CalcTimer("Second-order Exchange-Dispersion (Sinf) energy calculation"):
        # Calculate the exchange-dispersion energy with Sinfinity
        results["EXCH-DISP2"] = calc_exch_disp2_energy(mol)

    with CalcTimer("Second-order Exchange-Dispersion (S^2) energy calculation"):
        # Calculate the exchange-dispersion energy with S^2
        results["EXCH-DISP2(S^2)"] = calc_exch_disp2_s2_energy(mol)

    return results
