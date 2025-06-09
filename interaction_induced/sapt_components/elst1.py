import numpy as np

from ..molecule import Dimer
from ..utils import CalcTimer


def calc_elst1_energy(dimer: Dimer):
    """Calculate the first-order SAPT electrostatic interaction energy.

    Args:
        dimer (Dimer): Dimer object containing the necessary matrices and tensors.

    Returns:
        _type_: First-order SAPT electrostatic interaction energy.
    """

    with CalcTimer("Electrostatic energy calculation"):

        density_A = dimer.orbitals["a"] @ dimer.orbitals["a"].T
        density_B = dimer.orbitals["b"] @ dimer.orbitals["b"].T

        elst1 = dimer.nuc_rep
        elst1 += 2 * np.trace(density_B @ dimer.V_A)
        elst1 += 2 * np.trace(density_A @ dimer.V_B)
        elst1 += 4 * np.trace(density_A @ dimer.J_B)

    return elst1
