import numpy as np

from ..molecule import Dimer
from ..utils import CalcTimer


def calc_elst1_energy(mol: Dimer):
    """Calculate the first-order SAPT electrostatic interaction energy.

    Args:
        mol (Dimer): Dimer object containing the necessary matrices and tensors.

    Returns:
        _type_: First-order SAPT electrostatic interaction energy.
    """

    with CalcTimer("Electrostatic energy calculation"):

        density_A = mol.orbitals["a"] @ mol.orbitals["a"].T
        density_B = mol.orbitals["b"] @ mol.orbitals["b"].T

        elst1 = mol.nuc_rep
        elst1 += 2 * np.trace(density_B @ mol.V_A)
        elst1 += 2 * np.trace(density_A @ mol.V_B)
        elst1 += 4 * np.trace(density_A @ mol.J_B)

    return elst1
