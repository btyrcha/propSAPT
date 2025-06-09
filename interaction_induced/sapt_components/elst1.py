import opt_einsum as oe

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

        vA_bb = dimer.potential("bb", "A")
        vB_aa = dimer.potential("aa", "B")

        elst1 = dimer.nuc_rep
        elst1 += 2 * oe.contract("bb", vA_bb)
        elst1 += 2 * oe.contract("aa", vB_aa)
        elst1 += 4 * oe.contract("Qaa, Qbb", dimer.Qaa, dimer.Qbb)

    return elst1
