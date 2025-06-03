import opt_einsum as oe

from ..molecule import Dimer


def calc_elst1_energy(dimer: Dimer):
    """Calculate the first-order SAPT electrostatic interaction energy.

    Args:
        dimer (Dimer): Dimer object containing the necessary matrices and tensors.

    Returns:
        _type_: First-order SAPT electrostatic interaction energy.
    """

    vA_aa = dimer.potential("aa", "A")
    vB_bb = dimer.potential("bb", "B")

    elst1 = dimer.nuc_rep
    elst1 += 2 * oe.contract("aa", vA_aa)
    elst1 += +2 * oe.contract("bb", vB_bb)
    elst1 += +4 * oe.contract("Qaa, Qbb", dimer.Qaa, dimer.Qbb)

    return elst1
