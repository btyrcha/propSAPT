from time import time
import psi4
from .sinfinity import sinfinity


class Molecule(sinfinity):
    """
    Molecule class.

    Initialises and stores all necessary values
    for 'interaction_induced' clalculations.
    Performs SCF calculations for the monomers.
    """

    def __init__(self, geometry, reference="RHF", memory=8, **kwargs):

        ### starting initialisation
        t_start = time()
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\nInitializing Molecule object...\n\n")

        # create Psi4 Molecule object
        self.geometry = geometry
        self.dimer = psi4.geometry(geometry)
        self.dimer.print_out_in_angstrom()

        # initialize parent class
        super().__init__(self.dimer, reference, memory, **kwargs)

        ### Print time
        psi4.core.print_out(
            f"...finished initializing Molecule object in {(time() - t_start):5.2f} seconds.\n"
        )
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\n")

    def get_psi4_molecule(self) -> psi4.core.Molecule:
        """
        Get 'psi4.core.Molecule' object of the molecule.
        """
        return self.dimer
