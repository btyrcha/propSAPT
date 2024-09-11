from time import time
import psi4
from .sinfinity import sinfinity


class Molecule(sinfinity):

    def __init__(self, geometry, reference="RHF", memory=8, **kwargs):

        ### starting initialisation
        t_start = time()
        psi4.core.print_out("\nInitializing Molecule object...\n\n")

        # create Psi4 Molecule object
        self.geometry = geometry
        self.dimer = psi4.geometry(geometry)

        # initialize parent class
        super().__init__(self.dimer, reference, memory, **kwargs)

        ### Print time
        psi4.core.print_out(
            f"...finished initializing Molecule object in {(time() - t_start):5.2f} seconds.\n\n"
        )
