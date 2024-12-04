from time import time
from collections.abc import Iterable
import numpy as np
import psi4
from .sinfinity import sinfinity
from .cubes import Cube, make_cube


class Dimer(sinfinity):
    """
    Core dimer (molecule-like) class.

    Initialises and stores all necessary values
    for 'interaction_induced' clalculations.
    Performs SCF calculations for the monomers.
    """

    def __init__(self, geometry, reference="RHF", memory=8, **kwargs):

        # starting initialisation
        t_start = time()
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\nInitializing Molecule object...\n\n")
        psi4.core.tstart()

        # create Psi4 Molecule object
        self.geometry = geometry
        self.dimer = psi4.geometry(geometry)
        self.dimer.print_out_in_angstrom()

        # initialize parent class
        super().__init__(self.dimer, reference, memory, **kwargs)

        # print time
        psi4.core.print_out(
            f"...finished initializing Molecule object in {(time() - t_start):5.2f} seconds.\n"
        )
        psi4.core.tstop()
        psi4.core.print_out("\n")
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\n")

    def get_psi4_molecule(self) -> psi4.core.Molecule:
        """
        Get 'psi4.core.Molecule' object of the dimer.

        Returns:
            psi4.core.Molecule: Psi4 molecule object of the dimer.
        """

        return self.dimer

    def get_psi4_wavefunction(self) -> psi4.core.Wavefunction:
        """
        Get 'psi4.core.Wavefunction' object of the dimer.

        Returns:
            psi4.core.Wavefunction: Psi4 wavefunction object of the dimer.
        """

        mol_object = self.get_psi4_molecule()
        return psi4.core.Wavefunction.build(mol_object)

    def get_psi4_basisset(self) -> psi4.core.BasisSet:
        """
        Get 'psi4.core.BasisSet' object of the molecule.

        Returns:
            psi4.core.BasisSet: Psi4 basis set object of the dimer.
        """

        wfn_object = self.get_psi4_wavefunction()
        return wfn_object.basisset()

    def make_cube(
        self,
        matrix: np.ndarray | Iterable[np.ndarray],
        obj_type: str | Iterable[str] = "density",
        **kwargs,
    ) -> Cube | list[Cube]:
        """
        Create a `Cube` object with volumetric data of the given `matrix` calculated on a grid.

        Args:
            matrix (np.ndarray): Either density matrix or orbital coefficients vector.
            obj_type (str, optional): Specifies the `matrix` argument type, either "density" or
                "orbital". Defaults to "density".

        Returns:
            Cube: Resulting Cube.
        """

        return make_cube(
            self.get_psi4_molecule(), matrix=matrix, obj_type=obj_type, **kwargs
        )

    def save_cube(
        self,
        matrix: np.ndarray | Iterable[np.ndarray],
        obj_type: str | Iterable[str] = "density",
        filename: str | Iterable[str] = "density.cube",
        **kwargs,
    ):
        """
        Evaluate `matrix` values on a grid and save to a .cube file.

        Args:
            matrix (np.ndarray): Either density matrix or orbital coefficients vector.
            obj_type (str, optional): Specifies the `matrix` argument type, either "density" or
                "orbital". Defaults to "density".
            filename (str, optional): File path to save resulting cube. Defaults to "density.cube".
        """

        cubes = self.make_cube(matrix, obj_type=obj_type, **kwargs)

        if isinstance(cubes, Iterable):
            for cube, fname in zip(cubes, filename):
                cube.save(fname)

        else:
            cubes.save(filename)
