from time import time
import numpy as np
import psi4
from .sinfinity import sinfinity
from .cube_utils import prepare_grid, calculate_isocontour


class Molecule(sinfinity):
    """
    Molecule class.

    Initialises and stores all necessary values
    for 'interaction_induced' clalculations.
    Performs SCF calculations for the monomers.
    """

    def __init__(self, geometry, reference="RHF", memory=8, **kwargs):

        # starting initialisation
        t_start = time()
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\nInitializing Molecule object...\n\n")

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
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\n")

    def get_psi4_molecule(self) -> psi4.core.Molecule:
        """
        Get 'psi4.core.Molecule' object of the molecule.
        """
        return self.dimer

    def get_psi4_wavefunction(self) -> psi4.core.Wavefunction:
        """
        Get 'psi4.core.Wavefunction' object of the molecule.
        """
        mol_object = self.get_psi4_molecule()
        return psi4.core.Wavefunction.build(mol_object)

    def get_psi4_basisset(self) -> psi4.core.BasisSet:
        """
        Get 'psi4.core.BasisSet' object of the molecule.
        """
        wfn_object = self.get_psi4_wavefunction()
        return wfn_object.basisset()

    def save_cube(self, matrix: np.ndarray, filename: str = "density.cube", **kwargs):
        """
        Evaluate `matrix` values on a grid and save to a .cube file.
        """

        # initialize
        t_start = time()
        psi4.core.print_out("\n")
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\n\n")
        psi4.core.print_out("        |------------------------------------|        \n")
        psi4.core.print_out("        |            `save_cube`             |        \n")
        psi4.core.print_out("        |------------------------------------|        \n")
        psi4.core.print_out(f"\nSaving volumetric data to file `{filename}`.\n\n")

        # get basis set and geometry
        basisset = self.get_psi4_basisset()
        basisset.print_out()
        geo_matrix, _, _, elez, _ = self.get_psi4_molecule().to_arrays()

        # default grid parameters in bohr
        grid_step = kwargs.get("grid_step", 0.2)
        grid_overage = kwargs.get("grid_overage", 4.0)

        grid = prepare_grid(geo_matrix, grid_step, grid_overage)

        # compute values on the grid on the fly
        values = np.zeros((grid["n_x"], grid["n_y"], grid["n_z"]))
        for i, x in enumerate(grid["x"]):
            for j, y in enumerate(grid["y"]):
                for k, z in enumerate(grid["z"]):
                    basis_vals = basisset.compute_phi(x, y, z)
                    values[i][j][k] = basis_vals.dot(matrix).dot(basis_vals.T)

        # get isocontour values
        isovalues = calculate_isocontour(values)

        # create and save cube string
        with open(filename, "w", encoding="utf-8") as file:

            # write a header
            file.write("interaction-induced .cube file\n")
            file.write(f"isovalues for 85% of the denisty: ({isovalues[0]:.6E}, {isovalues[1]:.6E})\n")

            # wirte number of atoms and begining of the grid
            file.write(f"{len(elez):6d}  "
                       f"{grid['x'][0]: .6f}  "
                       f"{grid['y'][0]: .6f}  "
                       f"{grid['z'][0]: .6f}\n")

            # write grid details
            file.write(f"{grid["n_x"]:6d}  {grid["step_x"]: .6f}   0.000000   0.000000\n")
            file.write(f"{grid["n_y"]:6d}   0.000000  {grid["step_y"]: .6f}   0.000000\n")
            file.write(f"{grid["n_z"]:6d}   0.000000   0.000000  {grid["step_z"]: .6f}\n")

            # write geometry
            for atom_xyz, atom_num in zip(geo_matrix, elez):
                file.write(
                    f"{int(atom_num):3d}  "
                    " 0.000000  "
                    f"{atom_xyz[0]: .6f}  "
                    f"{atom_xyz[1]: .6f}  "
                    f"{atom_xyz[2]: .6f}\n"
                )

            # write volumetric information
            count = 0
            for value in values.flatten():

                file.write(f"{value: .5E} ")
                count += 1

                if count % 6 == 0:
                    file.write("\n")

        # finilize
        psi4.core.print_out("\n")
        psi4.core.print_out(
            f"...finished evaluating density on the grid in {(time() - t_start):5.2f} seconds.\n"
        )
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\n\n")
