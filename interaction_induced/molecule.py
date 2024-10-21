from time import time
import numpy as np
import psi4
from .sinfinity import sinfinity
from .cubes import Cube, prepare_grid, calculate_isocontour


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

    def make_cube(
        self, matrix: np.ndarray, obj_type: str = "density", **kwargs
    ) -> Cube:
        """
        Create a `Cube` object with volumetric data
        of the given `matrix` calculated on a grid.
        """

        if obj_type not in ["density", "orbital"]:
            raise ValueError(
                f"`obj_type` should be \"density\" or \"orbital\", was {obj_type}!"
            )

        # initialize
        t_start = time()
        psi4.core.print_out("\n")
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\n\n")
        psi4.core.print_out("        |------------------------------------|        \n")
        psi4.core.print_out("        |            `make_cube`             |        \n")
        psi4.core.print_out("        |------------------------------------|        \n")
        psi4.core.print_out("\n")
        psi4.core.tstart()

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

                    if obj_type == "density":
                        values[i][j][k] = basis_vals.dot(matrix).dot(basis_vals.T)

                    elif obj_type == "orbital":
                        values[i][j][k] = basis_vals.dot(matrix)

        # get isocontour values
        iso_sum_level = kwargs.get("iso_sum_level", 0.85)
        isovalues = calculate_isocontour(
            values, threshold=iso_sum_level, obj_type=obj_type
        )

        cube_dict = {
            "comment1": "interaction-induced .cube file",
            "comment2": f"isovalues for {iso_sum_level*100:.0f}%"
            f" of the {obj_type}: ({isovalues[0]:.6E}, {isovalues[1]:.6E})",
            "origin": [grid['x'][0], grid['y'][0], grid['z'][0]],
            "n_atoms": len(elez),
            "atoms": [
                (int(atom_num), float(0.0), atom_xyz)
                for atom_xyz, atom_num in zip(geo_matrix, elez)
            ],
            "n_x": grid["n_x"],
            "n_y": grid["n_y"],
            "n_z": grid["n_z"],
            "x_vector": np.array([grid["step_x"], 0.0, 0.0]),
            "y_vector": np.array([0.0, grid["step_y"], 0.0]),
            "z_vector": np.array([0.0, 0.0, grid["step_z"]]),
            "volumetric_data": values,
        }

        # finilize
        psi4.core.print_out("\n")
        psi4.core.print_out(
            f"...finished evaluating density on the grid in {(time() - t_start):5.2f} seconds.\n"
        )
        psi4.core.tstop()
        psi4.core.print_out("\n")
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\n\n")

        return Cube(**cube_dict)

    def save_cube(
        self,
        matrix: np.ndarray,
        obj_type: str = "density",
        filename: str = "density.cube",
        **kwargs,
    ):
        """
        Evaluate `matrix` values on a grid and save to a .cube file.
        """

        self.make_cube(matrix, obj_type=obj_type, **kwargs).save(filename)
