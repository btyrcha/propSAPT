"""
Cube files related things.

Cube file refefence:
https://paulbourke.net/dataformats/cube/
"""

from time import time
from collections.abc import Iterable
from typing import Sequence
import numpy as np
import psi4


class Cube:
    """
    Class for storing cube files.

    Params:
        comment1 (str): First comment line of `.cube` file.
        comment2 (str): Second comment line of `.cube` file.
        origin (list[float]): Origin of the grid.
        n_atoms (int): Number of atoms.
        atoms (list): List of tuples describing molecule.
            Should look like: (atomic_number, charge, xyz_coordinates).
            Coordinates should be in bohr.
        n_x (int): Number of grid points in x direction.
        n_y (int): Number of grid points in y direction.
        n_z (int): Number of grid points in z direction.
        x_vector (np.ndarray): Grid step in x direction.
        y_vector (np.ndarray): Grid step in y direction.
        z_vector (np.ndarray): Grid step in z direction.
        volumetric_data (np.ndarray): Array with values at the grid points.

    Returns:
        Cube: Initialised cube object.
    """

    def __init__(self, **kwargs):

        self.comment1: str = kwargs.get("comment1", "")
        self.comment2: str = kwargs.get("comment2", "")

        if kwargs.get("origin") is not None:
            self.origin: list[float] | np.ndarray = kwargs["origin"]
        else:
            raise ValueError("`origin` must be provided!")

        if kwargs.get("n_atoms") is not None:
            self.n_atoms: int = kwargs["n_atoms"]
        else:
            raise ValueError("`n_atoms` must be provided!")

        if kwargs.get("atoms") is not None:
            self.atoms: list[tuple[int, float, np.ndarray]] = kwargs["atoms"]
        else:
            raise ValueError("`atoms` must be provided!")

        if kwargs.get("n_x") is not None:
            self.n_x: int = kwargs["n_x"]
        else:
            raise ValueError("`n_x` must be provided!")

        if kwargs.get("n_y") is not None:
            self.n_y: int = kwargs["n_y"]
        else:
            raise ValueError("`n_y` must be provided!")

        if kwargs.get("n_z") is not None:
            self.n_z: int = kwargs["n_z"]
        else:
            raise ValueError("`n_z` must be provided!")

        if kwargs.get("x_vector") is not None:
            self.x_vector: np.ndarray = kwargs["x_vector"]
        else:
            raise ValueError("`x_vector` must be provided!")

        if kwargs.get("y_vector") is not None:
            self.y_vector: np.ndarray = kwargs["y_vector"]
        else:
            raise ValueError("`y_vector` must be provided!")

        if kwargs.get("z_vector") is not None:
            self.z_vector: np.ndarray = kwargs["z_vector"]
        else:
            raise ValueError("`z_vector` must be provided!")

        if kwargs.get("volumetric_data") is not None:
            self.volumetric_data: np.ndarray = kwargs["volumetric_data"]
        else:
            raise ValueError("`volumetric_data` must be provided!")

    def from_file(self, filename: str) -> "Cube":
        """
        Read cube data form `filename`.

        Args:
            filename (str): Path to file.

        Returns:
            Cube: Loaded cube.
        """

        return read_cube_file(filename)

    def save(self, filename: str):
        """
        "Save a `Cuve` object inot `filename`.

        Args:
            filename (str): File path.
        """

        save_cube_file(self, filename)

    def copy(self) -> "Cube":
        """
        Create a copy of the `Cube` object.

        Returns:
            Cube: Copied cube.
        """

        return Cube(
            comment1=self.comment1,
            comment2=self.comment2,
            origin=np.copy(self.origin),
            n_atoms=self.n_atoms,
            atoms=[(atom[0], atom[1], np.copy(atom[2])) for atom in self.atoms],
            n_x=self.n_x,
            n_y=self.n_y,
            n_z=self.n_z,
            x_vector=np.copy(self.x_vector),
            y_vector=np.copy(self.y_vector),
            z_vector=np.copy(self.z_vector),
            volumetric_data=np.copy(self.volumetric_data),
        )

    def moved_charge(self) -> float:
        """
        Calculate the total charge moved by the density described by the cube.

        Returns:
            float: Total charge moved.
        """

        return (
            0.5
            * np.sum(np.abs(self.volumetric_data))
            * np.linalg.norm(
                np.dot(np.cross(self.x_vector, self.y_vector), self.z_vector)
            )
        )


def make_cube(
    mol: psi4.core.Molecule,
    matrix: np.ndarray | Sequence[np.ndarray],
    obj_type: str | Iterable[str] = "density",
    **kwargs,
) -> Cube | list[Cube]:
    """
    Create a `Cube` object with volumetric data of the given `matrix`
    calculated on a grid for a psi4 `Molecule` object. If iterables with matrices and types
    are given as `matrix` and `obj_type` returns a list with a Cube for each element of the list.

    Args:
        mol (psi4.core.Molecule): A corresponding molecule.
        matrices (np.ndarray | Iterable[np.ndarray]): Either density matrix,
            orbital coefficients vector, or list of such objects.
        obj_types (str | Iterable[str], optional): Specifies the `matrices` argument type(s),
            must be either "density", "orbital" or a list with such strings. Defaults to "density".

    Returns:
        Cube | list[Cube]: Resulting cube(s).
    """

    if isinstance(matrix, np.ndarray):
        matrix = [matrix]

    if isinstance(obj_type, str):
        obj_type = [obj_type]

    for m_type in obj_type:
        if m_type not in ["density", "orbital"]:
            raise ValueError(
                f"Element of `obj_type` should be either \"density\" or \"orbital\", got {m_type}!"
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
    if isinstance(mol, psi4.core.Molecule):
        if kwargs.get("basisset", None) is not None and isinstance(
            kwargs["basisset"], psi4.core.BasisSet
        ):
            basisset = kwargs["basisset"]
        else:
            basisset = psi4.core.Wavefunction.build(mol).basisset()
        geo_matrix, _, _, elez, _ = mol.to_arrays()

    else:
        raise TypeError(
            f"`mol` should be of type `psi4.core.Molecule` was `{type(mol)}`!"
        )

    basisset.print_out()

    # default grid parameters in bohr
    grid_step = kwargs.get("grid_step", 0.2)
    grid_overage = kwargs.get("grid_overage", 4.0)

    grid = prepare_grid(geo_matrix, grid_step, grid_overage)

    # compute values on the grid on the fly
    values = np.zeros((len(matrix), grid["n_x"], grid["n_y"], grid["n_z"]))

    for i, x in enumerate(grid["x"]):
        for j, y in enumerate(grid["y"]):
            for k, z in enumerate(grid["z"]):
                basis_vals = basisset.compute_phi(x, y, z)

                for m_idx, m, m_type in zip(range(len(matrix)), matrix, obj_type):

                    if m_type == "density":
                        values[m_idx][i][j][k] = basis_vals.dot(m).dot(basis_vals.T)

                    elif m_type == "orbital":
                        values[m_idx][i][j][k] = basis_vals.dot(m)

    # get isocontour values
    iso_sum_level = kwargs.get("iso_sum_level", 0.85)

    cubes_list = []
    for m_idx, m, m_type in zip(range(len(matrix)), matrix, obj_type):

        isovalues = calculate_isocontour(
            values[m_idx], threshold=iso_sum_level, obj_type=m_type
        )

        cube_dict = {
            "comment1": "propSAPT .cube file",
            "comment2": f"isovalues for {iso_sum_level*100:.0f}%"
            f" of the {m_type}: ({isovalues[0]:.6E}, {isovalues[1]:.6E})",
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
            "volumetric_data": values[m_idx],
        }

        cubes_list.append(Cube(**cube_dict))

    # finilize
    psi4.core.print_out("\n")
    psi4.core.print_out(
        f"...finished evaluating values on the grid in {(time() - t_start):5.2f} seconds.\n"
    )
    psi4.core.tstop()
    psi4.core.print_out("\n")
    psi4.core.print_out("*" * 80)
    psi4.core.print_out("\n\n")

    if len(cubes_list) == 1:
        return cubes_list[0]
    else:
        return cubes_list


def read_cube_file(filename: str) -> Cube:
    """
    Reads the data form .cube file `filename`.

    Args:
        filename (str): Path to file.

    Returns:
        Cube: Loaded cube.
    """

    with open(filename, "r", encoding="utf-8") as f:
        # Read the comment lines
        comment1 = f.readline().strip()
        comment2 = f.readline().strip()

        # Read the number of atoms and the origin
        n_atoms, x_origin, y_origin, z_origin = map(float, f.readline().split())
        n_atoms = int(n_atoms)
        origin = np.array([x_origin, y_origin, z_origin])

        # Read the number of voxels and the axis vectors
        x_info = list(map(float, f.readline().split()))
        y_info = list(map(float, f.readline().split()))
        z_info = list(map(float, f.readline().split()))

        n_x = int(x_info[0])
        n_y = int(y_info[0])
        n_z = int(z_info[0])

        x_vector = np.array(x_info[1:])
        y_vector = np.array(y_info[1:])
        z_vector = np.array(z_info[1:])

        # Read the atomic information
        atoms = []
        for _ in range(n_atoms):
            atom_info = list(map(float, f.readline().split()))
            atomic_number = int(atom_info[0])
            charge = atom_info[1]
            position = np.array(atom_info[2:])
            atoms.append((atomic_number, charge, position))

        # Read the volumetric data
        volumetric_data = []
        for _ in range(n_x * n_y * n_z):
            line = f.readline().split()
            volumetric_data.extend(map(float, line))

        volumetric_data = np.array(volumetric_data).reshape((n_x, n_y, n_z))

    return Cube(
        **{
            "comment1": comment1,
            "comment2": comment2,
            "origin": origin,
            "n_atoms": n_atoms,
            "atoms": atoms,
            "n_x": n_x,
            "n_y": n_y,
            "n_z": n_z,
            "x_vector": x_vector,
            "y_vector": y_vector,
            "z_vector": z_vector,
            "volumetric_data": volumetric_data,
        }
    )


def save_cube_file(cube: Cube, filename: str):
    """
    Saves a `Cube` object, into the cube file: `filename`.

    Args:
        cube (Cube): Cube object to be saved.
        filename (str): File path.
    """

    # create and save cube string
    with open(filename, "w", encoding="utf-8") as file:

        # write a header
        file.write(cube.comment1 + "\n")
        file.write(cube.comment2 + "\n")

        # wirte number of atoms and begining of the grid
        file.write(
            f"{cube.n_atoms:6d}  "
            f"{cube.origin[0]: .6f}  "
            f"{cube.origin[1]: .6f}  "
            f"{cube.origin[2]: .6f}\n"
        )

        # write grid details
        file.write(
            f"{cube.n_x:6d}  " + "  ".join([f"{i: .6f}" for i in cube.x_vector]) + "\n"
        )
        file.write(
            f"{cube.n_y:6d}  " + "  ".join([f"{i: .6f}" for i in cube.y_vector]) + "\n"
        )
        file.write(
            f"{cube.n_z:6d}  " + "  ".join([f"{i: .6f}" for i in cube.z_vector]) + "\n"
        )

        # write geometry
        for atom in cube.atoms:
            file.write(
                f"{atom[0]:3d}  "
                f"{atom[1]: .6f}  "
                f"{atom[2][0]: .6f}  "
                f"{atom[2][1]: .6f}  "
                f"{atom[2][2]: .6f}\n"
            )

        # write volumetric information
        count = 0
        for value in cube.volumetric_data.flatten():

            file.write(f"{value: .5E} ")
            count += 1

            if count % 6 == 0:
                file.write("\n")


def subtract_cubes(cube_1: Cube, cube_2: Cube) -> Cube:
    """
    Calculate a difference between volumetric data of two cubes.

    Cube grids have to be the same for this operation.
    Data about molecule geometry is taken from `cube_1`.

    Args:
        cube_1 (Cube): Cube serving as minuend.
        cube_2 (Cube): Cube serving as subtrahend.

    Returns:
        Cube: Resulting Cube.
    """

    if False in np.isclose(cube_1.origin, cube_2.origin):
        raise ValueError(
            "Cube grids have different origins!\n"
            f"cube_1: {cube_1.origin}\n"
            f"cube_2: {cube_2.origin}"
        )

    if (
        False in np.isclose(cube_1.x_vector, cube_2.x_vector)
        or False in np.isclose(cube_1.y_vector, cube_2.y_vector)
        or False in np.isclose(cube_1.z_vector, cube_2.z_vector)
    ):
        raise ValueError(
            "Cube grids have different vectors!\n"
            f"cube_1: {cube_1.x_vector}\n"
            f"        {cube_1.y_vector}\n"
            f"        {cube_1.z_vector}\n"
            f"cube_2: {cube_2.x_vector}\n"
            f"        {cube_2.y_vector}\n"
            f"        {cube_2.z_vector}"
        )

    volumetric_data = cube_1.volumetric_data - cube_2.volumetric_data

    return Cube(
        **{
            "comment1": "",
            "comment2": "",
            "origin": cube_1.origin,
            "n_atoms": cube_1.n_atoms,
            "atoms": cube_1.atoms,
            "n_x": cube_1.n_x,
            "n_y": cube_1.n_y,
            "n_z": cube_1.n_z,
            "x_vector": cube_1.x_vector,
            "y_vector": cube_1.y_vector,
            "z_vector": cube_1.z_vector,
            "volumetric_data": volumetric_data,
        }
    )


def add_cubes(cubes: list[Cube] | tuple[Cube, ...]) -> Cube:
    """
    Calculate a sum of volumetric data of multiple cubes.

    Cube grids have to be the same for this operation.
    Data about molecule geometry is taken from the first cube.

    Args:
        cubes (list[Cube] | tuple[Cube, ...]): List or tuple of cubes to add together.

    Returns:
        Cube: Resulting Cube.
    """

    if not cubes:
        raise ValueError("Cannot add an empty list of cubes!")

    if len(cubes) == 1:
        return cubes[0].copy()

    # Use first cube as reference
    reference_cube = cubes[0]

    # Validate all cubes have the same grid
    for i, cube in enumerate(cubes[1:], start=1):
        if False in np.isclose(reference_cube.origin, cube.origin):
            raise ValueError(
                f"Cube grids have different origins!\n"
                f"cube_0: {reference_cube.origin}\n"
                f"cube_{i}: {cube.origin}"
            )

        if (
            False in np.isclose(reference_cube.x_vector, cube.x_vector)
            or False in np.isclose(reference_cube.y_vector, cube.y_vector)
            or False in np.isclose(reference_cube.z_vector, cube.z_vector)
        ):
            raise ValueError(
                f"Cube grids have different vectors!\n"
                f"cube_0: {reference_cube.x_vector}\n"
                f"        {reference_cube.y_vector}\n"
                f"        {reference_cube.z_vector}\n"
                f"cube_{i}: {cube.x_vector}\n"
                f"        {cube.y_vector}\n"
                f"        {cube.z_vector}"
            )

    # Sum all volumetric data
    volumetric_data = sum(cube.volumetric_data for cube in cubes)

    return Cube(
        **{
            "comment1": "",
            "comment2": "",
            "origin": reference_cube.origin,
            "n_atoms": reference_cube.n_atoms,
            "atoms": reference_cube.atoms,
            "n_x": reference_cube.n_x,
            "n_y": reference_cube.n_y,
            "n_z": reference_cube.n_z,
            "x_vector": reference_cube.x_vector,
            "y_vector": reference_cube.y_vector,
            "z_vector": reference_cube.z_vector,
            "volumetric_data": volumetric_data,
        }
    )


def prepare_grid(
    geometry: np.ndarray, grid_step: float | tuple, grid_overage: float | tuple
) -> dict:
    """
    Prepares a simple scalar grid in 3D space.

    Args:
        geometry (np.ndarray): Molecule geometry. Shloud have shape (N_atom, 3).
        grid_step (float | tuple): Step size of the grid (in bohr). If float then the same step
            is applied in all directions. If tuple, should have length 3, specifies step size
            for x, y and z directions separately.
        grid_overage (float | tuple): Overage of the grid (in bohr). If float then the same overage
            in all directions. If tuple, should have length 3, specifies grid overage
            for x, y and z directions separately.

    Returns:
        dict: Grid info.
    """

    if len(geometry.T) != 3:
        raise ValueError(
            f"`geometry` shloud have shape (N_atom, 3) was {geometry.shape}!"
        )

    # figure out grid sizes
    grid_min = np.zeros(3)
    grid_max = np.zeros(3)

    if isinstance(grid_overage, float):

        for i, axis in enumerate(geometry.T):
            grid_min[i] = np.min(axis) - grid_overage
            grid_max[i] = np.max(axis) + grid_overage

    elif isinstance(grid_overage, tuple):

        if len(grid_overage) != 3:
            raise ValueError(
                "`grid_overage` has to be float or tuple of length 3, "
                f"was length {len(grid_overage)}!"
            )

        for i, axis in enumerate(geometry.T):
            grid_min[i] = np.min(axis) - grid_overage[i]
            grid_max[i] = np.max(axis) + grid_overage[i]

    else:
        raise TypeError(
            f"`grid_overage` has to be float or tuple but was {type(grid_overage)}!"
        )

    # figure out grid points
    grid = {}

    if isinstance(grid_step, float):

        for i, axis in enumerate(["x", "y", "z"]):
            grid[axis] = np.arange(grid_min[i], grid_max[i] + grid_step, grid_step)
            grid[f"step_{axis}"] = grid_step

    elif isinstance(grid_step, tuple):

        if len(grid_step) != 3:
            raise ValueError(
                f"`grid_step` has to be float or tuple of length 3, was length {len(grid_step)}!"
            )

        for i, axis in enumerate(["x", "y", "z"]):
            grid[axis] = np.arange(
                grid_min[i], grid_max[i] + grid_step[i], grid_step[i]
            )
            grid[f"step_{axis}"] = grid_step[i]

    else:
        raise TypeError(
            f"`grid_step' has to be float or tuple but was {type(grid_step)}!"
        )

    grid["n_x"] = grid["x"].size
    grid["n_y"] = grid["y"].size
    grid["n_z"] = grid["z"].size

    return grid


def calculate_isocontour(
    volumetric_data: np.ndarray | Cube,
    threshold: float = 0.85,
    obj_type: str = "density",
) -> tuple[float, float]:
    """
    Calculate isocontour values for a given `threshlod`,
    assumed as the density fraction.

    Args:
        volumetric_data (np.ndarray | Cube): Data for isovale calculations.
        threshold (float, optional): Fraction of the density to be inside of the isosurface
            described by isovalues. Defaults to 0.85.
        obj_type (str, optional): Specifies the type of volumetric data,
            either "density" or "orbital". Defaults to "density".

    Returns:
        tuple[float, float]: Isovalues tuple.
    """

    if obj_type == "density":
        power = 1
    elif obj_type == "orbital":
        power = 2
    else:
        raise ValueError(
            f"`obj_type` should be \"density\" or \"orbital\", was {obj_type}!"
        )

    if isinstance(volumetric_data, Cube):
        values = volumetric_data.volumetric_data
    else:
        values = volumetric_data

    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError(
            f"`threshold` must be within range (0.0, 1.0), was {threshold:.2f}!"
        )

    flatened_vals = values.flatten()
    positive_target = threshold * np.sum(
        0.5 * (np.sign(flatened_vals) + 1) * np.abs(flatened_vals) ** power
    )
    negative_target = threshold * np.sum(
        0.5 * (np.sign(flatened_vals) - 1) * np.abs(flatened_vals) ** power
    )

    positive_sum = 0.0
    negative_sum = 0.0
    positive_isoval = 0.0
    negative_isoval = 0.0
    do_positive = True
    do_negative = True
    for i in np.argsort(np.abs(flatened_vals))[::-1]:

        if do_positive and flatened_vals[i] >= 0.0:
            positive_sum += flatened_vals[i] ** power
            if positive_sum >= positive_target:
                positive_isoval = flatened_vals[i]
                do_positive = False

        elif do_negative and flatened_vals[i] < 0.0:
            negative_sum -= np.abs(flatened_vals[i]) ** power
            if negative_sum <= negative_target:
                negative_isoval = flatened_vals[i]
                do_negative = False

    return (positive_isoval, negative_isoval)  # pyright: ignore reportReturnType
