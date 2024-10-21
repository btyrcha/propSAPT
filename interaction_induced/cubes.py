"""
Cube files related things.
"""

import numpy as np


class Cube:
    """
    Class for storing cube files.
    """

    def __init__(self, **kwargs):

        self.comment1 = kwargs.get("comment1", None)
        self.comment2 = kwargs.get("comment2", None)
        self.origin = kwargs.get("origin", None)
        self.n_atoms = kwargs.get("n_atoms", None)
        self.atoms = kwargs.get("atoms", None)
        self.n_x = kwargs.get("n_x", None)
        self.n_y = kwargs.get("n_y", None)
        self.n_z = kwargs.get("n_z", None)
        self.x_vector = kwargs.get("x_vector", None)
        self.y_vector = kwargs.get("y_vector", None)
        self.z_vector = kwargs.get("z_vector", None)
        self.volumetric_data = kwargs.get("volumetric_data", None)

    def from_file(self, filename: str):
        """
        Read cube data form `filename`.
        """

        return read_cube_file(filename)

    def save(self, filename: str):
        "Save a `Cuve` object inot `filename`."

        save_cube_file(self, filename)


def read_cube_file(filename: str):
    """
    Reads the data form .cube file `filename`.

    Returns `Cube` object.
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


def prepare_grid(
    geometry: np.ndarray, grid_step: float | tuple, grid_overage: float | tuple
) -> dict:
    """
    Prepares a simple scalar grid in 3D space.
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
):
    """
    Calculate isocontour values for a given `threshlod`,
    assumed as the density fraction.
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

    return (positive_isoval, negative_isoval)
