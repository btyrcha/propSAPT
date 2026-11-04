"""
Cube files related things.

Cube file reference:
https://paulbourke.net/dataformats/cube/
"""

import numbers
import os
from dataclasses import dataclass
from time import time
from collections.abc import Iterable
from typing import Sequence
import numpy as np
import psi4

from .options import PROPSAPT_OPTIONS
from .evaluate import (
    _electronic_esp,
    _evaluate_on_points,
    _get_chunk_size,
    _nuclear_potential,
    _stack_matrices,
)


@dataclass(kw_only=True, eq=False, repr=False)
class Cube:
    """
    Volumetric data on a regular grid, as stored in a `.cube` file.

    All arguments are keyword-only. Coordinates and grid vectors are in bohr.
    Arrays are converted to float NumPy arrays, without a copy when they already are.

    Keyword Args:
        origin (np.ndarray): Origin of the grid, shape (3,).
        atoms (list): List of tuples describing molecule.
            Should look like: (atomic_number, charge, xyz_coordinates).
        x_vector (np.ndarray): Grid step in x direction, shape (3,).
        y_vector (np.ndarray): Grid step in y direction, shape (3,).
        z_vector (np.ndarray): Grid step in z direction, shape (3,).
        volumetric_data (np.ndarray): Values at the grid points, shape (n_x, n_y, n_z).
        comment1 (str, optional): First comment line of `.cube` file. Defaults to "".
        comment2 (str, optional): Second comment line of `.cube` file. Defaults to "".
        n_atoms (int | None, optional): Number of atoms. Defaults to ``len(atoms)``.
        n_x (int | None, optional): Number of grid points in x direction.
            Defaults to ``volumetric_data.shape[0]``.
        n_y (int | None, optional): Number of grid points in y direction.
            Defaults to ``volumetric_data.shape[1]``.
        n_z (int | None, optional): Number of grid points in z direction.
            Defaults to ``volumetric_data.shape[2]``.

    Cubes on the same grid can be added and subtracted, and a cube multiplied or divided
    by a scalar: ``rho - rho_pol``, ``0.5 * rho``, ``-rho``. Grids are checked as in
    `add_cubes`. Results take the grid and atoms of the left cube, share no arrays with
    the operands and have empty comments.

    Raises:
        ValueError: If an array has the wrong shape, or a given count does not match
            `atoms` or `volumetric_data`.
    """

    origin: np.ndarray
    atoms: list[tuple[int, float, np.ndarray]]
    x_vector: np.ndarray
    y_vector: np.ndarray
    z_vector: np.ndarray
    volumetric_data: np.ndarray
    comment1: str = ""
    comment2: str = ""
    n_atoms: int | None = None
    n_x: int | None = None
    n_y: int | None = None
    n_z: int | None = None

    def __post_init__(self):

        self.origin = np.asarray(self.origin, dtype=float)
        self.x_vector = np.asarray(self.x_vector, dtype=float)
        self.y_vector = np.asarray(self.y_vector, dtype=float)
        self.z_vector = np.asarray(self.z_vector, dtype=float)
        self.volumetric_data = np.asarray(self.volumetric_data, dtype=float)
        self.atoms = [
            (int(number), float(charge), np.asarray(xyz, dtype=float))
            for number, charge, xyz in self.atoms
        ]

        # counts not given are read off the data
        if self.n_atoms is None:
            self.n_atoms = len(self.atoms)
        if self.volumetric_data.ndim == 3:
            for axis, n_points in zip("xyz", self.volumetric_data.shape):
                if getattr(self, f"n_{axis}") is None:
                    setattr(self, f"n_{axis}", n_points)

        self._validate()

        # validated equal to the data, so e.g. 4.0 or np.int64(4) become 4
        self.n_atoms = int(self.n_atoms)
        self.n_x, self.n_y, self.n_z = (int(n) for n in (self.n_x, self.n_y, self.n_z))

    def _validate(self):
        """
        Checks the shapes of the arrays and that the counts match them.
        Called at construction and before saving, since fields can be reassigned.

        Raises:
            ValueError: If an array has the wrong shape or a count does not match.
        """

        for name in ("origin", "x_vector", "y_vector", "z_vector"):
            shape = np.shape(getattr(self, name))
            if shape != (3,):
                raise ValueError(f"`{name}` should have shape (3,), was {shape}!")

        for idx, atom in enumerate(self.atoms):
            if len(atom) != 3 or np.shape(atom[2]) != (3,):
                raise ValueError(
                    f"Atom {idx} should be (atomic_number, charge, xyz) with xyz "
                    f"of shape (3,), was {atom}!"
                )

        if self.n_atoms != len(self.atoms):
            raise ValueError(
                f"`n_atoms` is {self.n_atoms} but {len(self.atoms)} atoms are given!"
            )

        data_shape = np.shape(self.volumetric_data)
        if len(data_shape) != 3 or min(data_shape) < 1:
            raise ValueError(
                "`volumetric_data` should have shape (n_x, n_y, n_z) with at least one "
                f"point along each axis, was {data_shape}!"
            )

        counts = (self.n_x, self.n_y, self.n_z)
        if counts != data_shape:
            raise ValueError(
                f"(n_x, n_y, n_z) = {counts} does not match the shape of "
                f"`volumetric_data`, {data_shape}!"
            )

    # NumPy defers to the reflected operators of Cube: `np.float64(2) * cube` works and
    # `array * cube` raises TypeError instead of building an array of cubes
    __array_ufunc__ = None

    def _like(self, volumetric_data: np.ndarray) -> "Cube":
        """
        A new cube on the grid of this one and with its atoms, holding `volumetric_data`.
        It shares no arrays with this cube and has empty comments.
        """

        return Cube(
            origin=np.copy(self.origin),
            atoms=[
                (number, charge, np.copy(xyz)) for number, charge, xyz in self.atoms
            ],
            x_vector=np.copy(self.x_vector),
            y_vector=np.copy(self.y_vector),
            z_vector=np.copy(self.z_vector),
            volumetric_data=volumetric_data,
        )

    def __add__(self, other):

        if not isinstance(other, Cube):
            return NotImplemented

        return add_cubes([self, other])

    def __sub__(self, other):

        if not isinstance(other, Cube):
            return NotImplemented

        return subtract_cubes(self, other)

    def __mul__(self, other):

        if not isinstance(other, numbers.Real):
            return NotImplemented

        return self._like(self.volumetric_data * other)

    __rmul__ = __mul__

    def __truediv__(self, other):

        if not isinstance(other, numbers.Real):
            return NotImplemented

        return self._like(self.volumetric_data / other)

    def __neg__(self):

        return self._like(-self.volumetric_data)

    def __repr__(self) -> str:

        origin = np.array2string(np.asarray(self.origin), precision=4, separator=", ")
        return (
            f"Cube(grid={self.n_x}x{self.n_y}x{self.n_z}, n_atoms={self.n_atoms}, "
            f"origin={origin}, voxel_volume={self.voxel_volume:.6g})"
        )

    @property
    def voxel_volume(self) -> float:
        """
        Volume of one voxel in bohr^3, the triple product of the grid vectors,
        ``|(x_vector x y_vector) . z_vector|``.
        """

        return abs(float(np.dot(np.cross(self.x_vector, self.y_vector), self.z_vector)))

    @classmethod
    def from_file(cls, filename: str | os.PathLike) -> "Cube":
        """
        Read cube data from `filename`.

        Args:
            filename (str | os.PathLike): Path to file.

        Returns:
            Cube: Loaded cube.
        """

        return read_cube_file(filename)

    def save(self, filename: str | os.PathLike):
        """
        Save a `Cube` object into `filename`.

        Args:
            filename (str | os.PathLike): File path.
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

        return 0.5 * np.sum(np.abs(self.volumetric_data)) * self.voxel_volume

    def integrate(
        self, weight: "Cube | np.ndarray | None" = None
    ) -> float | np.ndarray:
        """
        Integrate the volumetric data over the grid: the sum over voxels times
        `voxel_volume`, optionally with a weight at every voxel.

        For a density cube, ``cube.integrate()`` is the number of electrons in the box and
        ``cube.integrate(w_A)`` the electrons that the weight ``w_A`` assigns to fragment A.
        The L2 norm of the data is ``np.sqrt(cube.integrate(cube.volumetric_data))``.
        Integrals on a cube carry the error of its voxel grid; for fragment populations
        from density matrices see `prop_sapt.partition.FragmentOperators`.

        Args:
            weight (Cube | np.ndarray | None, optional): Weight at every voxel, either a
                `Cube` on the same grid or an array whose last three axes are
                (n_x, n_y, n_z). Leading axes of an array stack several weights, which are
                integrated at once. Defaults to None, no weight.

        Raises:
            ValueError: If `weight` is a cube on another grid or an array of another shape.

        Returns:
            float | np.ndarray: The integral, or for stacked weights an array of their
                leading shape.
        """

        data = self.volumetric_data

        if weight is None:
            return float(np.sum(data)) * self.voxel_volume

        if isinstance(weight, Cube):
            _check_same_grid(self, weight, "cube", "weight")
            weight = weight.volumetric_data

        weight = np.asarray(weight, dtype=float)
        if weight.shape[-3:] != data.shape:
            raise ValueError(
                f"`weight` should have shape (..., n_x, n_y, n_z) = (..., "
                f"{', '.join(str(n) for n in data.shape)}), was {weight.shape}!"
            )

        integral = np.tensordot(weight, data, axes=3) * self.voxel_volume

        return float(integral) if integral.ndim == 0 else integral


# cube files store the origin and grid vectors with 6 decimals
_GRID_RTOL = 1e-5
_GRID_ATOL = 1e-6


def _check_same_grid(
    cube_1: Cube, cube_2: Cube, name_1: str = "cube_1", name_2: str = "cube_2"
):
    """
    Checks that two cubes share one grid: the same numbers of points along each axis,
    and the same origin and grid vectors within the precision of a cube file.

    Args:
        cube_1 (Cube): First cube.
        cube_2 (Cube): Second cube.
        name_1 (str, optional): Name of `cube_1` in the error message. Defaults to "cube_1".
        name_2 (str, optional): Name of `cube_2` in the error message. Defaults to "cube_2".

    Raises:
        ValueError: If the grids differ.
    """

    shape_1 = np.shape(cube_1.volumetric_data)
    shape_2 = np.shape(cube_2.volumetric_data)
    if shape_1 != shape_2:
        raise ValueError(
            "Cube grids have different numbers of points!\n"
            f"{name_1}: {shape_1}\n"
            f"{name_2}: {shape_2}"
        )

    if not np.allclose(cube_1.origin, cube_2.origin, rtol=_GRID_RTOL, atol=_GRID_ATOL):
        raise ValueError(
            "Cube grids have different origins!\n"
            f"{name_1}: {cube_1.origin}\n"
            f"{name_2}: {cube_2.origin}"
        )

    vectors_1 = np.array([cube_1.x_vector, cube_1.y_vector, cube_1.z_vector])
    vectors_2 = np.array([cube_2.x_vector, cube_2.y_vector, cube_2.z_vector])
    if not np.allclose(vectors_1, vectors_2, rtol=_GRID_RTOL, atol=_GRID_ATOL):
        indent_1 = " " * (len(name_1) + 2)
        indent_2 = " " * (len(name_2) + 2)
        raise ValueError(
            "Cube grids have different vectors!\n"
            f"{name_1}: {vectors_1[0]}\n"
            f"{indent_1}{vectors_1[1]}\n"
            f"{indent_1}{vectors_1[2]}\n"
            f"{name_2}: {vectors_2[0]}\n"
            f"{indent_2}{vectors_2[1]}\n"
            f"{indent_2}{vectors_2[2]}"
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
        matrix (np.ndarray | Sequence[np.ndarray]): Either density matrix,
            orbital coefficients vector (shape (nbf,) or one column (nbf, 1) of Psi4's `Ca()`),
            or list of such objects.
        obj_type (str | Iterable[str], optional): Specifies the `matrix` argument type(s),
            must be either "density", "orbital", "esp" or a list with such strings. Defaults to "density".
            For "esp" the `matrix` is a density matrix and the cube holds the electrostatic
            potential it generates, from analytic potential integrals (electrons negative).
        **kwargs: `basisset` (psi4.core.BasisSet) is the basis set the matrices are in.
            Without it, one is built for `mol` from the current `BASIS` option; pass it
            for matrices of a `Dimer` (or use `Dimer.make_cube`, which does).
            `grid` (dict) is a grid from `prepare_grid`, reused as given; otherwise one is
            built around `mol` with `grid_step` and `grid_overage` (bohr), which default to
            the `cube_grid_step` and `cube_grid_overage` options.
            `esp_nuclei` (bool) adds the potential of the nuclear charges of `mol` to
            every "esp" cube, use it for total (not interaction-induced) densities; a nucleus
            lying exactly on a grid point is skipped at that point. Defaults to False.
            `chunk_size` (int) is the number of grid points for which basis function values
            are computed at once. Defaults to the `point_eval_chunk_size` option.
            `iso_sum_level` (float) is the fraction of the density (or squared orbital)
            enclosed by the isovalues written into the cube comments. Defaults to the
            `cube_iso_sum_level` option.
            Values at the grid points come from `prop_sapt.evaluate`.

    Returns:
        Cube | list[Cube]: Resulting cube(s).
    """

    if isinstance(matrix, np.ndarray):
        matrix = [matrix]
    else:
        matrix = list(matrix)

    if len(matrix) == 0:
        raise ValueError(
            "No matrices given for cube generation!\n"
            "`matrix` must be either np.ndarray or Iterable of nonzero length."
        )

    if isinstance(obj_type, str):
        obj_type = [obj_type]
    else:
        obj_type = list(obj_type)

    if len(obj_type) == 1 and len(matrix) > 1:
        obj_type = obj_type * len(matrix)
    elif len(obj_type) != len(matrix):
        raise ValueError(
            "`obj_type` and `matrix` must have the same length, or `obj_type` must have length 1. "
            f"Got len(obj_type)={len(obj_type)} and len(matrix)={len(matrix)}."
        )

    for m_type in obj_type:
        if m_type not in ["density", "orbital", "esp"]:
            raise ValueError(
                f'Element of `obj_type` should be either "density", "orbital" or "esp", got {m_type}!'
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
    t_setup = time()
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

    if kwargs.get("grid", None) is not None and isinstance(kwargs["grid"], dict):
        grid = kwargs["grid"]
    else:
        # default grid parameters in bohr
        grid_step = kwargs.get("grid_step", PROPSAPT_OPTIONS["cube_grid_step"])
        grid_overage = kwargs.get("grid_overage", PROPSAPT_OPTIONS["cube_grid_overage"])

        grid = prepare_grid(geo_matrix, grid_step, grid_overage)

    n_points = grid["n_x"] * grid["n_y"] * grid["n_z"]
    psi4.core.print_out(
        f"  Grid size      : {grid['n_x']} x {grid['n_y']} x {grid['n_z']}"
        f"  ({n_points} points)\n"
    )
    psi4.core.print_out(
        f"  Grid step      : {grid['step_x']:.4f} x {grid['step_y']:.4f}"
        f" x {grid['step_z']:.4f} bohr\n"
    )
    n_density = sum(1 for t in obj_type if t == "density")
    n_orbital = sum(1 for t in obj_type if t == "orbital")
    n_esp = sum(1 for t in obj_type if t == "esp")
    psi4.core.print_out(
        f"  Matrices       : {len(matrix)}"
        f"  ({n_density} density, {n_orbital} orbital, {n_esp} esp)\n"
    )
    esp_nuclei = kwargs.get("esp_nuclei", False)
    if n_esp > 0:
        psi4.core.print_out(f"  ESP nuclei     : {'yes' if esp_nuclei else 'no'}\n")
    psi4.core.print_out(f"  Setup time     : {(time() - t_setup):5.2f} s\n")
    psi4.core.print_out("\n")

    # compute values on the grid one x-slice at a time
    values = np.zeros((len(matrix), grid["n_x"], grid["n_y"], grid["n_z"]))
    slice_shape = (-1, grid["n_y"], grid["n_z"])

    density_indices = [
        idx for idx, m_type in enumerate(obj_type) if m_type == "density"
    ]
    orbital_indices = [
        idx for idx, m_type in enumerate(obj_type) if m_type == "orbital"
    ]
    esp_indices = [idx for idx, m_type in enumerate(obj_type) if m_type == "esp"]

    # integer index arrays for advanced-index assignment into `values`
    density_idx_arr = np.array(density_indices, dtype=int)
    orbital_idx_arr = np.array(orbital_indices, dtype=int)
    esp_idx_arr = np.array(esp_indices, dtype=int)

    nbf = basisset.nbf()
    density_stack = (
        _stack_matrices([matrix[idx] for idx in density_indices], nbf)[0]
        if density_indices
        else None
    )
    orbital_columns = (
        np.column_stack([_orbital_vector(matrix[idx], nbf) for idx in orbital_indices])
        if orbital_indices
        else None
    )
    esp_stack = (
        _stack_matrices([matrix[idx] for idx in esp_indices], nbf)[0]
        if esp_indices
        else None
    )
    if esp_indices:
        mints = psi4.core.MintsHelper(basisset)

    chunk_size = _get_chunk_size(kwargs.get("chunk_size", None))
    yz_points = np.stack(
        np.meshgrid(grid["y"], grid["z"], indexing="ij"), axis=-1
    ).reshape(-1, 2)

    t_grid = time()
    timings = {"phi": 0.0, "contract": 0.0}
    t_esp = 0.0

    for i, x in enumerate(grid["x"]):
        points = np.column_stack([np.full(len(yz_points), x), yz_points])

        # ESP from analytic potential integrals
        if esp_indices:
            _t0 = time()
            esp = _electronic_esp(points, mints, esp_stack)
            if esp_nuclei:
                esp += _nuclear_potential(points, mol)
            values[esp_idx_arr, i] = esp.reshape(slice_shape)
            t_esp += time() - _t0

        # ESP-only grids need no basis function values
        if density_stack is not None or orbital_columns is not None:
            densities, orbitals = _evaluate_on_points(
                points,
                basisset,
                chunk_size,
                density_stack=density_stack,
                orbital_columns=orbital_columns,
                timings=timings,
            )
            if densities is not None:
                values[density_idx_arr, i] = densities.reshape(slice_shape)
            if orbitals is not None:
                values[orbital_idx_arr, i] = orbitals.reshape(slice_shape)

        # progress report every x-slice
        psi4.core.print_out(
            f"\r  Grid progress  : x-slice {i + 1:4d} / {grid['n_x']:4d}"
            f"  (phi: {timings['phi']:6.2f} s, contract: {timings['contract']:6.2f} s,"
            f" esp: {t_esp:6.2f} s, elapsed: {(time() - t_grid):6.2f} s)"
        )

    t_grid_total = time() - t_grid
    t_phi = timings["phi"]
    t_contract = timings["contract"]
    t_overhead = t_grid_total - t_phi - t_contract - t_esp
    psi4.core.print_out(
        f"\n\n  Grid evaluation : {t_grid_total:5.2f} s total\n"
        f"    basis values  : {t_phi:5.2f} s  ({100 * t_phi / t_grid_total:.1f}%)\n"
        f"    contractions  : {t_contract:5.2f} s  ({100 * t_contract / t_grid_total:.1f}%)\n"
        f"    esp integrals : {t_esp:5.2f} s  ({100 * t_esp / t_grid_total:.1f}%)\n"
        f"    overhead      : {t_overhead:5.2f} s"
        f"  ({100 * t_overhead / t_grid_total:.1f}%)\n"
    )

    # get isocontour values
    iso_sum_level = kwargs.get("iso_sum_level", PROPSAPT_OPTIONS["cube_iso_sum_level"])

    cubes_list = []
    for m_idx, m_type in zip(range(len(matrix)), obj_type):

        if m_type == "esp":
            comment2 = "electrostatic potential (ESP) Cube, " + (
                "electrons and nuclei" if esp_nuclei else "electrons only"
            )
        else:
            isovalues = calculate_isocontour(
                values[m_idx], threshold=iso_sum_level, obj_type=m_type
            )
            comment2 = (
                f"isovalues for {iso_sum_level*100:.0f}%"
                f" of the {m_type}: ({isovalues[0]:.6E}, {isovalues[1]:.6E})"
            )

        cube_dict = {
            "comment1": "propSAPT .cube file",
            "comment2": comment2,
            "origin": [grid["x"][0], grid["y"][0], grid["z"][0]],
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

    # finalize
    psi4.core.print_out("\n")
    psi4.core.print_out(f"...finished in {(time() - t_start):5.2f} s total.\n")
    psi4.core.tstop()
    psi4.core.print_out("\n")
    psi4.core.print_out("*" * 80)
    psi4.core.print_out("\n\n")

    if len(cubes_list) == 1:
        return cubes_list[0]
    else:
        return cubes_list


def _orbital_vector(coeffs: np.ndarray, nbf: int) -> np.ndarray:
    """
    Returns orbital coefficients given as a vector (nbf,) or a column (nbf, 1) as a vector.

    Raises:
        ValueError: If `coeffs` holds other than one orbital of `nbf` coefficients.
    """

    vector = np.asarray(coeffs, dtype=float)
    if vector.shape not in ((nbf,), (nbf, 1)):
        raise ValueError(
            f"An orbital should have shape ({nbf},) or ({nbf}, 1) of the basis set, "
            f"was {vector.shape}! Pass several orbitals as a list, one cube each."
        )

    return vector.reshape(nbf)


def read_cube_file(filename: str | os.PathLike) -> Cube:
    """
    Reads the data from .cube file `filename`.

    The volumetric data may be laid out with any number of values per line. Cubes of
    several orbitals (a negative number of atoms), with several values per voxel, or in
    angstrom (negative numbers of points) are not supported.

    Args:
        filename (str | os.PathLike): Path to file.

    Raises:
        NotImplementedError: For the cube variants that are not supported.
        ValueError: If the file holds other than n_x * n_y * n_z values.

    Returns:
        Cube: Loaded cube.
    """

    with open(filename, "r", encoding="utf-8") as f:
        # Read the comment lines
        comment1 = f.readline().strip()
        comment2 = f.readline().strip()

        # Read the number of atoms and the origin, maybe followed by values per voxel
        fields = f.readline().split()
        n_atoms = int(float(fields[0]))
        origin = np.array(fields[1:4], dtype=float)

        if n_atoms < 0:
            raise NotImplementedError(
                f"{filename}: cubes of several orbitals (negative number of atoms) "
                "are not supported!"
            )
        if len(fields) > 4 and int(float(fields[4])) != 1:
            raise NotImplementedError(
                f"{filename}: cubes with {fields[4]} values per voxel are not supported!"
            )

        # Read the number of voxels and the axis vectors
        counts = []
        vectors = []
        for _ in range(3):
            fields = f.readline().split()
            counts.append(int(float(fields[0])))
            vectors.append(np.array(fields[1:4], dtype=float))

        if min(counts) < 0:
            raise NotImplementedError(
                f"{filename}: cubes in angstrom (negative numbers of points) "
                "are not supported!"
            )

        n_x, n_y, n_z = counts
        x_vector, y_vector, z_vector = vectors

        # Read the atomic information
        atoms = []
        for _ in range(n_atoms):
            fields = f.readline().split()
            atoms.append(
                (
                    int(float(fields[0])),
                    float(fields[1]),
                    np.array(fields[2:5], dtype=float),
                )
            )

        # Read the volumetric data, all at once
        volumetric_data = np.fromstring(f.read(), sep=" ")

    n_points = n_x * n_y * n_z
    if volumetric_data.size != n_points:
        raise ValueError(
            f"{filename}: expected n_x * n_y * n_z = {n_x} * {n_y} * {n_z} = {n_points} "
            f"values, found {volumetric_data.size}!"
        )

    return Cube(
        comment1=comment1,
        comment2=comment2,
        origin=origin,
        n_atoms=n_atoms,
        atoms=atoms,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        x_vector=x_vector,
        y_vector=y_vector,
        z_vector=z_vector,
        volumetric_data=volumetric_data.reshape((n_x, n_y, n_z)),
    )


def save_cube_file(cube: Cube, filename: str | os.PathLike):
    """
    Saves a `Cube` object into the cube file: `filename`.

    Args:
        cube (Cube): Cube object to be saved.
        filename (str | os.PathLike): File path.

    Raises:
        ValueError: If the counts of `cube` do not match its data, e.g. after
            `volumetric_data` was replaced by an array of another shape.
    """

    cube._validate()

    # create and save cube string
    with open(filename, "w", encoding="utf-8") as file:

        # write a header
        file.write(cube.comment1 + "\n")
        file.write(cube.comment2 + "\n")

        # write number of atoms and beginning of the grid
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

        # write volumetric information, 6 values per line, no newline after the last
        flat = np.ravel(cube.volumetric_data)
        n_head = (flat.size - 1) // 6 * 6  # values before the last line
        if n_head > 0:
            np.savetxt(
                file,
                flat[:n_head].reshape(-1, 6),
                fmt="% .5E",
                delimiter=" ",
                newline=" \n",
            )
        file.write("".join(f"{v: .5E} " for v in flat[n_head:]))


def subtract_cubes(cube_1: Cube, cube_2: Cube) -> Cube:
    """
    Calculate a difference between volumetric data of two cubes, ``cube_1 - cube_2``.

    Cube grids have to be the same for this operation.
    Data about molecule geometry is taken from `cube_1`; the result shares no arrays
    with the operands and has empty comments.

    Args:
        cube_1 (Cube): Cube serving as minuend.
        cube_2 (Cube): Cube serving as subtrahend.

    Raises:
        ValueError: If the cube grids differ.

    Returns:
        Cube: Resulting Cube.
    """

    _check_same_grid(cube_1, cube_2, "cube_1", "cube_2")

    return cube_1._like(cube_1.volumetric_data - cube_2.volumetric_data)


def add_cubes(cubes: list[Cube] | tuple[Cube, ...]) -> Cube:
    """
    Calculate a sum of volumetric data of multiple cubes, ``cube_1 + cube_2 + ...``.

    Cube grids have to be the same for this operation.
    Data about molecule geometry is taken from the first cube; the result shares no
    arrays with the operands and has empty comments (a single cube is copied as it is).

    Args:
        cubes (list[Cube] | tuple[Cube, ...]): List or tuple of cubes to add together.

    Raises:
        ValueError: If no cubes are given, or the cube grids differ.

    Returns:
        Cube: Resulting Cube.
    """

    if not cubes:
        raise ValueError("Cannot add an empty list of cubes!")

    if len(cubes) == 1:
        return cubes[0].copy()

    # Use first cube as reference
    reference_cube = cubes[0]

    for i, cube in enumerate(cubes[1:], start=1):
        _check_same_grid(reference_cube, cube, "cube_0", f"cube_{i}")

    return reference_cube._like(sum(cube.volumetric_data for cube in cubes))


def translate_cube(cube: Cube, vector: np.ndarray, inplace: bool = False) -> Cube:
    """Translates a given Cube in 3D space by a given vector.

    Args:
        cube (Cube): A Cube object to be translated.
        vector (np.ndarray): An array of length 3, a translation vector.
        inplace (bool): Make an inplace modification if True, else returns a copy. Defaults to False.


    Returns:
        Cube: Resulting Cube object.
    """

    if not inplace:
        cube = cube.copy()

    cube.comment1 += f" Translated by vector {vector}"
    cube.origin = cube.origin + vector
    cube.atoms = [
        (at_num, charge, coord + vector) for at_num, charge, coord in cube.atoms
    ]

    return cube


def cosine_similarity(cube_1: Cube, cube_2: Cube) -> float:
    """Calculate cosine similarity between two cubes on the same grid.

    The cosine similarity is calculated as:

        cosine_similarity = (A . B) / (||A|| ||B||)

    utilizing the L2 norm.

    Args:
        cube_1 (Cube): First cube.
        cube_2 (Cube): Second cube.

    Raises:
        ValueError: If the cube grids differ.

    Returns:
        float: Cosine similarity between the two cubes.
    """

    _check_same_grid(cube_1, cube_2, "cube_1", "cube_2")

    flat_1 = cube_1.volumetric_data.flatten()
    flat_2 = cube_2.volumetric_data.flatten()

    return np.sum(flat_1 * flat_2) / (np.linalg.norm(flat_1) * np.linalg.norm(flat_2))


def prepare_grid(
    geometry: np.ndarray, grid_step: float | tuple, grid_overage: float | tuple
) -> dict:
    """
    Prepares a simple scalar grid in 3D space.

    Args:
        geometry (np.ndarray): Molecule geometry. Should have shape (N_atom, 3).
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
            f"`geometry` should have shape (N_atom, 3), was {geometry.shape}!"
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
            f"`grid_step` has to be float or tuple but was {type(grid_step)}!"
        )

    grid["n_x"] = grid["x"].size
    grid["n_y"] = grid["y"].size
    grid["n_z"] = grid["z"].size

    return grid


def calculate_isocontour(
    volumetric_data: np.ndarray | Cube,
    threshold: float | None = None,
    obj_type: str = "density",
) -> tuple[float, float]:
    """
    Calculate isocontour values for a given `threshold`,
    assumed as the density fraction.

    Args:
        volumetric_data (np.ndarray | Cube): Data for isovalue calculations.
        threshold (float | None, optional): Fraction of the density to be inside of the isosurface
            described by isovalues. Defaults to the `cube_iso_sum_level` option.
        obj_type (str, optional): Specifies the type of volumetric data,
            either "density" or "orbital". Defaults to "density".

    Returns:
        tuple[float, float]: Isovalues tuple.
    """

    # grab default threshold value if not given
    if threshold is None:
        threshold = PROPSAPT_OPTIONS["cube_iso_sum_level"]

    if obj_type == "density":
        power = 1
    elif obj_type == "orbital":
        power = 2
    else:
        raise ValueError(
            f'`obj_type` should be "density" or "orbital", was {obj_type}!'
        )

    if isinstance(volumetric_data, Cube):
        values = volumetric_data.volumetric_data
    else:
        values = volumetric_data

    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError(
            f"`threshold` must be within range (0.0, 1.0), was {threshold:.2f}!"
        )

    t_iso = time()

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

    psi4.core.print_out(f"  Isocontour     : {(time() - t_iso):5.2f} s\n")

    return (positive_isoval, negative_isoval)  # pyright: ignore reportReturnType
