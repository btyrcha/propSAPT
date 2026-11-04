"""
Definition of a core Dimer (molecule-like) class.
"""

import os
import warnings
from time import time
from collections.abc import Callable, Iterable, Mapping
from typing import Sequence
import numpy as np
import psi4
from .sinfinity import sinfinity
from .cubes import Cube, make_cube
from .fragments import DimerFragments
from .evaluate import evaluate_density, evaluate_esp, evaluate_orbital
from .partition import (
    FragmentOperators,
    QuadratureGrid,
    becke_fragment_weights,
    hirshfeld_fragment_weights,
    molecular_grid,
)
from .utils import changed_psi4_options


class Dimer(sinfinity):
    """
    Core dimer (molecule-like) class.

    Initialises and stores all necessary values
    for 'propSAPT' calculations.
    Performs SCF calculations for the monomers.

    The fragment structure of the geometry (atoms, charges and multiplicities of
    monomers A and B and of any extra fragments) is in `fragments`, a
    `prop_sapt.fragments.DimerFragments`.

    The settings the dimer is built with are kept: `psi4_options`, the Psi4 options
    changed from their defaults, and `fitting_bases`, the fitting basis sets as built
    (``basis.blend()``) by option, "DF_BASIS_SCF" and "DF_BASIS_SAPT". Changing the
    options afterwards does not change the dimer.

    Memory comes from Psi4's setting (`psi4.set_memory`), as for any Psi4 calculation.

    Args:
        geometry (str): Dimer geometry in Psi4 format, with monomers A and B as its
            first two fragments; any further fragments are ghost atoms.
        reference (str, optional): "RHF" or "RKS". Defaults to "RHF".
        memory (float | None, optional): Deprecated, set `psi4.set_memory` instead.
            Memory of the JK object in GB; None, the default, follows Psi4's memory.

    Keyword Args:
        functional (str): Functional of an RKS reference. Defaults to the
            `default_functional` option.
        grac_A (float): GRAC shift of monomer A (Psi4's `dft_grac_shift`). Defaults to
            no shift.
        grac_B (float): GRAC shift of monomer B. Defaults to no shift.
    """

    def __init__(self, geometry, reference="RHF", memory=None, **kwargs):

        if memory is not None:
            message = (
                "`Dimer(memory=...)` is deprecated: set the memory with `psi4.set_memory`, "
                "which the dimer follows by default."
            )
            warnings.warn(message, FutureWarning, stacklevel=2)
            psi4.core.print_out(f"\nWarning: {message}\n")

        # starting initialisation
        t_start = time()
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\nInitializing Dimer object...\n\n")
        psi4.core.tstart()

        # create Psi4 Molecule object
        self.geometry = geometry
        self.dimer = psi4.geometry(geometry)
        self.dimer.print_out_in_angstrom()

        # the Psi4 options the dimer is built with, before the monomer SCFs change any
        self.psi4_options = changed_psi4_options()

        # initialize parent class
        super().__init__(self.dimer, reference, memory, **kwargs)

        self.fragments = DimerFragments.from_molecule(self.dimer)

        self.cpscf_ra = None
        self.cpscf_sb = None

        # theta amplitudes of the exchange density matrices, see `get_theta_amplitudes`
        self.theta_amplitudes: dict[str, np.ndarray | tuple[np.ndarray, ...]] = {}

        # print time
        psi4.core.print_out(
            f"...finished initializing Dimer object in {(time() - t_start):5.2f} seconds.\n"
        )
        psi4.core.tstop()
        psi4.core.print_out("\n")
        psi4.core.print_out("*" * 80)
        psi4.core.print_out("\n")

    def get_cpscf_ra(self) -> np.ndarray:
        """
        Grab CPSCF response amplitudes for monomer A or calculate them if they haven't been
        calculated already.

        Returns:
            np.ndarray: Response amplitudes for monomer A.
        """

        if self.cpscf_ra is None:
            self.cpscf_ra = self.cpscf("A")

        return self.cpscf_ra

    def get_cpscf_sb(self) -> np.ndarray:
        """
        Grab CPSCF response amplitudes for monomer B or calculate them if they haven't been
        calculated already.

        Returns:
            np.ndarray: Response amplitudes for monomer B.
        """

        if self.cpscf_sb is None:
            self.cpscf_sb = self.cpscf("B")

        return self.cpscf_sb

    def get_theta_amplitudes(
        self,
        key: str,
        build: Callable[["Dimer"], np.ndarray | tuple[np.ndarray, ...]],
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Grab the theta amplitudes `key` of an exchange density matrix, or calculate them
        with `build` if they haven't been calculated already.

        The density matrix of each monomer needs the amplitudes of both, so they are
        kept for the other monomer's `calc_density_matrix` call. They are read-only.
        Free them with `clear_theta_amplitudes`.

        Args:
            key (str): Density component, e.g. "exch-ind_S2".
            build (Callable[[Dimer], np.ndarray | tuple[np.ndarray, ...]]): Calculates
                the amplitudes for this dimer.

        Returns:
            np.ndarray | tuple[np.ndarray, ...]: The amplitudes, as returned by `build`.
        """

        if key not in self.theta_amplitudes:
            self.store_theta_amplitudes(key, build(self))

        return self.theta_amplitudes[key]

    def store_theta_amplitudes(
        self, key: str, amplitudes: np.ndarray | tuple[np.ndarray, ...]
    ) -> None:
        """
        Keep the theta amplitudes `key`, made read-only, for `get_theta_amplitudes`.

        Args:
            key (str): Density component, e.g. "exch-ind_S2".
            amplitudes (np.ndarray | tuple[np.ndarray, ...]): The amplitudes.
        """

        arrays = amplitudes if isinstance(amplitudes, tuple) else (amplitudes,)
        for array in arrays:
            array.flags.writeable = False
        self.theta_amplitudes[key] = amplitudes

    def clear_theta_amplitudes(self):
        """
        Free the theta amplitudes of the exchange density matrices, see
        `get_theta_amplitudes`. The exchange-dispersion ones are as large as the
        dispersion amplitudes. `calc_densities` frees them when it is done; call this
        after calling `calc_density_matrix` for both monomers yourself.
        """

        self.theta_amplitudes.clear()

    def get_psi4_molecule(self) -> psi4.core.Molecule:
        """
        Get 'psi4.core.Molecule' object of the dimer.

        Returns:
            psi4.core.Molecule: Psi4 molecule object of the dimer.
        """

        return self.dimer

    def get_psi4_wavefunction(self) -> psi4.core.Wavefunction:
        """
        Get 'psi4.core.Wavefunction' object of the dimer, in the basis set the dimer was
        built with, whatever the `BASIS` option is now.

        Returns:
            psi4.core.Wavefunction: Psi4 wavefunction object of the dimer.
        """

        return psi4.core.Wavefunction.build(
            self.get_psi4_molecule(), self.get_psi4_basisset()
        )

    def get_psi4_basisset(self) -> psi4.core.BasisSet:
        """
        Get 'psi4.core.BasisSet' object of the dimer: the one it was built with, whatever
        the `BASIS` option is now.

        Returns:
            psi4.core.BasisSet: Psi4 basis set object of the dimer.
        """

        return self.dimer_wfn.basisset()

    def make_cube(
        self,
        matrix: np.ndarray | Sequence[np.ndarray],
        obj_type: str | Iterable[str] = "density",
        **kwargs,
    ) -> Cube | list[Cube]:
        """
        Create a `Cube` object with volumetric data of the given `matrix` calculated on a grid.

        Args:
            matrix (np.ndarray | Sequence[np.ndarray]): Either density matrix,
                orbital coefficients vector, or list of such objects.
            obj_type (str | Iterable[str], optional): Specifies the `matrix` argument type(s),
                either "density", "orbital", "esp" or a list with such strings.
                Defaults to "density".
            **kwargs: Passed to `prop_sapt.cubes.make_cube`, e.g. `grid_step` or `esp_nuclei`.
                `basisset` defaults to the dimer's basis set, see `get_psi4_basisset`.

        Returns:
            Cube | list[Cube]: Resulting cube(s).
        """

        if kwargs.get("basisset") is None:
            kwargs["basisset"] = self.get_psi4_basisset()

        return make_cube(
            self.get_psi4_molecule(), matrix=matrix, obj_type=obj_type, **kwargs
        )

    def evaluate_density(
        self,
        dm: np.ndarray | Sequence[np.ndarray],
        points: np.ndarray,
        basisset: psi4.core.BasisSet | None = None,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """
        Evaluate the density of AO density matrix `dm` at `points`.
        No spin factor is applied: for the total density of an RHF monomer pass `2 * Da`.

        Args:
            dm (np.ndarray | Sequence[np.ndarray]): AO density matrix, shape (nbf, nbf),
                or several of them, as a sequence or an array of shape (M, nbf, nbf).
            points (np.ndarray): Points in bohr, shape (..., 3).
            basisset (psi4.core.BasisSet | None, optional): Basis set of `dm`.
                Defaults to the dimer basis set.
            chunk_size (int | None, optional): Number of points evaluated at once.
                Defaults to the `point_eval_chunk_size` option.

        Returns:
            np.ndarray: Density at the points, shape (...) for a single matrix
                and (M, ...) for several.
        """

        if basisset is None:
            basisset = self.get_psi4_basisset()

        return evaluate_density(dm, points, basisset, chunk_size=chunk_size)

    def evaluate_orbital(
        self,
        coeffs: np.ndarray,
        points: np.ndarray,
        basisset: psi4.core.BasisSet | None = None,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """
        Evaluate orbitals given by MO coefficients `coeffs` at `points`.
        Orbitals are the columns of `coeffs`, as in Psi4's `Ca()`.

        Args:
            coeffs (np.ndarray): MO coefficients, shape (nbf,) for one orbital
                or (nbf, k) for k orbitals.
            points (np.ndarray): Points in bohr, shape (..., 3).
            basisset (psi4.core.BasisSet | None, optional): Basis set of `coeffs`.
                Defaults to the dimer basis set.
            chunk_size (int | None, optional): Number of points evaluated at once.
                Defaults to the `point_eval_chunk_size` option.

        Returns:
            np.ndarray: Orbital values at the points, shape (...) for `coeffs` of shape
                (nbf,) and (k, ...) for `coeffs` of shape (nbf, k).
        """

        if basisset is None:
            basisset = self.get_psi4_basisset()

        return evaluate_orbital(coeffs, points, basisset, chunk_size=chunk_size)

    def evaluate_esp(
        self,
        dm: np.ndarray | Sequence[np.ndarray],
        points: np.ndarray,
        nuclei: bool = False,
        basisset: psi4.core.BasisSet | None = None,
    ) -> np.ndarray:
        """
        Evaluate the electrostatic potential of AO density matrix `dm` at `points`,
        from analytic potential integrals. Electrons contribute negatively.

        Args:
            dm (np.ndarray | Sequence[np.ndarray]): AO density matrix, shape (nbf, nbf),
                or several of them, as a sequence or an array of shape (M, nbf, nbf).
            points (np.ndarray): Points in bohr, shape (..., 3).
            nuclei (bool, optional): Add the potential of the nuclei of the basis set's
                molecule, use it for total (not interaction-induced) densities.
                Defaults to False.
            basisset (psi4.core.BasisSet | None, optional): Basis set of `dm`.
                Defaults to the dimer basis set.

        Returns:
            np.ndarray: Electrostatic potential at the points, shape (...) for a single
                matrix and (M, ...) for several.
        """

        if basisset is None:
            basisset = self.get_psi4_basisset()

        return evaluate_esp(dm, points, basisset, nuclei=nuclei)

    def fragment_operators(
        self,
        weights: str | Mapping[str, np.ndarray] = "becke",
        grid: QuadratureGrid | None = None,
        dipole: bool = True,
        chunk_size: int | None = None,
        partition_of_unity: bool = True,
        **grid_kwargs,
    ) -> FragmentOperators:
        """
        Compute the fragment population operators of monomers A and B in the dimer
        basis set, see `prop_sapt.partition.FragmentOperators`.

        Args:
            weights (str | Mapping[str, np.ndarray], optional): Fragment weights:
                "becke" for the Becke cells of each monomer's atoms, "hirshfeld" for
                rho_X / (rho_A + rho_B) from the unperturbed monomer densities, or the
                weight of each fragment at the points of `grid`. Defaults to "becke".
            grid (QuadratureGrid | None, optional): Quadrature grid, required for
                weights given as arrays. Defaults to `molecular_grid` of the dimer.
            dipole (bool, optional): Also compute the first-moment matrices, needed for
                dipoles. Defaults to True.
            chunk_size (int | None, optional): Number of points evaluated at once.
                Defaults to the `point_eval_chunk_size` option.
            partition_of_unity (bool, optional): Whether the fragments of `weights`
                together cover the whole system, i.e. their weights sum to 1 at every
                point. If True, this is checked, and every population is checked to add
                up to the electrons of the density, Tr(D S), see
                `FragmentOperators.from_grid`. Pass False for only some of the
                fragments. Defaults to True.
            **grid_kwargs: Passed to `prop_sapt.partition.molecular_grid` when no `grid`
                is given, e.g. `radial_points` or `spherical_points`.

        Returns:
            FragmentOperators: Operators of fragments "A" and "B" for named weights,
                or of the fragments of `weights`.
        """

        if grid is not None and grid_kwargs:
            raise TypeError(
                f"Grid options {sorted(grid_kwargs)} cannot be used with a given `grid`!"
            )

        basisset = self.get_psi4_basisset()

        if isinstance(weights, str):
            if grid is None:
                grid = molecular_grid(basisset, **grid_kwargs)

            if weights.lower() == "becke":
                fragment_weights = becke_fragment_weights(
                    grid, {"A": self.fragments.A.atoms, "B": self.fragments.B.atoms}
                )
            elif weights.lower() == "hirshfeld":
                fragment_weights = hirshfeld_fragment_weights(
                    grid.points,
                    {
                        "A": 2 * np.asarray(self.wfnA.Da()),  # RHF spin
                        "B": 2 * np.asarray(self.wfnB.Da()),
                    },
                    basisset,
                    chunk_size=chunk_size,
                )
            else:
                raise ValueError(
                    f'`weights` should be "becke", "hirshfeld" or a mapping of arrays, '
                    f'was "{weights}"!'
                )
        else:
            if grid is None:
                raise ValueError(
                    "Fragment weights given as arrays need the `grid` of their points!"
                )
            fragment_weights = weights

        return FragmentOperators.from_grid(
            grid.points,
            grid.weights,
            fragment_weights,
            basisset,
            dipole=dipole,
            chunk_size=chunk_size,
            partition_of_unity=partition_of_unity,
        )

    def save_cube(
        self,
        matrix: np.ndarray | Sequence[np.ndarray],
        obj_type: str | Iterable[str] = "density",
        filename: str | os.PathLike | Iterable[str | os.PathLike] = "density.cube",
        **kwargs,
    ):
        """
        Evaluate `matrix` values on a grid and save to a .cube file.

        Args:
            matrix (np.ndarray | Sequence[np.ndarray]): Either density matrix,
                orbital coefficients vector, or list of such objects.
            obj_type (str | Iterable[str], optional): Specifies the `matrix` argument type(s),
                either "density", "orbital", "esp" or a list with such strings.
                Defaults to "density".
            filename (str | os.PathLike | Iterable[str | os.PathLike], optional): File path(s)
                to save resulting cube(s), one per matrix. Defaults to "density.cube".

        Raises:
            ValueError: If the number of filenames differs from the number of matrices.
        """

        if isinstance(matrix, np.ndarray):
            matrix = [matrix]
        else:
            matrix = list(matrix)

        if isinstance(filename, (str, os.PathLike)):
            filename = [filename]
        else:
            filename = list(filename)

        # check before the grid evaluation, which is the expensive part
        if len(filename) != len(matrix):
            psi4.core.clean()
            raise ValueError(
                f"One filename per matrix is required, got {len(matrix)} matrices "
                f"and {len(filename)} filenames."
            )

        cubes = self.make_cube(matrix, obj_type=obj_type, **kwargs)

        if isinstance(cubes, Cube):
            cubes = [cubes]

        for cube, fname in zip(cubes, filename):
            cube.save(fname)
