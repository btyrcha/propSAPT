"""
Fragment population operators: weighted-overlap matrices of a partition of space.

Given a molecular quadrature grid (points r_q, weights w_q) and fragment weights w_X(r_q)
at its points, the weighted-overlap matrix of fragment X is

    W_X[mu,nu] = sum_q w_q w_X(r_q) phi_mu(r_q) phi_nu(r_q),

computed once per geometry. The number of electrons any AO density matrix D puts on
fragment X is then a trace, Tr(D W_X), with no further grid work. The first-moment
matrices (r_q times the same weights) give fragment dipoles in the same way.

Points are in bohr. Matrices are in the AO basis of the given Psi4 basis set, in Psi4 AO
order, and no spin factor is applied, as in `prop_sapt.evaluate`: the density matrices
of `calc_densities` follow the `Da` convention, pass `2 * D` for the total (RHF) density.

Fragment weights come from any partitioning: the Becke partition of Psi4's molecular grid
(`becke_fragment_weights`), Hirshfeld-like weights from the monomer densities
(`hirshfeld_fragment_weights`), or weights computed elsewhere, e.g. LISA or MBIS weights
from horton_part on its own grid, passed as arrays with that grid's points and weights.

Classes
-------
QuadratureGrid
    Points, weights and parent atoms of a molecular quadrature grid.
FragmentOperators
    Weighted-overlap and first-moment matrices of the fragments, with populations,
    charges and dipoles of density matrices.

Functions
---------
molecular_grid
    Psi4's Becke-partitioned molecular quadrature grid of a basis set's molecule.
becke_fragment_weights
    Becke fragment weights on a `QuadratureGrid`, from the atoms of each fragment.
hirshfeld_fragment_weights
    Hirshfeld-like fragment weights from fragment densities, e.g. of the monomers.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import numpy as np
import psi4
from psi4.driver import p4util

from .checks import check_charge_closure, check_partition_of_unity
from .evaluate import (
    _check_basisset,
    _get_chunk_size,
    _iter_phi_chunks,
    _prepare_points,
    _stack_matrices,
    evaluate_density,
)
from .options import PROPSAPT_OPTIONS

# below this total density the Hirshfeld weights are split equally
_HIRSHFELD_DENSITY_FLOOR = 1.0e-30


@dataclass(frozen=True, eq=False)
class QuadratureGrid:
    """
    Molecular quadrature grid: the integral of f over space is sum(weights * f(points)).
    """

    #: Points in bohr, shape (n, 3).
    points: np.ndarray
    #: Quadrature weights, shape (n,), with the Becke partition of space between the
    #: atoms included.
    weights: np.ndarray
    #: Index of the atom whose Becke cell holds each point, shape (n,), in the order of
    #: the atoms of the grid's molecule.
    atom_index: np.ndarray


def molecular_grid(
    basisset: psi4.core.BasisSet,
    radial_points: int | None = None,
    spherical_points: int | None = None,
    nuclear_scheme: str = "TREUTLER",
    bs_radius_alpha: float | None = None,
) -> QuadratureGrid:
    """
    Build Psi4's molecular quadrature grid around the atoms of `basisset.molecule()`.

    Every atom, ghost atoms included, gets a Treutler radial grid times a Lebedev
    spherical grid, unpruned, and space is divided between the atoms by `nuclear_scheme`.
    The grid options are set here and do not follow the global Psi4 DFT options.

    Args:
        basisset (psi4.core.BasisSet): Basis set whose molecule carries the grid.
        radial_points (int | None, optional): Radial points per atom.
            Defaults to the `quadrature_radial_points` option.
        spherical_points (int | None, optional): Lebedev points per radial shell, one of
            Psi4's Lebedev orders. Defaults to the `quadrature_spherical_points` option.
        nuclear_scheme (str, optional): Partition of space between the atoms, Psi4's
            `DFT_NUCLEAR_SCHEME` ("TREUTLER", "BECKE", "STRATMANN", ...).
            Defaults to "TREUTLER".
        bs_radius_alpha (float | None, optional): Scale of the Bragg-Slater radii setting
            the radial extent, Psi4's `DFT_BS_RADIUS_ALPHA`.
            Defaults to the `quadrature_bs_radius_alpha` option.

    Returns:
        QuadratureGrid: Points, weights and parent atoms of the grid.
    """

    _check_basisset(basisset)
    radial_points = _positive_int(
        radial_points, "quadrature_radial_points", "radial_points"
    )
    spherical_points = _positive_int(
        spherical_points, "quadrature_spherical_points", "spherical_points"
    )
    if bs_radius_alpha is None:
        bs_radius_alpha = PROPSAPT_OPTIONS["quadrature_bs_radius_alpha"]
    if (
        isinstance(bs_radius_alpha, bool)
        or not isinstance(bs_radius_alpha, (int, float, np.integer, np.floating))
        or not bs_radius_alpha > 0.0
    ):
        raise ValueError(
            f"`bs_radius_alpha` should be a positive number, was {bs_radius_alpha}!"
        )

    int_options = {
        "DFT_RADIAL_POINTS": radial_points,
        "DFT_SPHERICAL_POINTS": spherical_points,
    }
    str_options = {
        "DFT_NUCLEAR_SCHEME": str(nuclear_scheme).upper(),
        "DFT_RADIAL_SCHEME": "TREUTLER",
        "DFT_PRUNING_SCHEME": "NONE",
        "DFT_BLOCK_SCHEME": "ATOMIC",  # every block of points belongs to one atom
    }

    # the Python binding takes only integer and string options,
    # the radial extent is read from the global options
    saved_options = p4util.OptionsState(["DFT_BS_RADIUS_ALPHA"])
    try:
        psi4.core.set_global_option("DFT_BS_RADIUS_ALPHA", float(bs_radius_alpha))
        grid = psi4.core.DFTGrid.build(
            basisset.molecule(), basisset, int_options, str_options
        )
    finally:
        saved_options.restore()

    points, weights, atom_index = [], [], []
    for block in grid.blocks():
        n_block = block.npoints()
        points.append(
            np.column_stack(
                [np.asarray(v)[:n_block] for v in (block.x(), block.y(), block.z())]
            )
        )
        weights.append(np.asarray(block.w())[:n_block])
        atom_index.append(np.full(n_block, block.parent_atom(), dtype=int))

    atom_index = np.concatenate(atom_index)
    natom = basisset.molecule().natom()
    if np.any((atom_index < 0) | (atom_index >= natom)):
        raise RuntimeError("Psi4 returned grid blocks without a parent atom!")

    return QuadratureGrid(
        points=np.concatenate(points),
        weights=np.concatenate(weights),
        atom_index=atom_index,
    )


def becke_fragment_weights(
    grid: QuadratureGrid, fragment_atoms: Mapping[str, Iterable[int]]
) -> dict[str, np.ndarray]:
    """
    Becke fragment weights: 1 at the points in the Becke cells of the fragment's atoms,
    0 elsewhere. With the grid's weights they integrate over the fragment's cells.

    Every atom that carries grid points must belong to one fragment, ghost atoms
    included, so that the weights sum to 1 at every point.

    Args:
        grid (QuadratureGrid): Grid from `molecular_grid`.
        fragment_atoms (Mapping[str, Iterable[int]]): Atom indices of each fragment,
            e.g. {"A": range(0, 3), "B": range(3, 6)}.

    Returns:
        dict[str, np.ndarray]: Weights of each fragment, shape (n,).
    """

    _check_fragment_names(fragment_atoms)

    atoms_of = {}  # fragment name -> its atom indices
    owner = {}  # atom index -> its fragment name
    for name, atoms in fragment_atoms.items():
        atoms_of[name] = []
        for atom in atoms:
            if isinstance(atom, bool) or not isinstance(atom, (int, np.integer)):
                raise ValueError(
                    f"Atom indices should be integers, fragment `{name}` has {atom!r}!"
                )
            atom = int(atom)
            if atom in owner:
                raise ValueError(
                    f"Atom {atom} is in fragments `{owner[atom]}` and `{name}`!"
                )
            owner[atom] = name
            atoms_of[name].append(atom)

    unassigned = sorted(set(np.unique(grid.atom_index).tolist()) - set(owner))
    if unassigned:
        raise ValueError(
            f"Atoms {unassigned} carry grid points but belong to no fragment, so the "
            "weights would not sum to 1. Assign them (e.g. ghost atoms of midbond "
            "functions) to a fragment, or use Hirshfeld weights."
        )

    return {
        name: np.isin(grid.atom_index, atoms).astype(float)
        for name, atoms in atoms_of.items()
    }


def hirshfeld_fragment_weights(
    points: np.ndarray,
    densities: Mapping[str, np.ndarray],
    basisset: psi4.core.BasisSet,
    chunk_size: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Hirshfeld-like fragment weights, w_X(r) = rho_X(r) / sum_Y rho_Y(r), from fragment
    densities, e.g. the unperturbed monomer densities of a dimer.

    Negative density values (rounding) count as 0, and where the total density is below
    1e-30 the weights are split equally between the fragments.

    Args:
        points (np.ndarray): Points in bohr, shape (..., 3).
        densities (Mapping[str, np.ndarray]): AO density matrix of each fragment,
            shape (nbf, nbf), e.g. {"A": 2 * Da_A, "B": 2 * Da_B}.
        basisset (psi4.core.BasisSet): Basis set of the density matrices.
        chunk_size (int | None, optional): Number of points evaluated at once.
            Defaults to the `point_eval_chunk_size` option.

    Returns:
        dict[str, np.ndarray]: Weights of each fragment, with the shape of `points`
            without its last axis.
    """

    names = _check_fragment_names(densities)
    rho = evaluate_density(
        [densities[name] for name in names], points, basisset, chunk_size=chunk_size
    )
    rho = np.clip(rho, 0.0, None)

    total = rho.sum(axis=0)
    empty = total < _HIRSHFELD_DENSITY_FLOOR
    weights = np.where(empty, 1.0 / len(names), rho / np.where(empty, 1.0, total))

    return {name: weights[i] for i, name in enumerate(names)}


class FragmentOperators:
    """
    Weighted-overlap matrices W_X and first-moment matrices R_X of a set of fragments,

        W_X[mu,nu] = integral(w_X(r) phi_mu(r) phi_nu(r) dr),
        R_X[k,mu,nu] = integral(w_X(r) r_k phi_mu(r) phi_nu(r) dr),

    with r about the coordinate origin. Build them with `from_grid`; populations, charges
    and dipoles of density matrices are then traces.

    When the fragments cover the whole system, i.e. their weights are a partition of
    unity (the default of `from_grid`), the weights are checked to sum to 1, and every
    `population` and `charge` call checks that the populations of all fragments add up
    to the electrons Tr(dm S), see `closure_error` and `prop_sapt.checks`.
    """

    def __init__(
        self,
        overlaps: Mapping[str, np.ndarray],
        first_moments: Mapping[str, np.ndarray] | None = None,
        ao_overlap: np.ndarray | None = None,
    ):
        """
        Args:
            overlaps (Mapping[str, np.ndarray]): W_X of each fragment, shape (nbf, nbf).
            first_moments (Mapping[str, np.ndarray] | None, optional): R_X of each
                fragment, shape (3, nbf, nbf). Without them `dipole` is not available.
                Defaults to None.
            ao_overlap (np.ndarray | None, optional): The analytic AO overlap matrix S,
                shape (nbf, nbf). Giving it declares that the fragment weights are a
                partition of unity, i.e. the fragments cover the whole system:
                `population` and `charge` then check that the populations of all
                fragments add up to Tr(dm S), and `closure_error` is available.
                Defaults to None.
        """

        names = _check_fragment_names(overlaps)
        nbf = len(np.atleast_1d(overlaps[names[0]]))

        self._overlaps = {}
        for name in names:
            matrix = np.array(overlaps[name], dtype=float)
            if matrix.shape != (nbf, nbf):
                raise ValueError(
                    f"Every overlap should have the same shape (nbf, nbf), "
                    f"fragment `{name}` has {matrix.shape}!"
                )
            matrix.flags.writeable = False
            self._overlaps[name] = matrix

        self._first_moments = None
        if first_moments is not None:
            if set(first_moments) != set(names):
                raise ValueError(
                    "`first_moments` should have the same fragments as `overlaps`, "
                    f"has {sorted(first_moments)} instead of {sorted(names)}!"
                )
            self._first_moments = {}
            for name in names:
                matrix = np.array(first_moments[name], dtype=float)
                if matrix.shape != (3, nbf, nbf):
                    raise ValueError(
                        f"Every first moment should have shape (3, {nbf}, {nbf}), "
                        f"fragment `{name}` has {matrix.shape}!"
                    )
                matrix.flags.writeable = False
                self._first_moments[name] = matrix

        self._ao_overlap = None
        if ao_overlap is not None:
            matrix = np.array(ao_overlap, dtype=float)
            if matrix.shape != (nbf, nbf):
                raise ValueError(
                    f"`ao_overlap` should have shape ({nbf}, {nbf}), was {matrix.shape}!"
                )
            matrix.flags.writeable = False
            self._ao_overlap = matrix

        self._nbf = nbf

    @classmethod
    def from_grid(
        cls,
        points: np.ndarray,
        weights: np.ndarray,
        fragment_weights: Mapping[str, np.ndarray],
        basisset: psi4.core.BasisSet,
        dipole: bool = True,
        chunk_size: int | None = None,
        partition_of_unity: bool = True,
    ) -> "FragmentOperators":
        """
        Compute the fragment operators by quadrature, in one pass over the points.

        With `partition_of_unity` (the default) the fragment weights are first checked
        to sum to 1 wherever the basis functions reach
        (`prop_sapt.checks.check_partition_of_unity`), and the operators get the
        analytic overlap matrix, so that every population is checked to close
        (`closure_error`). A failed check warns, see `prop_sapt.checks`.

        Args:
            points (np.ndarray): Points of the quadrature grid in bohr, shape (..., 3).
            weights (np.ndarray): Quadrature weights, with the shape of `points`
                without its last axis.
            fragment_weights (Mapping[str, np.ndarray]): Weight w_X of each fragment at
                the points, with the same shape as `weights`.
            basisset (psi4.core.BasisSet): Basis set of the operators.
            dipole (bool, optional): Also compute the first-moment matrices, needed for
                `dipole`. Defaults to True.
            chunk_size (int | None, optional): Number of points evaluated at once.
                Defaults to the `point_eval_chunk_size` option.
            partition_of_unity (bool, optional): Whether the fragments together cover
                the whole system, i.e. `fragment_weights` sum to 1 at every point. If
                True, this is checked here, and the operators get the analytic overlap
                matrix S, so that every `population` and `charge` call checks that the
                populations of all fragments add up to Tr(dm S). Pass False for weights
                that do not sum to 1, e.g. only some of the fragments or atoms, or a
                grid covering part of space: nothing is checked then, and
                `closure_error` is not available. Defaults to True.

        Returns:
            FragmentOperators: Operators of the fragments in the basis of `basisset`.
        """

        _check_basisset(basisset)
        points = _prepare_points(points)
        chunk_size = _get_chunk_size(chunk_size)
        names = _check_fragment_names(fragment_weights)

        points_shape = points.shape[:-1]
        weights = _prepare_point_values(weights, points_shape, "`weights`")
        fragment_weights = {
            name: _prepare_point_values(
                fragment_weights[name], points_shape, f"Weights of fragment `{name}`"
            )
            for name in names
        }
        flat_points = points.reshape(-1, 3)

        if partition_of_unity:
            check_partition_of_unity(
                flat_points,
                weights,
                fragment_weights,
                basisset,
                chunk_size=chunk_size,
            )

        nbf = basisset.nbf()
        overlaps = {name: np.zeros((nbf, nbf)) for name in names}
        first_moments = (
            {name: np.zeros((3, nbf, nbf)) for name in names} if dipole else None
        )

        for chunk, phi in _iter_phi_chunks(flat_points, basisset, chunk_size):
            chunk_points = flat_points[chunk]
            for name in names:
                weighted_phi = (weights[chunk] * fragment_weights[name][chunk])[
                    :, None
                ] * phi
                overlaps[name] += phi.T @ weighted_phi
                if dipole:
                    for k in range(3):
                        first_moments[name][k] += phi.T @ (
                            chunk_points[:, k, None] * weighted_phi
                        )

        # symmetric by construction, remove the rounding of the products
        overlaps = {name: 0.5 * (m + m.T) for name, m in overlaps.items()}
        if dipole:
            first_moments = {
                name: 0.5 * (m + m.transpose(0, 2, 1))
                for name, m in first_moments.items()
            }

        ao_overlap = None
        if partition_of_unity:
            ao_overlap = np.asarray(psi4.core.MintsHelper(basisset).ao_overlap())

        return cls(overlaps, first_moments, ao_overlap=ao_overlap)

    @property
    def fragments(self) -> tuple[str, ...]:
        """Names of the fragments."""

        return tuple(self._overlaps)

    @property
    def nbf(self) -> int:
        """Number of basis functions of the operators."""

        return self._nbf

    @property
    def ao_overlap(self) -> np.ndarray | None:
        """The analytic AO overlap matrix S, read-only, if the fragment weights are a
        partition of unity; None otherwise."""

        return self._ao_overlap

    def overlap(self, fragment: str) -> np.ndarray:
        """
        Weighted-overlap matrix W_X of `fragment`, shape (nbf, nbf), read-only.
        The sum over fragments that cover the whole system is the overlap matrix, up to
        quadrature error.
        """

        self._check_fragment(fragment)

        return self._overlaps[fragment]

    def first_moment(self, fragment: str) -> np.ndarray:
        """
        First-moment matrices R_X of `fragment`, shape (3, nbf, nbf), read-only,
        with r about the coordinate origin and no charge sign.
        """

        self._check_fragment(fragment)
        self._check_first_moments()

        return self._first_moments[fragment]

    def population(
        self, fragment: str, dm: np.ndarray | Iterable[np.ndarray]
    ) -> float | np.ndarray:
        """
        Number of electrons of AO density matrix `dm` on `fragment`, Tr(dm W_X).
        No spin factor is applied: for the total density of an RHF matrix pass `2 * Da`.

        Args:
            fragment (str): Name of the fragment.
            dm (np.ndarray | Iterable[np.ndarray]): AO density matrix, shape (nbf, nbf),
                or several of them, as a sequence or an array of shape (M, nbf, nbf).

        Returns:
            float | np.ndarray: Population, a float for a single matrix
                and shape (M,) for several.

        Warns:
            InvariantWarning: For a partition of unity, if the populations of all
                fragments miss Tr(dm S) by more than the `check_closure_atol` option,
                see `closure_error`.
        """

        self._check_fragment(fragment)
        dm_stack, single = _stack_matrices(dm, self._nbf)

        populations = self._populations(dm_stack)
        if self._ao_overlap is not None:
            check_charge_closure(populations, self._electrons(dm_stack))

        population = populations[fragment]

        return float(population[0]) if single else population

    def charge(
        self, fragment: str, dm: np.ndarray | Iterable[np.ndarray]
    ) -> float | np.ndarray:
        """
        Electronic charge of AO density matrix `dm` on `fragment`, -Tr(dm W_X).
        For an interaction-induced density this is the charge moved onto the fragment.
        No spin factor is applied: for the total density of an RHF matrix pass `2 * Da`.

        Args:
            fragment (str): Name of the fragment.
            dm (np.ndarray | Iterable[np.ndarray]): AO density matrix, shape (nbf, nbf),
                or several of them, as a sequence or an array of shape (M, nbf, nbf).

        Returns:
            float | np.ndarray: Charge, a float for a single matrix
                and shape (M,) for several.

        Warns:
            InvariantWarning: As `population`.
        """

        return -self.population(fragment, dm)

    def closure_error(
        self, dm: np.ndarray | Iterable[np.ndarray]
    ) -> float | np.ndarray:
        """
        Electrons of AO density matrix `dm` that the fragments miss or count twice,
        sum_X Tr(dm W_X) - Tr(dm S): the quadrature error of the grid and the defect of
        the fragment weights, for this density. No spin factor is applied to `dm`.

        Args:
            dm (np.ndarray | Iterable[np.ndarray]): AO density matrix, shape (nbf, nbf),
                or several of them, as a sequence or an array of shape (M, nbf, nbf).

        Raises:
            ValueError: If the operators have no `ao_overlap`, i.e. the fragment
                weights are not a partition of unity.

        Returns:
            float | np.ndarray: Closure error in electrons, a float for a single matrix
                and shape (M,) for several.
        """

        if self._ao_overlap is None:
            raise ValueError(
                "No AO overlap matrix: the closure error needs fragment weights that "
                "are a partition of unity, built by `from_grid` with "
                "`partition_of_unity=True` or given `ao_overlap`!"
            )
        dm_stack, single = _stack_matrices(dm, self._nbf)

        closure = sum(self._populations(dm_stack).values()) - self._electrons(dm_stack)

        return float(closure[0]) if single else closure

    def dipole(
        self,
        fragment: str,
        dm: np.ndarray | Iterable[np.ndarray],
        origin: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Electronic dipole moment of AO density matrix `dm` on `fragment`,
        -integral(w_X(r) (r - origin) rho(r) dr) = -Tr(dm R_X) + origin Tr(dm W_X),
        in atomic units. Electrons contribute negatively, as in Psi4's `ao_dipole`, so
        at the coordinate origin the dipoles of fragments that cover the whole system
        sum to the electronic dipole Tr(dm @ ao_dipole). No spin factor is applied to
        `dm`.

        Args:
            fragment (str): Name of the fragment.
            dm (np.ndarray | Iterable[np.ndarray]): AO density matrix, shape (nbf, nbf),
                or several of them, as a sequence or an array of shape (M, nbf, nbf).
            origin (np.ndarray | None, optional): Origin of the dipole in bohr,
                shape (3,). Defaults to the coordinate origin.

        Returns:
            np.ndarray: Dipole, shape (3,) for a single matrix and (M, 3) for several.
        """

        self._check_fragment(fragment)
        self._check_first_moments()
        dm_stack, single = _stack_matrices(dm, self._nbf)

        dipole = -np.einsum("mij,kji->mk", dm_stack, self._first_moments[fragment])
        if origin is not None:
            origin = np.asarray(origin, dtype=float)
            if origin.shape != (3,):
                raise ValueError(
                    f"`origin` should have shape (3,), was {origin.shape}!"
                )
            population = np.einsum("mij,ji->m", dm_stack, self._overlaps[fragment])
            dipole += population[:, None] * origin

        return dipole[0] if single else dipole

    def _populations(self, dm_stack: np.ndarray) -> dict[str, np.ndarray]:
        """Populations Tr(dm W_X) of every fragment, shape (M,), for stacked matrices."""

        return {
            name: np.einsum("mij,ji->m", dm_stack, overlap)
            for name, overlap in self._overlaps.items()
        }

    def _electrons(self, dm_stack: np.ndarray) -> np.ndarray:
        """Electrons Tr(dm S), shape (M,), for stacked matrices."""

        return np.einsum("mij,ji->m", dm_stack, self._ao_overlap)

    def _check_fragment(self, fragment):
        """Raises `KeyError` if `fragment` is not one of the fragments."""

        if fragment not in self._overlaps:
            raise KeyError(
                f"Unknown fragment `{fragment}`, the fragments are {list(self._overlaps)}!"
            )

    def _check_first_moments(self):
        """Raises `ValueError` if the first moments were not computed."""

        if self._first_moments is None:
            raise ValueError(
                "No first moments, build the operators with `dipole=True` for dipoles!"
            )


def _positive_int(value, option_key: str, label: str) -> int:
    """Returns `value`, or the `option_key` option if None, checking it is a positive int."""

    if value is None:
        value = PROPSAPT_OPTIONS[option_key]

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"`{label}` should be a positive integer, was {value}!")

    return int(value)


def _check_fragment_names(fragments: Mapping) -> list[str]:
    """Returns the fragment names of a nonempty mapping with string keys."""

    if not isinstance(fragments, Mapping):
        raise TypeError(
            f"Fragments should be given as a mapping of names, e.g. a dict, "
            f"was `{type(fragments)}`!"
        )
    if len(fragments) == 0:
        raise ValueError("No fragments given!")
    for name in fragments:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"Fragment names should be nonempty strings, was {name!r}!"
            )

    return list(fragments)


def _prepare_point_values(values, points_shape: tuple, label: str) -> np.ndarray:
    """Returns finite values given at the points, flattened to shape (n,)."""

    values = np.asarray(values, dtype=float)
    if values.shape != points_shape:
        raise ValueError(
            f"{label} should have the shape {points_shape} of the points, "
            f"was {values.shape}!"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} should be finite!")

    return values.reshape(-1)
