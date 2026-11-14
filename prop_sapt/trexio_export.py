"""
Export of the interaction-induced density changes to TREXIO files, to view them in VIAMD.

- **What is written:** TREXIO (https://github.com/TREX-CoE/trexio) has no AO density
  matrix: a one-particle density is given by orbitals and their occupations. Each density
  change is therefore written as its natural orbitals (`prop_sapt.natural_orbitals`), one
  file per density, ``delta_dm_<key>.h5``, with the names of the cube files.
- **Signs as spins:** VIAMD (https://github.com/scanberg/viamd) shows a signed density
  only as the difference of the alpha and beta densities. The orbitals of positive
  occupation are therefore written as alpha orbitals (``mo.spin`` 0), and those of
  negative occupation as beta orbitals with ``|n|``. The occupations are for both spins
  (twice the one-spin ``n``).
- **Reading a file:** its alpha - beta density is the total density change, and alpha +
  beta means nothing. Every file says this in ``metadata.description``; other TREXIO
  readers would take the orbitals for spin orbitals.
- **The basis set:** Psi4's basis set is written as it is, because Psi4's order of the AOs
  in a shell (spherical: m = 0, +1, -1, ...; Cartesian: alphabetical) and its
  normalisation are TREXIO's.
- **Nuclei:** ghost atoms are nuclei of charge 0 labelled "X". Atoms with an ECP have their
  charge reduced by its core electrons; the ``ecp`` group is not written.
- **Optional dependency:** trexio is needed only here, installed with ``pip install trexio``
  or the ``trexio`` extra. It is imported when a file is written.

Functions
---------
export_trexio
    Write density changes into TREXIO files, one per density.
import_trexio
    Import trexio, explaining how to install it when it is missing.
"""

import os
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psi4

from ._version import __version__
from .natural_orbitals import NaturalOrbitals, density_natural_orbitals

#: How to install the optional trexio package.
TREXIO_INSTALL = 'pip install trexio (or: pip install "prop_sapt[trexio]")'
#: Prefix of the file names, as for the cube files: ``delta_dm_<key>.h5``.
TREXIO_PREFIX = "delta_dm_"
#: Extension of the files (TREXIO's HDF5 back end).
TREXIO_SUFFIX = ".h5"
#: Default of `export_trexio`'s `relative_cutoff`: the natural orbitals left out are the
#: numerically zero ones (a density change has far fewer non-zero ones than orbitals), and
#: the density is unchanged to round-off.
DEFAULT_RELATIVE_CUTOFF = 1.0e-12
#: Highest angular momentum VIAMD reads (g): it refuses files with higher shells.
VIAMD_MAX_ANGULAR_MOMENTUM = 4
#: The encoding of the orbitals, recorded in the files and in a bundle's metadata.
ENCODING = (
    "natural orbitals of the density change: alpha orbitals carry the positive "
    "occupations and beta orbitals the negative ones, as |n|; the occupations are for "
    "the total density (2 x the one-spin matrix), so the density change is the "
    "alpha - beta density"
)

# Largest |coefficient - Psi4's| allowed, relative to the shell's largest coefficient
_COEFFICIENT_RTOL = 1.0e-10


class TrexioExportError(RuntimeError):
    """Raised when TREXIO files cannot be written: trexio is missing, or the basis set
    cannot be written."""


def import_trexio():
    """
    Import the trexio package.

    Raises:
        TrexioExportError: If trexio is not installed, saying how to install it.

    Returns:
        module: The trexio module.
    """

    try:
        import trexio
    except ImportError as err:
        raise TrexioExportError(
            f"Writing TREXIO files needs the trexio package ({err}). Install it with:\n"
            f"  {TREXIO_INSTALL}"
        ) from err

    return trexio


def export_trexio(
    densities: Mapping[str, np.ndarray],
    directory: str | os.PathLike = ".",
    *,
    dimer=None,
    keys: Iterable[str] | None = None,
    relative_cutoff: float = DEFAULT_RELATIVE_CUTOFF,
) -> list[Path]:
    """
    Write density changes into TREXIO files, one per density, as natural orbitals.

    See `prop_sapt.trexio_export` for the encoding. A file of the same name is replaced.

    Args:
        densities (Mapping[str, np.ndarray]): AO density matrices as `calc_densities`
            returns them (the default ``orbital_basis="AO"``), keyed "<component>_A",
            "<component>_B" and "<component>" for the dimer.
        directory (str | os.PathLike, optional): Directory of the files, created if
            needed. Defaults to ".".
        dimer (Dimer): The dimer of the matrices: its geometry, basis set and orbitals.
        keys (Iterable[str] | None, optional): The densities to write. Defaults to every
            key of `densities`.
        relative_cutoff (float, optional): Natural orbitals with ``|n|`` up to
            `relative_cutoff` times the largest ``|n|`` of their density are left out; 0
            keeps all but those of zero occupation. Defaults to `DEFAULT_RELATIVE_CUTOFF`.

    Raises:
        TrexioExportError: If trexio is missing, or the basis set mixes spherical and
            Cartesian shells.
        TypeError: If no `dimer` is given.

    Warns:
        UserWarning: If the basis set has shells above g, which VIAMD does not read.

    Returns:
        list[Path]: The files written, in the order of `keys`.
    """

    trexio = import_trexio()

    if dimer is None:
        raise TypeError("`dimer` is needed for its geometry, basis set and orbitals!")

    matrices = densities
    keys = list(matrices) if keys is None else list(keys)
    system = _system_arrays(dimer.fragments, dimer.get_psi4_basisset())
    orbitals = density_natural_orbitals(matrices, dimer, keys=keys)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    paths = []
    for key in keys:
        path = directory / f"{TREXIO_PREFIX}{key}{TREXIO_SUFFIX}"
        selected = _select(orbitals[key], relative_cutoff)
        _write_file(trexio, path, system, selected, _description(key))
        paths.append(path)

    return paths


def _select(orbitals: NaturalOrbitals, relative_cutoff: float) -> NaturalOrbitals:
    """The orbitals with ``|n|`` above `relative_cutoff` times the largest ``|n|``."""

    largest = np.max(np.abs(orbitals.occupations), initial=0.0)
    return orbitals.select(relative_cutoff * largest)


@dataclass(frozen=True)
class _System:
    """The nucleus, basis and ao groups of the files, the same in each."""

    labels: list[str]
    charges: np.ndarray
    coordinates: np.ndarray
    cartesian: bool
    shell_center: list[int]
    shell_am: list[int]
    shell_factor: np.ndarray
    prim_shell: list[int]
    exponents: np.ndarray
    coefficients: np.ndarray
    prim_factor: np.ndarray
    ao_shell: list[int]


def _system_arrays(fragments, basisset: psi4.core.BasisSet) -> _System:
    """The nuclei of `fragments` (bohr) and `basisset`, in TREXIO's conventions."""

    shells = [basisset.shell(index) for index in range(basisset.nshell())]

    # p shells count too: Psi4 orders spherical ones z, x, y and Cartesian ones x, y, z
    pure = {shell.is_pure() for shell in shells if shell.am > 0}
    if len(pure) > 1:
        raise TrexioExportError(
            f"Basis set {basisset.name()!r} mixes spherical and Cartesian shells, which "
            "VIAMD does not read; use a basis set of one kind."
        )

    max_am = basisset.max_am()
    if max_am > VIAMD_MAX_ANGULAR_MOMENTUM:
        warnings.warn(
            f"Basis set {basisset.name()!r} has shells up to l = {max_am}: VIAMD reads "
            f"shells up to l = {VIAMD_MAX_ANGULAR_MOMENTUM} (g) and will refuse these "
            "TREXIO files.",
            UserWarning,
            stacklevel=3,
        )

    molecule = basisset.molecule()
    charges = []
    labels = []
    for atom in range(fragments.natom):
        if fragments.real[atom]:
            core = (
                basisset.n_ecp_core(molecule.label(atom)) if basisset.has_ECP() else 0
            )
            charges.append(float(fragments.atomic_numbers[atom] - core))
            labels.append(fragments.symbols[atom].capitalize())
        else:
            charges.append(0.0)
            labels.append("X")

    exponents, coefficients, prim_factor, prim_shell, shell_factor = [], [], [], [], []
    for index, shell in enumerate(shells):
        exps = np.array([shell.exp(k) for k in range(shell.nprimitive)])
        coefs = np.array([shell.original_coef(k) for k in range(shell.nprimitive)])
        prim, factor = _normalisation(shell.am, exps, coefs)

        # TREXIO's normalisation of the shell is Psi4's
        psi4_coefs = np.array([shell.coef(k) for k in range(shell.nprimitive)])
        if not np.max(np.abs(coefs * prim * factor - psi4_coefs)) <= (
            _COEFFICIENT_RTOL * np.max(np.abs(psi4_coefs))
        ):
            raise TrexioExportError(
                f"Shell {index} of basis set {basisset.name()!r} is not normalised as "
                "TREXIO expects it."
            )

        exponents.extend(exps)
        coefficients.extend(coefs)
        prim_factor.extend(prim)
        prim_shell.extend([index] * shell.nprimitive)
        shell_factor.append(factor)

    return _System(
        labels=labels,
        charges=np.array(charges),
        coordinates=np.array(fragments.coordinates, dtype=np.float64),
        cartesian=not (pure.pop() if pure else basisset.has_puream()),
        shell_center=[basisset.shell_to_center(index) for index in range(len(shells))],
        shell_am=[shell.am for shell in shells],
        shell_factor=np.array(shell_factor),
        prim_shell=prim_shell,
        exponents=np.array(exponents),
        coefficients=np.array(coefficients),
        prim_factor=np.array(prim_factor),
        ao_shell=[
            index for index, shell in enumerate(shells) for _ in range(shell.nfunction)
        ],
    )


def _double_factorial(n: int) -> int:
    return 1 if n <= 0 else n * _double_factorial(n - 2)


def _normalisation(am: int, exps: np.ndarray, coefs: np.ndarray) -> tuple:
    """The primitive factors and the shell factor that normalise the x^l component of a
    contracted shell: TREXIO's ``basis.prim_factor`` and ``basis.shell_factor``."""

    df = _double_factorial(2 * am - 1)
    prim = (2 * exps / np.pi) ** 0.75 * (4 * exps) ** (am / 2) / np.sqrt(df)
    weighted = coefs * prim
    sums = exps[:, None] + exps[None, :]
    overlap = (
        df * np.pi**1.5 / (2**am * sums ** (am + 1.5)) * np.outer(weighted, weighted)
    )

    return prim, 1.0 / np.sqrt(overlap.sum())


def _description(key: str) -> str:
    """The ``metadata.description`` of the file of `key`."""

    if key.endswith("_A") or key.endswith("_B"):
        what = f"monomer {key[-1]}"
    else:
        what = "the dimer (A + B)"

    return (
        f"propSAPT interaction-induced density change {key!r} of {what}, as {ENCODING}."
    )


def _write_file(trexio, path: Path, system: _System, orbitals: NaturalOrbitals, text):
    """Write one TREXIO file: the system and the orbitals of `orbitals`, alpha for the
    positive occupations and beta for the negative ones."""

    if path.exists():
        path.unlink()  # mode "w" adds to an existing file

    positive = np.where(orbitals.occupations > 0.0)[0]
    negative = np.where(orbitals.occupations < 0.0)[0][::-1]  # decreasing |n|
    order = np.concatenate([positive, negative])

    handle = trexio.File(str(path), mode="w", back_end=trexio.TREXIO_HDF5)
    try:
        trexio.write_metadata_code_num(handle, 2)
        trexio.write_metadata_code(
            handle, [f"propSAPT {__version__}", f"Psi4 {psi4.__version__}"]
        )
        trexio.write_metadata_description(handle, text)

        trexio.write_nucleus_num(handle, len(system.labels))
        trexio.write_nucleus_charge(handle, system.charges)
        trexio.write_nucleus_coord(handle, system.coordinates)
        trexio.write_nucleus_label(handle, system.labels)

        trexio.write_basis_type(handle, "Gaussian")
        trexio.write_basis_shell_num(handle, len(system.shell_am))
        trexio.write_basis_prim_num(handle, len(system.exponents))
        trexio.write_basis_nucleus_index(handle, system.shell_center)
        trexio.write_basis_shell_ang_mom(handle, system.shell_am)
        trexio.write_basis_shell_factor(handle, system.shell_factor)
        trexio.write_basis_r_power(handle, [0] * len(system.shell_am))
        trexio.write_basis_shell_index(handle, system.prim_shell)
        trexio.write_basis_exponent(handle, system.exponents)
        trexio.write_basis_coefficient(handle, system.coefficients)
        trexio.write_basis_prim_factor(handle, system.prim_factor)

        trexio.write_ao_cartesian(handle, int(system.cartesian))
        trexio.write_ao_num(handle, len(system.ao_shell))
        trexio.write_ao_shell(handle, system.ao_shell)
        trexio.write_ao_normalization(handle, np.ones(len(system.ao_shell)))

        if len(order):
            trexio.write_mo_type(handle, "Natural")
            trexio.write_mo_num(handle, len(order))
            trexio.write_mo_coefficient(
                handle, np.ascontiguousarray(orbitals.coefficients[:, order].T)
            )
            trexio.write_mo_occupation(
                handle, 2.0 * np.abs(orbitals.occupations[order])
            )
            trexio.write_mo_spin(
                handle,
                np.array([0] * len(positive) + [1] * len(negative), dtype=np.int32),
            )
    finally:
        handle.close()
