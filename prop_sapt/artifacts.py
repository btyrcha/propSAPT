"""
Self-describing density artifacts: interaction-induced density matrices in one ``.npz``
bundle that records what they are, and a loader that checks them before use.

`calc_densities(..., save_matrices=True)` writes the bundle ``densities.npz``;
`load_densities` reads it back. The bundle holds

- ``metadata``: a JSON document (a 0-d string array) with the conventions, the method,
  the basis set, the Psi4 and propSAPT options, the versions, and the geometry with its
  fragments (`prop_sapt.fragments.DimerFragments`);
- ``delta_dm_<key>``: the density matrices, e.g. ``delta_dm_pol_A``;
- ``wfn_mon_A`` and ``wfn_mon_B``: the monomer wavefunctions, Psi4's
  `Wavefunction.to_file()` data with its arrays stored as entries ``wfn_mon_A/...``
  and everything else in a JSON manifest.

Nothing in the bundle is pickled: it is written and read with ``allow_pickle=False``.

Conventions (`CONVENTIONS`): the matrices are in the AO basis of the dimer basis set, in
Psi4 AO order, for one spin (the ``Da`` convention of the matrices `calc_densities`
returns). The total density of a closed-shell system is twice the stored matrix, see
`DensitySet.total`; the ``.cube`` files written by `calc_densities` hold the total density.

Classes
-------
DensitySet
    Density matrices and metadata read from a bundle, with the monomer wavefunctions.
ArtifactError
    Raised for a malformed bundle and for a bundle that does not match the expected
    geometry or basis set.

Functions
---------
save_densities
    Write density matrices of a dimer into a bundle.
load_densities
    Read a bundle, checking its conventions and, optionally, its geometry and basis set.
artifact_metadata
    The metadata of a dimer recorded in the bundle.
"""

import datetime
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import psi4

from ._version import __version__
from .checks import check_symmetric
from .fragments import DimerFragments, parse_fragments
from .options import get_propsapt_options

#: Filename of the bundle written by `calc_densities`.
DENSITIES_FILENAME = "densities.npz"
#: Value of the ``format`` metadata entry.
DENSITIES_FORMAT = "propsapt-densities"
#: Version of the bundle layout and metadata, raised on incompatible changes.
FORMAT_VERSION = 1
#: Conventions of the stored matrices, recorded in and checked on every bundle.
CONVENTIONS = {
    "orbital_basis": "AO",
    "ao_order": "psi4",
    "spin": "alpha",
    "total_density_factor": 2,
}

# entry names inside the bundle
_METADATA_ENTRY = "metadata"
_MATRIX_PREFIX = "delta_dm_"
_WAVEFUNCTION_ENTRY = "wfn_mon_{}"

# tags of the JSON manifest of a flattened tree
_ARRAY_TAG = "__array__"
_TUPLE_TAG = "__tuple__"


class ArtifactError(ValueError):
    """Raised when a bundle is malformed, or does not match what it is checked against."""


@dataclass(frozen=True, eq=False)
class DensitySet:
    """
    Density matrices read from a bundle by `load_densities`, with their metadata.
    """

    #: Density matrices by key (e.g. "pol_A", "pol_B", "pol"): AO basis, Psi4 AO order,
    #: one spin, as `calc_densities` returns them. See `total` for the total density.
    matrices: dict[str, np.ndarray]
    #: The metadata of the bundle, see `prop_sapt.artifacts`.
    metadata: dict
    #: Geometry and fragments of the dimer.
    fragments: DimerFragments
    #: Path of the bundle.
    path: Path

    @property
    def components(self) -> tuple[str, ...]:
        """The density components in the bundle, e.g. "pol", "exch", ..."""

        return tuple(self.metadata["components"])

    def total(self, key: str) -> np.ndarray:
        """
        Total (both spins) density matrix of `key`, twice the stored matrix.

        Args:
            key (str): Matrix key, e.g. "pol_A".

        Returns:
            np.ndarray: The total density matrix.
        """

        return self.metadata["total_density_factor"] * self.matrices[key]

    def wavefunction_data(self, monomer: str) -> dict:
        """
        Monomer wavefunction data as written by Psi4's `Wavefunction.to_file()`, read
        from the bundle without building any Psi4 object.

        Args:
            monomer (str): "A" or "B".

        Returns:
            dict: e.g. ``data["matrix"]["Da"]`` or ``data["vector"]["epsilon_a"]``.
        """

        if monomer not in self.metadata["wavefunctions"]:
            raise KeyError(
                f"No wavefunction of monomer {monomer!r} in '{self.path}', "
                f"it has {self.metadata['wavefunctions']}."
            )

        with np.load(self.path, allow_pickle=False) as npz:
            manifest = json.loads(npz[_WAVEFUNCTION_ENTRY.format(monomer)].item())
            return _unflatten_tree(manifest, npz)

    def wavefunction(self, monomer: str) -> psi4.core.Wavefunction:
        """
        Monomer wavefunction rebuilt with Psi4's `Wavefunction.from_file`.

        Psi4 builds the orbital basis set by its name, so a basis set given by per-element
        overrides or read from a basis set directory has to be available to Psi4 in the
        current session; `wavefunction_data` needs no basis set.

        Args:
            monomer (str): "A" or "B".

        Raises:
            RuntimeError: If Psi4 cannot rebuild the wavefunction, e.g. its basis set.

        Returns:
            psi4.core.Wavefunction: The monomer wavefunction.
        """

        data = self.wavefunction_data(monomer)
        try:
            return psi4.core.Wavefunction.from_file(data)
        except Exception as err:  # Psi4 and QCDB raise their own exception types
            raise RuntimeError(
                f"Psi4 could not rebuild the wavefunction of monomer {monomer} from "
                f"'{self.path}' (orbital basis {data['string']['basisname']!r}, "
                f"{self.metadata['basis']['blend']!r}): {err}. A basis set from "
                "per-element overrides or a basis set directory has to be made available "
                "to Psi4 first; `wavefunction_data` needs no basis set."
            ) from err

    def check(
        self,
        dimer=None,
        geometry: "DimerFragments | str | os.PathLike | None" = None,
        basisset: psi4.core.BasisSet | None = None,
        atol: float = 1.0e-6,
    ):
        """
        Check that the bundle belongs to a dimer, a geometry and/or a basis set.

        Args:
            dimer (Dimer | None, optional): Check the geometry, fragments and basis set
                against this dimer. Defaults to None.
            geometry (DimerFragments | str | os.PathLike | None, optional): Check the
                geometry and fragments against these fragments, a geometry string or a
                geometry file (see `prop_sapt.fragments.parse_fragments`).
                Defaults to None.
            basisset (psi4.core.BasisSet | None, optional): Check the basis set against
                this one. Defaults to None.
            atol (float, optional): Largest coordinate difference allowed, in bohr.
                Defaults to 1e-6.

        Raises:
            ArtifactError: Naming the first difference found.
        """

        where = f"Density bundle '{self.path}'"

        expected = []
        if dimer is not None:
            expected.append(("the dimer", dimer.fragments))
        if geometry is not None:
            if not isinstance(geometry, DimerFragments):
                geometry = parse_fragments(geometry)
            expected.append(("the given geometry", geometry))

        for name, fragments in expected:
            try:
                self.fragments.check_same_system(fragments, atol=atol)
            except ValueError as err:
                raise ArtifactError(
                    f"{where} does not match {name} (bundle vs expected): {err}."
                ) from err

        if dimer is not None:
            _check_basis(self.metadata["basis"], dimer.dimer_wfn.basisset(), where)
        if basisset is not None:
            _check_basis(self.metadata["basis"], basisset, where)


def artifact_metadata(dimer) -> dict:
    """
    Metadata of a dimer recorded in a bundle: format, versions, conventions, method,
    basis set, options, geometry and fragments. The basis sets and the Psi4 options are
    those the dimer was built with, whatever the options are now.

    Args:
        dimer (Dimer): The dimer.

    Returns:
        dict: JSON-serialisable metadata, without the entries describing the stored
            matrices, which `save_densities` adds.
    """

    return {
        "format": DENSITIES_FORMAT,
        "format_version": FORMAT_VERSION,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "prop_sapt_version": __version__,
        "psi4_version": psi4.__version__,
        **CONVENTIONS,
        "method": {
            "reference": dimer.reference,
            "functional": dimer.functional,
            "grac_A": dimer.grac_A,
            "grac_B": dimer.grac_B,
        },
        "basis": _basis_fingerprint(dimer.dimer_wfn.basisset()),
        "fitting_bases": {
            option: dimer.fitting_bases[option]
            for option in ("DF_BASIS_SCF", "DF_BASIS_SAPT")
        },
        "psi4_options": dimer.psi4_options,
        "propsapt_options": get_propsapt_options(),
        "geometry": dimer.fragments.to_dict(),
        "geometry_input": dimer.geometry,
    }


def save_densities(
    path: str | os.PathLike,
    matrices: Mapping[str, np.ndarray],
    dimer,
    metadata: Mapping | None = None,
    wavefunctions: bool = True,
) -> Path:
    """
    Write density matrices of a dimer into a bundle.

    Args:
        path (str | os.PathLike): Path of the bundle; ".npz" is appended if missing.
        matrices (Mapping[str, np.ndarray]): AO density matrices of the dimer basis set
            by key, one spin (the ``Da`` convention). Keys ending in "_A" or "_B" are
            monomer components, see `DensitySet.components`.
        dimer (Dimer): The dimer the matrices belong to.
        metadata (Mapping | None, optional): Additional top-level metadata entries; they
            cannot replace the entries written here. Defaults to None.
        wavefunctions (bool, optional): Also store the monomer wavefunctions.
            Defaults to True.

    Raises:
        ValueError: If a matrix is not (nbf, nbf), or `metadata` repeats an entry.

    Warns:
        InvariantWarning: If a matrix is not symmetric within the `check_symmetry_atol`
            option, see `prop_sapt.checks.check_symmetric`; it is saved anyway.

    Returns:
        Path: Path of the written bundle.
    """

    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_name(path.name + ".npz")

    record = artifact_metadata(dimer)
    nbf = record["basis"]["nbf"]

    if not matrices:
        raise ValueError("No density matrices to save!")

    arrays = {}
    checked = {}
    for key, matrix in matrices.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Matrix keys should be non-empty strings, got {key!r}!")
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (nbf, nbf):
            raise ValueError(
                f"Density matrix '{key}' has shape {matrix.shape}, AO matrices of the "
                f"dimer basis set are ({nbf}, {nbf})!"
            )
        arrays[_MATRIX_PREFIX + key] = matrix
        checked[key] = matrix

    check_symmetric(checked, where=f"bundle '{path}'")

    record["matrices"] = list(matrices)
    record["components"] = _components(matrices)
    record["wavefunctions"] = []

    if wavefunctions:
        for monomer, wfn in (("A", dimer.wfnA), ("B", dimer.wfnB)):
            entry = _WAVEFUNCTION_ENTRY.format(monomer)
            manifest, wfn_arrays = _flatten_tree(wfn.to_file(), entry)
            arrays[entry] = np.array(json.dumps(manifest))
            arrays.update(wfn_arrays)
            record["wavefunctions"].append(monomer)

    if metadata:
        repeated = sorted(set(metadata) & set(record))
        if repeated:
            raise ValueError(f"Metadata entries {repeated} are written by propSAPT!")
        record.update(metadata)

    arrays[_METADATA_ENTRY] = np.array(json.dumps(record, default=str))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

    return path


def load_densities(
    path: str | os.PathLike,
    dimer=None,
    *,
    geometry: "DimerFragments | str | os.PathLike | None" = None,
    basisset: psi4.core.BasisSet | None = None,
    atol: float = 1.0e-6,
) -> DensitySet:
    """
    Read a density bundle written by `calc_densities` or `save_densities`.

    The bundle is always checked for its format and conventions and for the shapes of its
    matrices. Give `dimer`, `geometry` and/or `basisset` to also check that it belongs to
    them, see `DensitySet.check`.

    Args:
        path (str | os.PathLike): The bundle, or a directory holding ``densities.npz``.
        dimer (Dimer | None, optional): Check geometry, fragments and basis set against
            this dimer. Defaults to None.
        geometry (DimerFragments | str | os.PathLike | None, optional): Check geometry and
            fragments against these fragments, a geometry string or a geometry file.
            Defaults to None.
        basisset (psi4.core.BasisSet | None, optional): Check the basis set against this
            one. Defaults to None.
        atol (float, optional): Largest coordinate difference allowed, in bohr.
            Defaults to 1e-6.

    Raises:
        ArtifactError: If the bundle is malformed, uses unknown conventions, or does not
            match `dimer`, `geometry` or `basisset`.

    Returns:
        DensitySet: Matrices, metadata and fragments of the bundle.
    """

    path = Path(path)
    if path.is_dir():
        path = path / DENSITIES_FILENAME
    where = f"Density bundle '{path}'"

    try:
        npz = np.load(path, allow_pickle=False)
    except ValueError as err:
        raise ArtifactError(f"{where} is not an .npz archive: {err}") from err
    if not isinstance(npz, np.lib.npyio.NpzFile):
        raise ArtifactError(f"{where} is not an .npz archive.")

    with npz:
        metadata = _read_metadata(npz, where)

        stored = [
            name[len(_MATRIX_PREFIX) :]
            for name in npz.files
            if name.startswith(_MATRIX_PREFIX)
        ]
        listed = metadata["matrices"]
        missing = [key for key in listed if key not in stored]
        unlisted = [key for key in stored if key not in listed]
        if missing or unlisted:
            raise ArtifactError(
                f"{where} lists matrices {listed} but stores {stored}: "
                f"missing {missing}, unlisted {unlisted}."
            )

        nbf = metadata["basis"]["nbf"]
        matrices = {}
        for key in listed:
            matrix = _read_entry(npz, _MATRIX_PREFIX + key, where)
            if matrix.shape != (nbf, nbf):
                raise ArtifactError(
                    f"{where}: matrix '{key}' has shape {matrix.shape}, "
                    f"the basis set has {nbf} functions."
                )
            matrices[key] = matrix

        for monomer in metadata["wavefunctions"]:
            entry = _WAVEFUNCTION_ENTRY.format(monomer)
            if entry not in npz.files:
                raise ArtifactError(
                    f"{where} lists wavefunction {monomer} but has no '{entry}'."
                )

    try:
        fragments = DimerFragments.from_dict(metadata["geometry"])
    except ValueError as err:
        raise ArtifactError(f"{where} has invalid geometry metadata: {err}") from err

    densities = DensitySet(
        matrices=matrices, metadata=metadata, fragments=fragments, path=path
    )
    densities.check(dimer=dimer, geometry=geometry, basisset=basisset, atol=atol)

    return densities


def _read_entry(npz, name: str, where: str) -> np.ndarray:
    """Reads one entry of an open bundle, refusing pickled data."""

    try:
        return npz[name]
    except ValueError as err:
        raise ArtifactError(f"{where}: entry '{name}' cannot be read: {err}") from err


def _read_metadata(npz, where: str) -> dict:
    """Reads and checks the metadata entry of an open bundle."""

    if _METADATA_ENTRY not in npz.files:
        raise ArtifactError(f"{where} is not a propSAPT density bundle: no metadata.")

    entry = _read_entry(npz, _METADATA_ENTRY, where)
    if entry.ndim != 0 or entry.dtype.kind != "U":
        raise ArtifactError(f"{where} has a metadata entry that is not a JSON string.")
    try:
        metadata = json.loads(entry.item())
    except json.JSONDecodeError as err:
        raise ArtifactError(f"{where} has invalid metadata JSON: {err}") from err
    if not isinstance(metadata, dict):
        raise ArtifactError(f"{where} has metadata that is not a JSON object.")

    if metadata.get("format") != DENSITIES_FORMAT:
        raise ArtifactError(
            f"{where} is not a propSAPT density bundle (format {metadata.get('format')!r})."
        )

    version = metadata.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ArtifactError(f"{where} has an invalid format version {version!r}.")
    if version > FORMAT_VERSION:
        raise ArtifactError(
            f"{where} was written in format version {version} by propSAPT "
            f"{metadata.get('prop_sapt_version')}; propSAPT {__version__} reads format "
            f"versions up to {FORMAT_VERSION}."
        )

    for key, expected in CONVENTIONS.items():
        if metadata.get(key) != expected:
            raise ArtifactError(
                f"{where} uses an unknown convention {key}={metadata.get(key)!r}, "
                f"expected {expected!r}."
            )

    for key in ("matrices", "components", "basis", "geometry", "wavefunctions"):
        if key not in metadata:
            raise ArtifactError(f"{where} has no '{key}' metadata entry.")

    return metadata


def _basis_fingerprint(basisset: psi4.core.BasisSet) -> dict:
    """
    Name, size and shell structure of a basis set, with a SHA-256 digest of its shells
    (centers, angular momenta, exponents and coefficients).
    """

    digest = hashlib.sha256()
    digest.update(f"puream={int(basisset.has_puream())}".encode())

    shells = []
    for index in range(basisset.nshell()):
        shell = basisset.shell(index)
        center = basisset.shell_to_center(index)
        nprimitive = shell.nprimitive
        shells.append([center, shell.am, nprimitive])

        digest.update(np.array([center, shell.am, nprimitive], dtype="<i8").tobytes())
        digest.update(
            np.array(
                [shell.exp(k) for k in range(nprimitive)]
                + [shell.original_coef(k) for k in range(nprimitive)],
                dtype="<f8",
            ).tobytes()
        )

    return {
        "name": basisset.name(),
        "blend": basisset.blend(),
        "puream": bool(basisset.has_puream()),
        "nbf": basisset.nbf(),
        "shells": shells,
        "sha256": digest.hexdigest(),
    }


def _check_basis(recorded: Mapping, basisset: psi4.core.BasisSet, where: str):
    """Raises `ArtifactError` if a basis set differs from the recorded fingerprint."""

    actual = _basis_fingerprint(basisset)

    for key in ("nbf", "puream", "shells", "sha256"):
        if recorded.get(key) == actual[key]:
            continue

        if key == "shells":
            mine, theirs = recorded.get("shells") or [], actual["shells"]
            index = next(
                (i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b),
                min(len(mine), len(theirs)),
            )
            detail = (
                f"shells differ from shell {index} on ({len(mine)} vs {len(theirs)} "
                "shells; [center, am, nprimitive])"
            )
        elif key == "sha256":
            detail = "shell exponents or coefficients differ"
        else:
            detail = f"{key} differs: {recorded.get(key)!r} vs {actual[key]!r}"

        raise ArtifactError(
            f"{where} does not match the basis set (bundle {recorded.get('blend')!r} vs "
            f"expected {actual['blend']!r}): {detail}."
        )


def _components(keys) -> list[str]:
    """Density components of matrix keys: the keys without a "_A" or "_B" suffix."""

    components = []
    for key in keys:
        component = key[:-2] if key.endswith(("_A", "_B")) else key
        if component not in components:
            components.append(component)

    return components


def _flatten_tree(tree, prefix: str) -> tuple[object, dict[str, np.ndarray]]:
    """
    Split a tree of dicts, lists and tuples into a JSON-serialisable manifest and its
    arrays, named by their path below `prefix`, so nothing has to be pickled.

    Leaves can be arrays of booleans, numbers or strings, NumPy scalars, None, bool, int,
    float and str; anything else raises `TypeError`.
    """

    arrays = {}

    def encode(value, path):
        if isinstance(value, Mapping):
            encoded = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"Keys should be strings, '{path}' has {key!r}!")
                if key in (_ARRAY_TAG, _TUPLE_TAG):
                    raise TypeError(f"Key '{key}' in '{path}' is reserved!")
                encoded[key] = encode(item, f"{path}/{key}")
            return encoded

        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biufcUS":
                raise TypeError(
                    f"'{path}' is an array of {value.dtype}, which cannot be stored "
                    "without pickling!"
                )
            if path in arrays:
                raise TypeError(f"Two entries are stored as '{path}'!")
            arrays[path] = value
            return {_ARRAY_TAG: path}

        if isinstance(value, tuple):
            return {_TUPLE_TAG: [encode(v, f"{path}/{i}") for i, v in enumerate(value)]}
        if isinstance(value, list):
            return [encode(v, f"{path}/{i}") for i, v in enumerate(value)]

        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        raise TypeError(
            f"'{path}' holds a {type(value).__name__}, which cannot be stored without "
            "pickling!"
        )

    return encode(tree, prefix), arrays


def _unflatten_tree(encoded, arrays: Mapping[str, np.ndarray]):
    """Rebuild a tree flattened by `_flatten_tree` from its manifest and arrays."""

    if isinstance(encoded, dict):
        if set(encoded) == {_ARRAY_TAG}:
            return arrays[encoded[_ARRAY_TAG]]
        if set(encoded) == {_TUPLE_TAG}:
            return tuple(_unflatten_tree(v, arrays) for v in encoded[_TUPLE_TAG])
        return {key: _unflatten_tree(value, arrays) for key, value in encoded.items()}

    if isinstance(encoded, list):
        return [_unflatten_tree(value, arrays) for value in encoded]

    return encoded
