"""
Fragment structure of a dimer geometry: the atoms of monomers A and B and of any extra
fragments, with their charges and multiplicities.

Fragments follow the `Dimer` convention: the first fragment of the geometry (the atoms
before the first ``--``) is monomer A, the second is monomer B, and every further fragment,
e.g. midbond functions, only adds basis functions to both monomer calculations.

Coordinates are in bohr, in the frame the `Dimer` uses: the input orientation, moved to
the center of mass unless the geometry says ``no_com``.

Classes
-------
Fragment
    Atoms, charge and multiplicity of one fragment.
DimerFragments
    Atoms of a dimer geometry (elements, ghost flags, coordinates) and their fragments.

Functions
---------
parse_fragments
    Fragment structure of a Psi4 geometry string or a ``.p4m`` file, without running any
    calculation.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import psi4

# labels of the first two fragments, the monomers; further fragments are "X1", "X2", ...
MONOMER_LABELS = ("A", "B")


@dataclass(frozen=True)
class Fragment:
    """
    One fragment of a dimer geometry.
    """

    #: "A" and "B" for the monomers, "X1", "X2", ... for the extra fragments.
    label: str
    #: Indices of the fragment's atoms in the dimer geometry, counted from 0.
    atoms: tuple[int, ...]
    #: Charge of the fragment.
    charge: int
    #: Spin multiplicity of the fragment.
    multiplicity: int

    def __post_init__(self):
        object.__setattr__(self, "atoms", tuple(int(atom) for atom in self.atoms))
        object.__setattr__(self, "charge", int(self.charge))
        object.__setattr__(self, "multiplicity", int(self.multiplicity))


@dataclass(frozen=True, eq=False)
class DimerFragments:
    """
    Atoms of a dimer geometry and the fragments they belong to.

    Build it with `parse_fragments` from a geometry, or with `from_molecule` from a Psi4
    molecule; `Dimer.fragments` holds the one of a dimer.
    """

    #: Element symbol of each atom, e.g. "He"; ghost atoms keep their element.
    symbols: tuple[str, ...]
    #: Atomic number of each atom, ghost atoms included.
    atomic_numbers: tuple[int, ...]
    #: False for ghost atoms (``Gh(X)`` or ``@X``), which carry basis functions only.
    real: tuple[bool, ...]
    #: Coordinates in bohr, shape (natom, 3).
    coordinates: np.ndarray
    #: Monomer A, monomer B, then the extra fragments, in the order of the geometry.
    fragments: tuple[Fragment, ...]

    def __post_init__(self):
        object.__setattr__(self, "symbols", tuple(str(s) for s in self.symbols))
        object.__setattr__(
            self, "atomic_numbers", tuple(int(z) for z in self.atomic_numbers)
        )
        object.__setattr__(self, "real", tuple(bool(r) for r in self.real))
        object.__setattr__(self, "fragments", tuple(self.fragments))

        coordinates = np.array(self.coordinates, dtype=float)
        coordinates.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)

        natom = len(self.symbols)
        if len(self.atomic_numbers) != natom or len(self.real) != natom:
            raise ValueError(
                f"Got {natom} symbols, {len(self.atomic_numbers)} atomic numbers and "
                f"{len(self.real)} ghost flags, one of each per atom is required!"
            )
        if coordinates.shape != (natom, 3):
            raise ValueError(
                f"Coordinates should have shape ({natom}, 3), had {coordinates.shape}!"
            )

        if len(self.fragments) < 2:
            raise ValueError(
                f"Found {len(self.fragments)} fragments, must be at least 2."
            )
        labels = [fragment.label for fragment in self.fragments]
        expected = _fragment_labels(len(self.fragments))
        if labels != expected:
            raise ValueError(f"Fragment labels should be {expected}, were {labels}!")

        owners = [None] * natom
        for fragment in self.fragments:
            for atom in fragment.atoms:
                if not 0 <= atom < natom:
                    raise ValueError(
                        f"Fragment {fragment.label} has atom {atom}, "
                        f"the geometry has {natom} atoms!"
                    )
                if owners[atom] is not None:
                    raise ValueError(
                        f"Atom {atom} is in fragments {owners[atom]} and {fragment.label}!"
                    )
                owners[atom] = fragment.label
        unassigned = [atom for atom, owner in enumerate(owners) if owner is None]
        if unassigned:
            raise ValueError(f"Atoms {unassigned} belong to no fragment!")

    def __eq__(self, other):
        if not isinstance(other, DimerFragments):
            return NotImplemented
        return (
            self.symbols == other.symbols
            and self.atomic_numbers == other.atomic_numbers
            and self.real == other.real
            and self.fragments == other.fragments
            and np.array_equal(self.coordinates, other.coordinates)
        )

    __hash__ = None

    @property
    def natom(self) -> int:
        """Number of atoms, ghost atoms included."""

        return len(self.symbols)

    @property
    def A(self) -> Fragment:
        """Monomer A, the first fragment."""

        return self.fragments[0]

    @property
    def B(self) -> Fragment:
        """Monomer B, the second fragment."""

        return self.fragments[1]

    @property
    def extra(self) -> tuple[Fragment, ...]:
        """The fragments after the monomers, which carry basis functions only."""

        return self.fragments[2:]

    def fragment_of(self, atom: int) -> str:
        """
        Label of the fragment holding `atom`.

        Args:
            atom (int): Atom index in the dimer geometry, counted from 0.

        Returns:
            str: "A", "B", or "X1", "X2", ... for the extra fragments.
        """

        for fragment in self.fragments:
            if atom in fragment.atoms:
                return fragment.label

        raise IndexError(f"Atom {atom} is not in the geometry of {self.natom} atoms!")

    def check_same_system(self, other: "DimerFragments", atol: float = 1.0e-6):
        """
        Check that `other` describes the same atoms, fragments and coordinates.

        Args:
            other (DimerFragments): Fragments to compare with.
            atol (float, optional): Largest coordinate difference allowed, in bohr.
                Defaults to 1e-6.

        Raises:
            ValueError: Naming the first difference found, values of `self` first.
        """

        if self.natom != other.natom:
            raise ValueError(f"numbers of atoms differ: {self.natom} vs {other.natom}")

        for name in ("atomic_numbers", "real"):
            mine, theirs = getattr(self, name), getattr(other, name)
            for atom, (a, b) in enumerate(zip(mine, theirs)):
                if a != b:
                    what = (
                        "atomic numbers" if name == "atomic_numbers" else "ghost flags"
                    )
                    raise ValueError(
                        f"{what} differ at atom {atom}: "
                        f"{_atom_desc(self, atom)} vs {_atom_desc(other, atom)}"
                    )

        if len(self.fragments) != len(other.fragments):
            raise ValueError(
                f"numbers of fragments differ: {len(self.fragments)} vs "
                f"{len(other.fragments)}"
            )
        for mine, theirs in zip(self.fragments, other.fragments):
            if mine != theirs:
                raise ValueError(f"fragments {mine.label} differ: {mine} vs {theirs}")

        deviation = np.abs(self.coordinates - other.coordinates).max(axis=1)
        atom = int(np.argmax(deviation))
        if deviation[atom] > atol:
            raise ValueError(
                f"coordinates differ by up to {deviation[atom]:.3e} bohr "
                f"(atom {atom}, {self.symbols[atom]}), more than atol={atol:.1e}"
            )

    def to_dict(self) -> dict:
        """
        Return the fragments as a plain, JSON-serialisable dictionary.

        Returns:
            dict: Read back with `DimerFragments.from_dict`; coordinates are nested lists
                in bohr, which JSON stores without loss.
        """

        return {
            "units": "bohr",
            "symbols": list(self.symbols),
            "atomic_numbers": list(self.atomic_numbers),
            "real": list(self.real),
            "coordinates": self.coordinates.tolist(),
            "fragments": [
                {
                    "label": fragment.label,
                    "atoms": list(fragment.atoms),
                    "charge": fragment.charge,
                    "multiplicity": fragment.multiplicity,
                }
                for fragment in self.fragments
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "DimerFragments":
        """
        Build the fragments from a dictionary written by `DimerFragments.to_dict`.

        Args:
            data (Mapping): Dictionary from `to_dict`.

        Raises:
            ValueError: If an entry is missing or the entries are inconsistent.

        Returns:
            DimerFragments: The fragments.
        """

        expected_keys = {
            "units",
            "symbols",
            "atomic_numbers",
            "real",
            "coordinates",
            "fragments",
        }
        if not isinstance(data, Mapping) or set(data) != expected_keys:
            found = sorted(data) if isinstance(data, Mapping) else type(data).__name__
            raise ValueError(
                f"Fragment data should have the entries {sorted(expected_keys)}, "
                f"had {found}!"
            )
        if data["units"] != "bohr":
            raise ValueError(f"Coordinates should be in bohr, were in {data['units']}!")

        try:
            fragments = tuple(Fragment(**fragment) for fragment in data["fragments"])
        except TypeError as err:
            raise ValueError(f"Invalid fragment entry: {err}") from err

        return cls(
            symbols=data["symbols"],
            atomic_numbers=data["atomic_numbers"],
            real=data["real"],
            coordinates=data["coordinates"],
            fragments=fragments,
        )

    @classmethod
    def from_molecule(cls, molecule: psi4.core.Molecule) -> "DimerFragments":
        """
        Read the fragments of a Psi4 molecule, after updating its geometry.

        For the frame of a `Dimer`, fix the orientation and the center of mass of a
        molecule not yet updated, as `parse_fragments` does.

        Args:
            molecule (psi4.core.Molecule): Molecule with at least two fragments.

        Raises:
            ValueError: If the molecule has fewer than two fragments.

        Returns:
            DimerFragments: The fragments.
        """

        molecule.update_geometry()

        ranges = molecule.get_fragments()
        if len(ranges) < 2:
            raise ValueError(f"Found {len(ranges):d} fragments, must be at least 2.")

        fragments = tuple(
            Fragment(
                label=label,
                atoms=tuple(range(start, end)),
                charge=charge,
                multiplicity=multiplicity,
            )
            for label, (start, end), charge, multiplicity in zip(
                _fragment_labels(len(ranges)),
                ranges,
                molecule.get_fragment_charges(),
                molecule.get_fragment_multiplicities(),
            )
        )

        natom = molecule.natom()
        return cls(
            symbols=[molecule.symbol(atom).capitalize() for atom in range(natom)],
            atomic_numbers=[molecule.true_atomic_number(atom) for atom in range(natom)],
            real=[molecule.Z(atom) != 0.0 for atom in range(natom)],
            coordinates=np.asarray(molecule.geometry()),
            fragments=fragments,
        )


def parse_fragments(geometry: str | os.PathLike) -> DimerFragments:
    """
    Read the fragment structure of a dimer geometry without running any calculation.

    The coordinates are those a `Dimer` built from the same geometry uses.

    Args:
        geometry (str | os.PathLike): Geometry in Psi4 format, or the path of a file
            holding one (e.g. a ``.p4m`` file). A string without line breaks that names
            an existing file is read as a path.

    Raises:
        ValueError: If the geometry has fewer than two fragments.

    Returns:
        DimerFragments: The fragments.
    """

    if isinstance(geometry, os.PathLike) or (
        "\n" not in geometry and Path(geometry).is_file()
    ):
        geometry = Path(geometry).read_text(encoding="utf-8")

    molecule = psi4.core.Molecule.from_string(geometry)

    # the frame of `helper_SAPT.__init__`: input orientation (centered by Psi4 unless
    # `no_com`), symmetry not imposed
    molecule.reset_point_group("c1")
    molecule.fix_orientation(True)
    molecule.fix_com(True)
    molecule.update_geometry()

    return DimerFragments.from_molecule(molecule)


def _fragment_labels(nfragments: int) -> list[str]:
    """Labels of `nfragments` fragments: "A", "B", "X1", "X2", ..."""

    return list(MONOMER_LABELS) + [f"X{i}" for i in range(1, nfragments - 1)]


def _atom_desc(fragments: DimerFragments, atom: int) -> str:
    """Element and ghost flag of an atom, e.g. "He" or "Gh(He)"."""

    symbol = fragments.symbols[atom]
    return symbol if fragments.real[atom] else f"Gh({symbol})"
