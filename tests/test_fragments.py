"""Tests of the fragment structure of dimer geometries (`prop_sapt.fragments`)."""

import json

import numpy as np
import psi4
import pytest

from .he_h2 import HE_H2
from prop_sapt.fragments import DimerFragments, Fragment, parse_fragments

# Angstrom, off the origin, not centered, with a charged monomer and a ghost fragment
LI_WATER_MIDBOND = """
1 1
Li  1.0  0.3  0.2
--
0 1
O   3.0  0.1  0.0
H   3.5  0.9  0.0
H   3.5 -0.7  0.1
--
0 1
Gh(He)  2.0  0.0  0.0
"""


def test_parse_he_h2():
    fragments = parse_fragments(HE_H2)

    assert fragments.symbols == ("He", "H", "H")
    assert fragments.atomic_numbers == (2, 1, 1)
    assert fragments.real == (True, True, True)
    assert fragments.natom == 3
    assert fragments.A == Fragment("A", (0,), 0, 1)
    assert fragments.B == Fragment("B", (1, 2), 0, 1)
    assert fragments.extra == ()
    np.testing.assert_array_equal(
        fragments.coordinates,
        [[-2.5, 0.0, 0.0], [1.7755, 0.0, 0.0], [3.2245, 0.0, 0.0]],
    )


def test_parse_from_file(tmp_path):
    path = tmp_path / "he-h2.p4m"
    path.write_text(HE_H2, encoding="utf-8")

    from_string = parse_fragments(HE_H2)

    assert parse_fragments(path) == from_string
    assert parse_fragments(str(path)) == from_string


def test_parse_keeps_dimer_frame_with_extra_fragment():
    """Angstrom input off the origin gets the frame of `helper_SAPT.__init__`."""

    fragments = parse_fragments(LI_WATER_MIDBOND)

    molecule = psi4.geometry(LI_WATER_MIDBOND)
    molecule.reset_point_group("c1")
    molecule.fix_orientation(True)
    molecule.fix_com(True)
    molecule.update_geometry()

    np.testing.assert_array_equal(
        fragments.coordinates, np.asarray(molecule.geometry())
    )

    # the input orientation, translated to the center of mass
    input_bohr = (
        np.array(
            [
                [1.0, 0.3, 0.2],
                [3.0, 0.1, 0.0],
                [3.5, 0.9, 0.0],
                [3.5, -0.7, 0.1],
                [2.0, 0.0, 0.0],
            ]
        )
        / psi4.constants.bohr2angstroms
    )
    shift = fragments.coordinates - input_bohr
    np.testing.assert_allclose(
        shift, np.broadcast_to(shift[0], shift.shape), atol=1e-12
    )

    assert fragments.symbols == ("Li", "O", "H", "H", "He")
    assert fragments.atomic_numbers == (3, 8, 1, 1, 2)
    assert fragments.real == (True, True, True, True, False)
    assert fragments.A == Fragment("A", (0,), 1, 1)
    assert fragments.B == Fragment("B", (1, 2, 3), 0, 1)
    assert fragments.extra == (Fragment("X1", (4,), 0, 1),)
    assert [fragments.fragment_of(atom) for atom in range(5)] == [
        "A",
        "B",
        "B",
        "B",
        "X1",
    ]

    with pytest.raises(IndexError):
        fragments.fragment_of(5)


def test_dict_round_trip_is_exact():
    fragments = parse_fragments(LI_WATER_MIDBOND)

    restored = DimerFragments.from_dict(json.loads(json.dumps(fragments.to_dict())))

    assert restored == fragments
    assert not restored.coordinates.flags.writeable


def test_one_fragment_is_rejected():
    with pytest.raises(ValueError, match="Found 1 fragments"):
        parse_fragments("0 1\nHe 0.0 0.0 0.0\n")


def _modified(original, **changes):
    """`original` rebuilt from its dictionary with some entries replaced."""

    data = original.to_dict()
    data.update(changes)
    return DimerFragments.from_dict(data)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"atomic_numbers": [3, 8, 1, 1, 10]}, "atomic numbers differ at atom 4"),
        ({"real": [True, True, True, True, True]}, "ghost flags differ at atom 4"),
        (
            {
                "fragments": [
                    {"label": "A", "atoms": [0], "charge": 1, "multiplicity": 1},
                    {
                        "label": "B",
                        "atoms": [1, 2, 3, 4],
                        "charge": 0,
                        "multiplicity": 1,
                    },
                ]
            },
            "numbers of fragments differ",
        ),
        (
            {
                "fragments": [
                    {"label": "A", "atoms": [0], "charge": 0, "multiplicity": 1},
                    {"label": "B", "atoms": [1, 2, 3], "charge": 0, "multiplicity": 1},
                    {"label": "X1", "atoms": [4], "charge": 0, "multiplicity": 1},
                ]
            },
            "fragments A differ",
        ),
    ],
)
def test_check_same_system_finds_differences(changes, message):
    fragments = parse_fragments(LI_WATER_MIDBOND)

    with pytest.raises(ValueError, match=message):
        fragments.check_same_system(_modified(fragments, **changes))


def test_check_same_system_coordinate_tolerance():
    fragments = parse_fragments(LI_WATER_MIDBOND)
    coordinates = fragments.coordinates.copy()
    coordinates[2, 1] += 1.0e-4
    moved = _modified(fragments, coordinates=coordinates.tolist())

    fragments.check_same_system(moved, atol=1.0e-3)
    with pytest.raises(ValueError, match=r"atom 2, H"):
        fragments.check_same_system(moved)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"units": "angstrom"}, "bohr"),
        ({"real": [True]}, "one of each per atom"),
        ({"coordinates": [[0.0, 0.0, 0.0]] * 4}, r"shape \(5, 3\)"),
        (
            {
                "fragments": [
                    {"label": "A", "atoms": [0, 1], "charge": 1, "multiplicity": 1},
                    {"label": "B", "atoms": [1, 2, 3], "charge": 0, "multiplicity": 1},
                    {"label": "X1", "atoms": [4], "charge": 0, "multiplicity": 1},
                ]
            },
            "in fragments A and B",
        ),
        (
            {
                "fragments": [
                    {"label": "A", "atoms": [0], "charge": 1, "multiplicity": 1},
                    {"label": "B", "atoms": [1, 2, 3], "charge": 0, "multiplicity": 1},
                ]
            },
            r"Atoms \[4\] belong to no fragment",
        ),
        (
            {
                "fragments": [
                    {"label": "B", "atoms": [0], "charge": 1, "multiplicity": 1},
                    {
                        "label": "A",
                        "atoms": [1, 2, 3, 4],
                        "charge": 0,
                        "multiplicity": 1,
                    },
                ]
            },
            "labels",
        ),
    ],
)
def test_inconsistent_data_is_rejected(changes, message):
    fragments = parse_fragments(LI_WATER_MIDBOND)

    with pytest.raises(ValueError, match=message):
        _modified(fragments, **changes)


def test_dimer_fragments_match_parsed_geometry(prepare_dimer):
    assert prepare_dimer.fragments == parse_fragments(HE_H2)
    assert prepare_dimer.fragments.A.atoms == (0,)
    assert prepare_dimer.fragments.B.atoms == (1, 2)


def test_dimer_records_dft_settings(prepare_dimer):
    assert prepare_dimer.reference == "RHF"
    assert prepare_dimer.functional is None
    assert prepare_dimer.grac_A is None
    assert prepare_dimer.grac_B is None
