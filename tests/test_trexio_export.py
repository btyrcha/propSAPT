"""Tests of the TREXIO export of the density changes (`prop_sapt.trexio_export`).

The files are read back with trexio and evaluated by `tests.trexio_spec`, which follows the
TREXIO specification and is independent of Psi4. The tests that write files need trexio
and skip without it.
"""

import importlib.util
import sys

import numpy as np
import psi4
import pytest

from prop_sapt import calc_densities, reset_propsapt_options
from prop_sapt.evaluate import evaluate_basis_functions
from prop_sapt.fragments import DimerFragments
from prop_sapt.natural_orbitals import NaturalOrbitals, density_natural_orbitals
from prop_sapt.trexio_export import (
    DEFAULT_CUTOFF,
    ENCODING,
    TrexioExportError,
    _system_arrays,
    _write_file,
    export_trexio,
    import_trexio,
)

from . import trexio_spec
from .he_h2 import OPTIONS

needs_trexio = pytest.mark.skipif(
    importlib.util.find_spec("trexio") is None, reason="trexio not installed"
)

WATER_HE = """
0 1
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
--
0 1
He  0.000000   0.000000   3.200000
symmetry c1
no_com
no_reorient
"""


@pytest.fixture(autouse=True, scope="module")
def restore_psi4_state():
    """Restores the global Psi4 and propSAPT state for the test modules after this one,
    as the tests here build molecules and basis sets of their own."""

    threads = psi4.core.get_num_threads()

    yield

    psi4.core.clean_options()
    psi4.set_options(OPTIONS)
    psi4.set_memory("2 GB")
    psi4.set_num_threads(threads)
    psi4.core.set_output_file("output.dat", True)
    reset_propsapt_options()


@pytest.fixture
def restore_puream():
    """A test setting the puream option leaves it to the basis sets again."""

    yield
    psi4.core.revoke_global_option_changed("PUREAM")


def _points(molecule, count=400, seed=7):
    rng = np.random.default_rng(seed)
    centers = np.asarray(molecule.geometry())
    return centers[rng.integers(0, len(centers), count)] + rng.normal(
        scale=0.9, size=(count, 3)
    )


def _write_aos(path, molecule, basisset):
    """A file whose orbitals are the AOs of `basisset`, one by one."""

    trexio = import_trexio()
    nbf = basisset.nbf()
    system = _system_arrays(DimerFragments.from_molecule(molecule), basisset)
    aos = NaturalOrbitals(np.ones(nbf), np.eye(nbf), np.full(nbf, "A"))
    _write_file(trexio, path, system, aos, "AOs")


@needs_trexio
@pytest.mark.parametrize(
    "basis, puream",
    [("cc-pvqz", None), ("cc-pvqz", False), ("6-31gss", None), ("sto-3g", None)],
)
def test_aos_are_psi4s(tmp_path, restore_puream, basis, puream):
    """The AOs written, evaluated as the TREXIO specification defines them, are Psi4's:
    spherical (up to g) and Cartesian shells, in Psi4's order and normalisation."""

    psi4.core.set_output_file("output.dat", True)
    if puream is not None:
        psi4.set_options({"puream": puream})
    molecule = psi4.geometry(WATER_HE)
    molecule.update_geometry()
    basisset = psi4.core.BasisSet.build(molecule, "ORBITAL", basis)
    path = tmp_path / "aos.h5"

    _write_aos(path, molecule, basisset)

    data = trexio_spec.read_file(path)
    assert data["cartesian"] == (0 if basisset.has_puream() else 1)
    points = _points(molecule)
    np.testing.assert_allclose(
        trexio_spec.orbital_values(data, points),
        evaluate_basis_functions(points, basisset),
        rtol=0,
        atol=1e-12,
    )


@needs_trexio
def test_nuclei_ghosts_and_ecps():
    """Ghost atoms are nuclei of charge 0 labelled "X"; an ECP reduces the charge."""

    psi4.core.set_output_file("output.dat", True)
    molecule = psi4.geometry("""
0 1
I  0.0 0.0 0.0
H  0.0 0.0 1.6
--
0 1
He 0.0 0.0 5.0
--
0 1
@Ne 0.0 3.0 0.0
symmetry c1
no_com
no_reorient
units bohr
""")
    molecule.update_geometry()
    basisset = psi4.core.BasisSet.build(molecule, "ORBITAL", "def2-svp")

    system = _system_arrays(DimerFragments.from_molecule(molecule), basisset)

    assert system.labels == ["I", "H", "He", "X"]
    np.testing.assert_array_equal(system.charges, [53 - 28, 1, 2, 0])
    np.testing.assert_allclose(system.coordinates, np.asarray(molecule.geometry()))


@needs_trexio
def test_shells_above_g_warn():
    """VIAMD reads shells up to g: a basis set with h shells warns."""

    psi4.core.set_output_file("output.dat", True)
    molecule = psi4.geometry("""
0 1
Ne 0.0 0.0 0.0
--
0 1
He 0.0 0.0 6.0
symmetry c1
no_com
no_reorient
units bohr
""")
    molecule.update_geometry()
    basisset = psi4.core.BasisSet.build(molecule, "ORBITAL", "cc-pv5z")

    with pytest.warns(UserWarning, match="VIAMD reads shells up to l = 4"):
        _system_arrays(DimerFragments.from_molecule(molecule), basisset)


def test_mixed_shells_are_refused():
    """A basis set of spherical and Cartesian shells cannot be written."""

    class Shell:
        def __init__(self, am, pure):
            self.am = am
            self._pure = pure

        def is_pure(self):
            return self._pure

    class Basis:
        shells = [Shell(0, True), Shell(2, True), Shell(2, False)]

        def nshell(self):
            return len(self.shells)

        def shell(self, index):
            return self.shells[index]

        def name(self):
            return "MIXED"

    with pytest.raises(TrexioExportError, match="mixes spherical and Cartesian"):
        _system_arrays(None, Basis())


# the He-H2 densities, computed once


@pytest.fixture(scope="module")
def he_h2_densities(prepare_dimer):
    return calc_densities(prepare_dimer)


@needs_trexio
def test_files_hold_the_densities(tmp_path, prepare_dimer, he_h2_densities):
    """Every file's alpha - beta density is the total density change (twice the one-spin
    matrix): as a matrix, and at points as the specification evaluates it."""

    paths = export_trexio(he_h2_densities, tmp_path, dimer=prepare_dimer, cutoff=0.0)

    assert [path.name for path in paths] == [
        f"delta_dm_{key}.h5" for key in he_h2_densities
    ]
    points = _points(prepare_dimer.get_psi4_molecule(), count=300)
    for key, path in zip(he_h2_densities, paths):
        data = trexio_spec.read_file(path)
        signs = np.where(data["mo_spin"] == 0, 1.0, -1.0)
        coefficients = data["mo_coefficient"].T
        rebuilt = (coefficients * (signs * data["mo_occupation"])) @ coefficients.T
        np.testing.assert_allclose(
            rebuilt, 2 * he_h2_densities[key], rtol=0, atol=1e-12, err_msg=key
        )
        if key in ("total", "pol_A", "Disp_B"):
            np.testing.assert_allclose(
                trexio_spec.difference_density(data, points),
                prepare_dimer.evaluate_density(2 * he_h2_densities[key], points),
                rtol=0,
                atol=1e-14,
                err_msg=key,
            )


@needs_trexio
def test_file_contents(tmp_path, prepare_dimer, he_h2_densities):
    """The orbitals of a file: alpha for the positive occupations by decreasing n, beta for
    the negative ones by decreasing |n|, without those up to the cutoff; the nuclei, the
    description and the orbital type."""

    (path,) = export_trexio(
        he_h2_densities, tmp_path, dimer=prepare_dimer, keys=["Ind"]
    )

    data = trexio_spec.read_file(path)
    orbitals = density_natural_orbitals(
        he_h2_densities, prepare_dimer, keys=["Ind"], cutoff=DEFAULT_CUTOFF
    )["Ind"]
    n = orbitals.occupations
    expected = 2 * np.concatenate([n[n > 0], -n[n < 0][::-1]])

    np.testing.assert_allclose(data["mo_occupation"], expected, rtol=0, atol=0)
    np.testing.assert_array_equal(
        data["mo_spin"], [0] * np.sum(n > 0) + [1] * np.sum(n < 0)
    )
    assert len(n) < 2 * prepare_dimer.nmo  # the cutoff dropped some
    assert data["mo_type"] == "Natural"
    assert "'Ind' of the dimer (A + B)" in data["description"]
    assert ENCODING in data["description"]
    assert list(data["label"]) == ["He", "H", "H"]
    np.testing.assert_array_equal(data["charge"], [2.0, 1.0, 1.0])


@needs_trexio
def test_export_needs_a_dimer():
    with pytest.raises(TypeError, match="`dimer` is needed"):
        export_trexio({"pol_A": np.zeros((2, 2))})


def test_missing_trexio_is_explained(monkeypatch):
    """Without trexio, writing a file says how to install it."""

    monkeypatch.setitem(sys.modules, "trexio", None)  # makes the import fail

    with pytest.raises(TrexioExportError, match="pip install trexio"):
        import_trexio()
