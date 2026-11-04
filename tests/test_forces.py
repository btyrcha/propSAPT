import numpy as np
import pytest
import psi4

from prop_sapt.forces import calc_force_nuc_nuc, calc_force_at_nuc


@pytest.fixture(scope="session")
def psi4_ready():
    psi4.set_memory("4 GB")
    psi4.core.set_output_file("output_forces_test.dat", False)


@pytest.fixture(scope="session")
def h2(psi4_ready):
    mol = psi4.geometry("""
symmetry c1
no_com
no_reorient
units bohr
H  0.0  0.0  0.0
H  1.6  0.0  0.0
""")
    mol.update_geometry()
    return mol


@pytest.fixture(scope="session")
def water(psi4_ready):
    mol = psi4.geometry("""
symmetry c1
no_com
no_reorient
units bohr
O   0.000000   0.000000   0.000000
H   1.809000   0.000000   1.205000
H  -1.809000   0.000000   1.205000
""")
    mol.update_geometry()
    return mol


@pytest.fixture(scope="session")
def h2_wfn(psi4_ready, h2):
    psi4.set_options(
        {
            "basis": "aug-cc-pv5z",
            "scf_type": "direct",
        }
    )
    _, wfn = psi4.energy("scf", molecule=h2, return_wfn=True)
    return wfn


@pytest.fixture(scope="session")
def water_wfn(psi4_ready, water):
    psi4.set_options(
        {
            "basis": "cc-pvdz",
            "scf_type": "direct",
        }
    )
    _, wfn = psi4.energy("scf", molecule=water, return_wfn=True)
    return wfn


def test_nuc_nuc_h2_analytical(h2):
    """Test 1: nuc-nuc x-force on H2 equals -1/R^2; Newton's 3rd law holds."""
    geom, _, _, _, _ = h2.to_arrays()
    R = np.linalg.vector_norm(geom[1])
    f0 = calc_force_nuc_nuc(h2, atidx=0)
    f1 = calc_force_nuc_nuc(h2, atidx=1)

    assert f0[0] == pytest.approx(-1.0 / R**2)
    assert f0[1] == pytest.approx(0.0, abs=1e-14)
    assert f0[2] == pytest.approx(0.0, abs=1e-14)
    assert np.allclose(f0 + f1, 0.0)


def test_nuc_nuc_distance_scaling():
    """Test 2: nuc-nuc force on H2 atom 0 equals -1/R^2 across a range of bond lengths."""
    for d in np.linspace(0.5, 5.0, 10):
        mol = psi4.geometry(f"""
symmetry c1
no_com
no_reorient
units bohr
H  0.0  0.0  0.0
H  {d}  0.0  0.0
""")
        mol.update_geometry()
        assert calc_force_nuc_nuc(mol, 0)[0] == pytest.approx(-1.0 / d**2)


def test_total_force_vs_psi4_gradient(h2, h2_wfn):
    """Test 3: total force from calc_force_at_nuc agrees with Psi4 analytical gradient."""
    rho = h2_wfn.Da().to_array() + h2_wfn.Db().to_array()
    our_forces = np.array(
        [
            calc_force_at_nuc(h2, rho, atidx=i, basisset=h2_wfn.basisset())
            for i in range(h2.natom())
        ]
    )
    psi4_forces = -np.array(psi4.gradient("scf", molecule=h2, ref_wfn=h2_wfn))
    difference = our_forces - psi4_forces
    assert np.allclose(difference, [np.zeros(3), np.zeros(3)], atol=1e-3)


def test_nuc_nuc_h2o_newton(water):
    """Test 4: sum of nuc-nuc forces over all atoms is zero for H2O."""
    forces = np.array([calc_force_nuc_nuc(water, i) for i in range(water.natom())])
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-12)
