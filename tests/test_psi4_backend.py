"""The Psi4 backend: the data it gives the dimer, and the dimer's Psi4 attributes."""

from types import SimpleNamespace

import numpy as np
import psi4
import pytest

from prop_sapt import Dimer
from prop_sapt.backends import MonomerReference
from prop_sapt.backends.psi4_backend import Psi4Backend
from prop_sapt.sinfinity import sinfinity

from .he_h2 import OPTIONS
from .test_cpscf_type import DF_BASIS
from .test_dimer import HE_H


def test_backend_of_the_dimer(prepare_dimer):
    backend = prepare_dimer.backend

    assert isinstance(backend, Psi4Backend)
    assert backend.reference == prepare_dimer.reference == "RHF"
    assert backend.nbf == backend.basisset.nbf()


@pytest.mark.parametrize("monomer", ["A", "B"])
def test_monomer_data(prepare_dimer, monomer):
    """Orthonormal orbitals in the dimer basis set, in order of their energies, and a
    symmetric attractive potential."""

    backend = prepare_dimer.backend
    reference = backend.monomers[monomer]
    wfn = backend.wavefunctions[monomer]

    assert isinstance(reference, MonomerReference)
    assert reference.C.shape == (backend.nbf, wfn.nmo())
    assert reference.ndocc == wfn.doccpi()[0]
    assert reference.energy == wfn.energy()
    np.testing.assert_allclose(
        reference.C.T @ backend.S @ reference.C, np.eye(wfn.nmo()), atol=1e-10
    )
    assert np.all(np.diff(reference.eps) >= 0)
    np.testing.assert_allclose(reference.V, reference.V.T, atol=1e-12)
    assert np.trace(reference.V @ backend.S) < 0


def test_dimer_data(prepare_dimer):
    backend = prepare_dimer.backend

    np.testing.assert_allclose(backend.S, backend.S.T, atol=1e-14)
    np.testing.assert_allclose(backend.Qpq, backend.Qpq.transpose(0, 2, 1), atol=1e-12)
    assert backend.nuclear_repulsion == backend.molecule.nuclear_repulsion_energy()
    assert backend.fragments.A.atoms == (0,)
    assert backend.fragments.B.atoms == (1, 2)


def test_naux_counts_the_fitting_functions(prepare_dimer):
    """`naux` is the number of functions of the SAPT fitting basis, the rows of `Qpq`:
    spherical ones for a spherical basis, fewer than its Cartesian ones."""

    mol = prepare_dimer
    fitting_basis = OPTIONS["DF_BASIS_SAPT"]
    aux = psi4.core.BasisSet.build(
        mol.dimer, target=fitting_basis, other=fitting_basis, fitrole="RIFIT"
    )

    assert aux.has_puream()
    assert mol.naux == mol.Qpq.shape[0] == aux.nbf() < aux.nao()


def test_dipole_integrals(prepare_dimer):
    """Psi4's, with the electron charge in the electronic ones; the nuclear dipole is
    sum Z R over the real atoms."""

    backend = prepare_dimer.backend
    dipoles = backend.dipole_integrals()

    assert len(dipoles) == 3
    for component, psi4_component in zip(dipoles, backend.mints.ao_dipole()):
        np.testing.assert_array_equal(component, np.asarray(psi4_component))
    fragments = backend.fragments
    charges = np.where(fragments.real, fragments.atomic_numbers, 0)
    np.testing.assert_allclose(
        backend.nuclear_dipole(), charges @ fragments.coordinates, rtol=0, atol=1e-12
    )


def test_dimer_takes_the_backend_arrays(prepare_dimer):
    """The dimer's arrays are the backend's, not copies."""

    mol = prepare_dimer
    backend = mol.backend

    assert mol.S is backend.S
    assert mol.Qpq is backend.Qpq
    assert mol.C_A is backend.monomers["A"].C
    assert mol.eps_B is backend.monomers["B"].eps
    assert mol.V_A is backend.monomers["A"].V
    assert mol.fragments is backend.fragments
    assert mol.rhfA == backend.monomers["A"].energy


def test_psi4_attributes_are_the_backends(prepare_dimer):
    mol = prepare_dimer
    backend = mol.backend

    assert mol.dimer is backend.molecule
    assert mol.get_psi4_molecule() is backend.molecule
    assert mol.wfnA is backend.wavefunctions["A"]
    assert mol.wfnB is backend.wavefunctions["B"]
    assert mol.dimer_wfn is backend.dimer_wfn
    assert mol.mints is backend.mints
    assert mol.jk is backend.jk
    assert mol.jk_memory == backend.jk_memory
    assert mol.memory == backend.memory
    assert mol.psi4_options is backend.psi4_options
    assert mol.fitting_bases is backend.fitting_bases


@pytest.mark.parametrize("attribute", ["wfnA", "mints", "dimer"])
def test_psi4_attributes_are_read_only(prepare_dimer, attribute):
    """Replacing a Psi4 object of the dimer would not reach the backend, which does the
    calculations, so it raises."""

    with pytest.raises(AttributeError, match="read-only"):
        setattr(prepare_dimer, attribute, None)


def test_psi4_attributes_of_another_backend():
    """A dimer whose backend is not Psi4's says so when asked for a Psi4 object."""

    dimer = Dimer.__new__(Dimer)
    dimer.backend = SimpleNamespace()

    with pytest.raises(AttributeError, match="backend is SimpleNamespace"):
        dimer.wfnA


def test_rks_needs_a_functional():
    """Checked before any SCF is run."""

    with pytest.raises(ValueError, match="RKS reference needs a functional"):
        Psi4Backend(
            HE_H.format(atom_A="He", mult_A=1, atom_B="He", mult_B=1),
            "RKS",
            functional=None,
        )


def test_helper_from_the_backend_alone(prepare_dimer):
    """The SAPT arrays are built from the backend alone: a new object on the dimer's
    backend reproduces them."""

    mol = prepare_dimer
    rebuilt = sinfinity(mol.backend)

    for name in ["J_A", "K_B", "omegaA_bs", "omega_exchB_ar", "Qar", "t_rsab"]:
        np.testing.assert_allclose(
            getattr(rebuilt, name), getattr(mol, name), rtol=0, atol=1e-12
        )
    assert rebuilt.nuc_rep == mol.nuc_rep


def test_compute_jk_of_the_occupied_orbitals(prepare_dimer):
    mol = prepare_dimer

    (J_A, J_B), (K_A, K_B) = mol.backend.compute_jk(
        [mol.Co_A, mol.Co_B], [mol.Co_A, mol.Co_B]
    )

    np.testing.assert_allclose(J_A, mol.J_A, rtol=0, atol=1e-12)
    np.testing.assert_allclose(K_B, mol.K_B, rtol=0, atol=1e-12)


def test_failed_compute_jk_leaves_no_inputs(prepare_dimer):
    """A J/K build that fails part way clears the pairs it had added, so the next build
    returns the J and K matrices of its own pairs only."""

    mol = prepare_dimer
    backend = mol.backend

    # a 3-dimensional array is no matrix
    with pytest.raises(ValueError):
        backend.compute_jk([mol.Co_A, np.zeros((2, 2, 2))], [mol.Co_A, mol.Co_B])

    J, K = backend.compute_jk([mol.Co_B], [mol.Co_B])

    assert len(J) == len(K) == 1
    np.testing.assert_allclose(J[0], mol.J_B, rtol=0, atol=1e-12)
    np.testing.assert_allclose(K[0], mol.K_B, rtol=0, atol=1e-12)


class FailingHx:
    """Wraps a monomer wavefunction whose Hessian products fail, recording the name of
    its J/K object at the failure."""

    def __init__(self, wfn):
        self.wfn = wfn
        self.jk_names = []

    def __getattr__(self, name):
        return getattr(self.wfn, name)

    def cphf_Hx(self, x_vec):
        self.jk_names.append(self.wfn.jk().name())
        raise RuntimeError("Hessian product failed")


def test_scf_jk_restored_after_failed_product(prepare_dimer, monkeypatch):
    """A Hessian product that fails on a CPSCF J/K object leaves the monomer SCF's J/K
    object on the wavefunction."""

    mol = prepare_dimer
    backend = mol.backend
    failing = FailingHx(backend.wavefunctions["A"])
    monkeypatch.setitem(backend.wavefunctions, "A", failing)
    cpscf_jk = backend.build_cpscf_jk("DF", DF_BASIS)

    with pytest.raises(RuntimeError, match="Hessian product failed"):
        backend.orbital_hessian_product(
            "A", [np.ones((mol.sizes["a"], mol.sizes["r"]))], cpscf_jk
        )

    assert failing.jk_names == ["MemDFJK"]
    assert backend.wavefunctions["A"].jk().name() == "DirectJK"
