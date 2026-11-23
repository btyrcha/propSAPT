"""The Psi4 backend: the data it gives the dimer, and the dimer's Psi4 attributes."""

from types import SimpleNamespace

import numpy as np
import pytest

from prop_sapt import Dimer
from prop_sapt.backends import MonomerReference
from prop_sapt.backends.psi4_backend import Psi4Backend
from prop_sapt.sinfinity import sinfinity

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
