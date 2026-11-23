"""The backend contract, `prop_sapt.backends.base`, on arrays alone."""

import numpy as np
import pytest

from prop_sapt.backends import Backend, MonomerReference, check_reference

NBF, NAUX = 6, 9


class ArrayBackend(Backend):
    """A backend of given arrays, with operations that do nothing."""

    def compute_jk(self, lefts, rights):
        return [], []

    def orbital_hessian_product(self, monomer, vectors, cpscf_jk=None):
        return []

    def dipole_integrals(self):
        return [np.zeros((NBF, NBF))] * 3

    def nuclear_dipole(self):
        return np.zeros(3)


def monomer(ndocc=2, nmo=NBF, nbf=NBF, **changes):
    rng = np.random.default_rng(7)
    data = {
        "C": rng.standard_normal((nbf, nmo)),
        "eps": np.sort(rng.standard_normal(nmo)),
        "ndocc": ndocc,
        "V": -np.eye(nbf),
        "energy": -1.0,
        "nuclear_repulsion": 0.5,
    }
    data.update(changes)
    return MonomerReference(**data)


def backend_data(**changes):
    data = {
        "reference": "RHF",
        "monomers": {"A": monomer(1), "B": monomer(2)},
        "S": np.eye(NBF),
        "Qpq": np.zeros((NAUX, NBF, NBF)),
        "nuclear_repulsion": 2.0,
        "fragments": None,
    }
    data.update(changes)
    return data


@pytest.mark.parametrize("reference", ["RHF", "RKS"])
def test_supported_references(reference):
    assert check_reference(reference) == reference


@pytest.mark.parametrize("reference", ["ROHF", "UHF", "UKS"])
def test_open_shell_references_not_implemented(reference):
    with pytest.raises(ValueError, match=f"Reference '{reference}' not implemented"):
        check_reference(reference)


@pytest.mark.parametrize("reference", ["MP2", "rhf"])
def test_unknown_reference(reference):
    with pytest.raises(ValueError, match=f"Unsupported reference type '{reference}'"):
        check_reference(reference)


def test_monomer_sizes():
    reference = monomer(ndocc=2, nmo=5)

    assert (reference.nbf, reference.nmo) == (NBF, 5)
    assert reference.nsocc == 0


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"eps": np.zeros(NBF - 1)}, "Orbital energies should have shape"),
        ({"V": np.zeros((NBF - 1, NBF - 1))}, "V should have shape"),
        ({"C": np.zeros(NBF)}, "C should be a 2-dimensional array"),
        ({"ndocc": NBF + 1}, "do not fit"),
        ({"ndocc": -1}, "do not fit"),
        ({"nsocc": NBF}, "do not fit"),
    ],
)
def test_inconsistent_monomer(changes, message):
    with pytest.raises(ValueError, match=message):
        monomer(**changes)


def test_backend_keeps_its_arrays():
    """The backend keeps the arrays it is given, without copies."""

    data = backend_data()
    backend = ArrayBackend(**data)

    assert backend.nbf == NBF
    assert backend.S is data["S"]
    assert backend.Qpq is data["Qpq"]
    assert backend.monomers["A"] is data["monomers"]["A"]
    assert backend.nuclear_repulsion == 2.0


def test_functional_only_for_rks():
    rhf = ArrayBackend(**backend_data(functional="PBE0"))
    rks = ArrayBackend(**backend_data(reference="RKS", functional="PBE0"))

    assert rhf.functional is None
    assert rks.functional == "PBE0"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"reference": "UHF"}, "not implemented"),
        ({"monomers": {"A": monomer()}}, "Monomers should be"),
        ({"monomers": {"A": monomer(), "B": None}}, "should be a MonomerReference"),
        ({"S": np.zeros((NBF, NBF + 1))}, "S should be square"),
        ({"Qpq": np.zeros((NAUX, NBF, NBF + 1))}, "Qpq should have shape"),
        ({"Qpq": np.zeros((NBF, NBF))}, "Qpq should be a 3-dimensional array"),
        (
            {"monomers": {"A": monomer(), "B": monomer(nbf=NBF - 1, nmo=NBF - 1)}},
            "Monomer B has 5 basis functions",
        ),
    ],
)
def test_inconsistent_backend(changes, message):
    with pytest.raises(ValueError, match=message):
        ArrayBackend(**backend_data(**changes))


def test_default_hooks_do_nothing():
    backend = ArrayBackend(**backend_data())

    assert backend.clean_scratch() is None
    with backend.timer():
        pass


def test_backend_needs_the_operations():
    class NoOperations(Backend):
        pass

    with pytest.raises(TypeError, match="abstract"):
        NoOperations(**backend_data())


def test_no_cpscf_jk_types_by_default():
    backend = ArrayBackend(**backend_data())

    with pytest.raises(NotImplementedError, match="ArrayBackend backend has no CPSCF"):
        backend.build_cpscf_jk("DF")
