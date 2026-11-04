import numpy as np
import psi4
import pytest

from .he_h2 import HE_H2
from prop_sapt import Dimer
from prop_sapt.helper_SAPT_DF import jk_memory_doubles

HE_H = """
symmetry c1
no_com
no_reorient
units bohr
0 {mult_A}
{atom_A}  0.0  0.0  0.0
--
0 {mult_B}
{atom_B}  0.0  0.0  5.0
"""


@pytest.mark.parametrize("reference", ["RHF", "RKS"])
@pytest.mark.parametrize(
    "monomers, open_shell",
    [
        ({"atom_A": "H", "mult_A": 2, "atom_B": "He", "mult_B": 1}, "A"),
        ({"atom_A": "He", "mult_A": 1, "atom_B": "H", "mult_B": 2}, "B"),
    ],
    ids=["A-doublet", "B-doublet"],
)
def test_open_shell_monomer_rejected(monomers, open_shell, reference):
    """A non-singlet monomer is rejected up front, before any SCF is run."""

    with pytest.raises(ValueError, match=f"Monomer {open_shell} has multiplicity 2"):
        Dimer(HE_H.format(**monomers), reference=reference)


@pytest.fixture
def reset_freeze_core():
    yield
    psi4.set_options({"freeze_core": "false"})
    psi4.core.revoke_global_option_changed("FREEZE_CORE")


@pytest.mark.parametrize("freeze_core", [True, -1, "POLICY"])
def test_frozen_core_rejected(reset_freeze_core, freeze_core):
    """Frozen core is rejected up front, not ignored: propSAPT is all-electron."""

    psi4.set_options({"freeze_core": freeze_core})
    with pytest.raises(ValueError, match="does not support frozen core"):
        Dimer(HE_H.format(atom_A="He", mult_A=1, atom_B="He", mult_B=1))


@pytest.mark.parametrize("freeze_core", [False, "false", 0])
def test_frozen_core_off_accepted(reset_freeze_core, freeze_core):
    """Frozen core off passes, whether Psi4 stores it as "FALSE" or "0" (`False`).

    The open-shell monomer then stops the dimer, before any SCF is run.
    """

    psi4.set_options({"freeze_core": freeze_core})
    with pytest.raises(ValueError, match="Monomer A has multiplicity 2"):
        Dimer(HE_H.format(atom_A="H", mult_A=2, atom_B="He", mult_B=1))


def test_jk_memory_follows_psi4_memory(monkeypatch):
    """Without `memory`, the JK object gets Psi4's memory times SCF_MEM_SAFETY_FACTOR, in
    doubles (8 bytes each), as Psi4's SCF gives its own JK object."""

    monkeypatch.setattr(psi4, "get_memory", lambda: 2_000_000_000)
    factor = psi4.core.get_global_option("SCF_MEM_SAFETY_FACTOR")

    assert jk_memory_doubles() == int(2e9 * factor / 8)


def test_jk_memory_of_memory_argument_in_doubles():
    """The deprecated `memory` in GB becomes doubles: 8 GB are 1e9 doubles, not 8e9."""

    assert jk_memory_doubles(8) == 1_000_000_000


def test_memory_argument_deprecated():
    """`Dimer(memory=...)` warns before any SCF; the open-shell monomer then stops it."""

    with pytest.warns(FutureWarning, match="psi4.set_memory"):
        with pytest.raises(ValueError, match="Monomer A has multiplicity 2"):
            Dimer(HE_H.format(atom_A="H", mult_A=2, atom_B="He", mult_B=1), memory=2)


@pytest.fixture
def reset_save_jk():
    yield
    psi4.set_options({"save_jk": False})
    psi4.core.revoke_global_option_changed("SAVE_JK")


def test_dimer_keeps_jk_without_save_jk(prepare_dimer, he_h2_settings, reset_save_jk):
    """With SAVE_JK off, the dimer still keeps the monomer JK objects that CPSCF needs,
    and leaves the option off."""

    psi4.set_options({"save_jk": False})
    dimer = Dimer(HE_H2)

    assert not psi4.core.get_global_option("SAVE_JK")
    assert not psi4.core.get_option("SCF", "SAVE_JK")

    # the induction energy does not depend on how degenerate orbitals are rotated
    _, ind_A = dimer.cpscf("A", ind=True)
    _, ind_A_fixture = prepare_dimer.cpscf("A", ind=True)
    assert ind_A == pytest.approx(ind_A_fixture, abs=1e-9)


@pytest.mark.parametrize("string", ["ab", "ba", "ar", "ra", "aa"])
def test_overlap_of_empty_block(prepare_dimer, monkeypatch, string):
    """An empty occupied block gives an overlap block of the matching, empty, shape.

    No closed-shell monomer has an empty block, so one is made by hand; the case is the
    singly occupied block of a closed-shell monomer under an open-shell reference.
    """

    mol = prepare_dimer
    monkeypatch.setitem(mol.orbitals, "a", mol.orbitals["a"][:, :0])
    monkeypatch.setitem(mol.sizes, "a", 0)

    expected = tuple(mol.sizes[index] for index in string)

    assert mol.s(string).shape == expected


class CountedHx:
    """Wraps a monomer wavefunction, recording how many vectors each `cphf_Hx` call gets."""

    def __init__(self, wfn):
        self.wfn = wfn
        self.calls = []

    def cphf_Hx(self, x_vec):
        self.calls.append(len(x_vec))
        return self.wfn.cphf_Hx(x_vec)


def random_perturbations(mol, monomer, count):
    occ, vir = ("a", "r") if monomer == "A" else ("b", "s")
    rng = np.random.default_rng(11)
    return [rng.standard_normal((mol.sizes[occ], mol.sizes[vir])) for _ in range(count)]


@pytest.mark.parametrize("monomer", ["A", "B"])
def test_cpscf_batch_matches_single_solves(prepare_dimer, monomer):
    """Perturbations solved together give the amplitudes of separate solves."""

    perturbations = random_perturbations(prepare_dimer, monomer, 3)

    batch = prepare_dimer.cpscf_batch(monomer, perturbations)
    single = [prepare_dimer.cpscf(monomer, perturbation=p) for p in perturbations]

    assert len(batch) == 3
    for x_batch, x_single in zip(batch, single):
        assert x_batch.shape == x_single.shape
        np.testing.assert_allclose(x_batch, x_single, rtol=0, atol=1e-12)


@pytest.mark.parametrize("monomer", ["A", "B"])
def test_cpscf_batch_builds_products_together(prepare_dimer, monkeypatch, monomer):
    """The batch takes one Hessian-product call per iteration for all unconverged
    vectors: as many calls as its slowest vector alone, the first with every vector."""

    perturbations = random_perturbations(prepare_dimer, monomer, 3)
    attribute = "wfnA" if monomer == "A" else "wfnB"
    counted = CountedHx(getattr(prepare_dimer, attribute))
    monkeypatch.setattr(prepare_dimer, attribute, counted)

    single_calls = []
    for p in perturbations:
        counted.calls.clear()
        prepare_dimer.cpscf(monomer, perturbation=p)
        single_calls.append(len(counted.calls))

    counted.calls.clear()
    prepare_dimer.cpscf_batch(monomer, perturbations)

    assert counted.calls[0] == 3
    assert len(counted.calls) == max(single_calls)


def test_cpscf_batch_of_nothing(prepare_dimer):
    assert prepare_dimer.cpscf_batch("A", []) == []


def test_cpscf_rejects_unknown_monomer(prepare_dimer):
    with pytest.raises(ValueError, match="not a valid monomer for CPSCF"):
        prepare_dimer.cpscf_batch("C", [])


def test_basisset_is_the_dimers(prepare_dimer):
    """The dimer's basis set is the one it was built with, not the one the `BASIS`
    option names now: cubes, point evaluation and fragment operators use it."""

    saved_options = psi4.p4util.OptionsState(["BASIS"])
    try:
        psi4.set_options({"basis": "cc-pvdz"})
        assert prepare_dimer.get_psi4_basisset().nbf() == prepare_dimer.nbf
        wavefunction = prepare_dimer.get_psi4_wavefunction()
        assert wavefunction.basisset().nbf() == prepare_dimer.nbf
    finally:
        saved_options.restore()
