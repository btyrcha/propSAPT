"""The `cpscf_type` option: the CPSCF solves of the density matrices and the properties
on another J/K object, while the SAPT energies keep the monomer SCF's."""

import numpy as np
import pytest

import prop_sapt.helper_SAPT_DF as helper_module
from prop_sapt import (
    calc_densities,
    calc_sapt_energy,
    get_propsapt_options,
    reset_propsapt_options,
    set_propsapt_options,
)
from prop_sapt.options import check_cpscf_type
from prop_sapt.properties import perform_property_contractions

# aug-cc-pVTZ-JKFIT, the JKFIT basis of the fixture's aug-cc-pVTZ, has no helium
DF_BASIS = "def2-universal-jkfit"


@pytest.fixture
def options():
    """Resets the propSAPT options after the test."""

    yield
    reset_propsapt_options()


@pytest.fixture
def df_options(options):
    """The density-fitted CPSCF for the densities and properties."""

    set_propsapt_options({"cpscf_type": "DF", "df_basis_cpscf": DF_BASIS})


@pytest.fixture
def empty_theta_cache(prepare_dimer, monkeypatch):
    """The dimer with no cached theta amplitudes; its own cache is restored afterwards."""

    monkeypatch.setattr(prepare_dimer, "theta_amplitudes", {})
    monkeypatch.setattr(prepare_dimer, "_theta_cpscf_key", None)

    return prepare_dimer


def record_cpscf_jk(mol, monkeypatch):
    """Records the name of the CPSCF J/K object of every `cpscf_batch` call of `mol`,
    "SCF" for the monomer SCF's."""

    names = []
    cpscf_batch = mol.cpscf_batch

    def recording(*args, **kwargs):
        names.append("SCF" if mol.cpscf_jk is None else mol.cpscf_jk.name())
        return cpscf_batch(*args, **kwargs)

    monkeypatch.setattr(mol, "cpscf_batch", recording)

    return names


def assert_close_not_equal(value, reference, rtol=1e-2):
    """`value` agrees with `reference` to `rtol` of its largest element, but differs."""

    value, reference = np.asarray(value), np.asarray(reference)
    scale = np.abs(reference).max()
    np.testing.assert_allclose(value, reference, rtol=0, atol=rtol * scale)
    assert np.abs(value - reference).max() > 1e-10 * scale


@pytest.mark.parametrize(
    "value, expected", [(None, None), ("df", "DF"), ("Direct", "DIRECT")]
)
def test_cpscf_type_values(options, value, expected):
    """`cpscf_type` takes None, "DIRECT" or "DF" in any case."""

    assert check_cpscf_type(value) == expected

    set_propsapt_options({"cpscf_type": value})

    assert get_propsapt_options()["cpscf_type"] == value


@pytest.mark.parametrize(
    "new_options, match",
    [
        ({"cpscf_type": "PK"}, "cpscf_type"),
        ({"cpscf_type": "mem_df"}, "cpscf_type"),
        ({"cpscf_type": 1}, "cpscf_type"),
        ({"r_convergence": 1e-8, "cpscf_type": "CD"}, "cpscf_type"),
        ({"df_basis_cpscf": ""}, "df_basis_cpscf"),
        ({"df_basis_cpscf": 5}, "df_basis_cpscf"),
    ],
)
def test_invalid_cpscf_options_rejected(options, new_options, match):
    """Invalid values are rejected when they are set, and no option changes."""

    before = get_propsapt_options()

    with pytest.raises(ValueError, match=match):
        set_propsapt_options(new_options)

    assert get_propsapt_options() == before


def test_direct_reuses_scf_jk(prepare_dimer):
    """With the monomer SCFs run DIRECT, "DIRECT" is their own J/K object, and the
    induction amplitudes are those of the energy."""

    mol = prepare_dimer
    assert {mol.wfnA.jk().name(), mol.wfnB.jk().name()} == {"DirectJK"}
    assert mol.build_cpscf_jk("direct") is None

    with mol.use_cpscf_type("DIRECT", DF_BASIS):
        assert mol.cpscf_jk is None
        t_ra = mol.get_cpscf_ra()

    assert t_ra is mol.get_cpscf_ra()
    assert t_ra is mol.cpscf_ra


@pytest.mark.parametrize("monomer", ["A", "B"])
def test_df_cpscf_batch(prepare_dimer, monomer):
    """The DF solves agree with the DIRECT ones to the fitting error, and the monomer
    SCF's J/K objects are back afterwards."""

    mol = prepare_dimer
    occ, vir = ("a", "r") if monomer == "A" else ("b", "s")
    rng = np.random.default_rng(5)
    perturbations = [
        rng.standard_normal((mol.sizes[occ], mol.sizes[vir])) for _ in range(2)
    ]

    direct = mol.cpscf_batch(monomer, perturbations)

    with mol.use_cpscf_type("DF", DF_BASIS):
        assert mol.cpscf_jk.name() == "MemDFJK"
        df = mol.cpscf_batch(monomer, perturbations)

    assert mol.cpscf_jk is None
    assert {mol.wfnA.jk().name(), mol.wfnB.jk().name()} == {"DirectJK"}
    for x_df, x_direct in zip(df, direct):
        assert_close_not_equal(x_df, x_direct)


def test_scf_jk_restored_after_failed_solve(prepare_dimer, monkeypatch):
    """A failing solve leaves the monomer SCF's J/K object on the wavefunction."""

    mol = prepare_dimer

    def failing_solver(*args, **kwargs):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(helper_module, "cg_solver", failing_solver)

    with pytest.raises(RuntimeError, match="solver failed"):
        with mol.use_cpscf_type("DF", DF_BASIS):
            mol.cpscf_batch("A", [np.ones((mol.sizes["a"], mol.sizes["r"]))])

    assert mol.wfnA.jk().name() == "DirectJK"
    assert mol.cpscf_jk is None
    assert mol._cpscf_key is None


def test_nested_cpscf_types(prepare_dimer):
    """A nested context of the type in use keeps its J/K object; another type replaces
    it until the nested context exits."""

    mol = prepare_dimer

    with mol.use_cpscf_type("DF", DF_BASIS):
        jk = mol.cpscf_jk
        with mol.use_cpscf_type("df", DF_BASIS.upper()):
            assert mol.cpscf_jk is jk
        with mol.use_cpscf_type(None):
            assert mol.cpscf_jk is None
        assert mol.cpscf_jk is jk

    assert mol.cpscf_jk is None


def test_induction_amplitudes_kept_per_type(prepare_dimer):
    """The induction amplitudes of each CPSCF type are solved and kept apart."""

    mol = prepare_dimer
    t_scf = mol.get_cpscf_sb()

    with mol.use_cpscf_type("DF", DF_BASIS):
        t_df = mol.get_cpscf_sb()
        assert mol.get_cpscf_sb() is t_df

    assert mol.get_cpscf_sb() is t_scf
    assert_close_not_equal(t_df, t_scf)


def test_theta_amplitudes_freed_on_type_change(empty_theta_cache):
    """Theta amplitudes are kept within their CPSCF type and freed on entering another."""

    mol = empty_theta_cache
    mol.store_theta_amplitudes("scf", np.zeros(2))

    with mol.use_cpscf_type(None):
        assert "scf" in mol.theta_amplitudes

    with mol.use_cpscf_type("DF", DF_BASIS):
        assert not mol.theta_amplitudes
        mol.store_theta_amplitudes("df", np.zeros(2))

    with mol.use_cpscf_type("DF", DF_BASIS):
        assert "df" in mol.theta_amplitudes

    with mol.use_cpscf_type(None):
        assert not mol.theta_amplitudes


def test_energy_ignores_cpscf_type(prepare_dimer, options, monkeypatch):
    """The SAPT energy solves on the monomer SCF's J/K object whatever the option."""

    reference = calc_sapt_energy(prepare_dimer, results=False)

    set_propsapt_options({"cpscf_type": "DF", "df_basis_cpscf": DF_BASIS})
    names = record_cpscf_jk(prepare_dimer, monkeypatch)
    # the energy solves its induction amplitudes again
    monkeypatch.setattr(prepare_dimer, "cpscf_ra", None)
    monkeypatch.setattr(prepare_dimer, "cpscf_sb", None)

    energy = calc_sapt_energy(prepare_dimer, results=False)

    assert names == ["SCF", "SCF"]
    for key, value in reference.items():
        assert energy[key] == pytest.approx(value, rel=0, abs=1e-12)


def test_densities_use_cpscf_type(prepare_dimer, df_options, monkeypatch):
    """Every CPSCF solve of the densities is density fitted, with one J/K object for
    both monomers, and the densities agree with the DIRECT ones to the fitting error."""

    mol = prepare_dimer
    reset_propsapt_options("cpscf_type")
    reference = calc_densities(mol)

    set_propsapt_options({"cpscf_type": "DF"})
    names = record_cpscf_jk(mol, monkeypatch)
    builds = []
    build_cpscf_jk = mol.build_cpscf_jk

    def counted_build(*args, **kwargs):
        builds.append(args)
        return build_cpscf_jk(*args, **kwargs)

    monkeypatch.setattr(mol, "build_cpscf_jk", counted_build)

    densities = calc_densities(mol)

    assert len(builds) == 1
    assert names and set(names) == {"MemDFJK"}
    assert not mol.theta_amplitudes
    for key in ["pol", "exch", "ind", "exch-ind_S2", "disp", "exch-disp_S2", "total"]:
        assert_close_not_equal(densities[key], reference[key])


def test_properties_use_cpscf_type(prepare_dimer, df_options, monkeypatch):
    """Every CPSCF solve of a property is density fitted, and the terms agree with the
    DIRECT ones to the fitting error."""

    mol = prepare_dimer
    d_X = np.asarray(mol.mints.ao_dipole()[0])
    reset_propsapt_options("cpscf_type")
    reference = perform_property_contractions(mol, d_X)

    set_propsapt_options({"cpscf_type": "DF"})
    names = record_cpscf_jk(mol, monkeypatch)

    terms = perform_property_contractions(mol, d_X)

    assert names and set(names) == {"MemDFJK"}
    assert_close_not_equal(terms.to_numpy(dtype=float), reference.to_numpy(dtype=float))
