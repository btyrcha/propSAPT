"""Tests of the configurable runner (`prop_sapt.runner`).

The plumbing tests replace `Dimer` and the task drivers with recorders; the He-H2 tests run
real calculations under the settings of `he_h2.py`.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import psi4

import prop_sapt.runner as runner
from .he_h2 import HE_H2, OPTIONS
from .test_sapt_components import EXPECTED_RESULTS, THRESHOLD
from prop_sapt import __version__, reset_propsapt_options, get_propsapt_options
from prop_sapt.basis_overrides import apply_basis_overrides
from prop_sapt.config import resolve_config
from prop_sapt.fragments import parse_fragments
from prop_sapt.sapt_components import calc_elst1_energy, calc_disp2_energy

# The runner equivalent of the Psi4 options in he_h2.py. DF_BASIS_SCF is left to
# Psi4 there: aug-cc-pvtz has no JKFIT basis for He.
HE_H2_CONFIG = {
    "basis": "aug-cc-pvtz",
    "df_basis_sapt": "aug-cc-pvqz-ri",
    "memory": "2 GB",
    "psi4_options": {
        key: value
        for key, value in OPTIONS.items()
        if key.lower() not in ("basis", "df_basis_sapt", "save_jk")
    },
}


@pytest.fixture(autouse=True, scope="module")
def restore_psi4_state():
    """Restores the global Psi4 and propSAPT state that the runner resets, so the test
    modules after this one see the settings of he_h2.py."""

    threads = psi4.core.get_num_threads()

    yield

    psi4.core.clean_options()
    psi4.set_options(OPTIONS)
    psi4.set_memory("2 GB")
    psi4.set_num_threads(threads)
    psi4.core.set_output_file("output.dat", True)
    reset_propsapt_options()


@pytest.fixture()
def geometry_file(tmp_path):
    """The He-H2 geometry of he_h2.py in a .p4m file."""

    path = tmp_path / "he-h2.p4m"
    path.write_text(HE_H2, encoding="utf-8")

    return path


class Recorder:
    """Stands in for `Dimer` and the task drivers, recording how they are called."""

    def __init__(self):
        self.calls = {}

    def record(self, name, result=None):
        def recorded(*args, **kwargs):
            self.calls[name] = (args, kwargs)
            return result

        return recorded


@pytest.fixture()
def recorder(monkeypatch):
    """Replaces `Dimer` and the task drivers in the runner with recorders."""

    recorder = Recorder()
    monkeypatch.setattr(runner, "Dimer", recorder.record("Dimer", "dimer"))
    for name in ("calc_sapt_energy", "calc_property", "calc_densities"):
        monkeypatch.setattr(runner, name, recorder.record(name, name + " result"))

    return recorder


def test_run_forwards_settings(recorder, geometry_file, tmp_path):
    config = resolve_config(
        {
            "reference": "RKS",
            "functional": "b3lyp",
            "grac": {"A": 0.08, "B": 0.09},
            "memory": 3,
            "threads": 2,
            "propsapt_options": {"r_convergence": 1.0e-9},
            "output": {"dir": str(tmp_path / "out" / "{geometry}")},
            "tasks": {
                "sapt_energy": {"response": "both"},
                "property": {"response": False},
                "densities": {"orbital_basis": "mo", "save_cubes": False},
            },
        }
    )

    results = runner.run(config, geometry_file)

    out_dir = tmp_path / "out" / "he-h2"
    assert results == {
        "sapt_energy": "calc_sapt_energy result",
        "property": "calc_property result",
        "densities": "calc_densities result",
    }

    # the dimer follows Psi4's memory, which the runner sets
    assert psi4.get_memory() == 3_000_000_000
    args, kwargs = recorder.calls["Dimer"]
    assert args == (HE_H2,)
    assert kwargs == {
        "reference": "RKS",
        "functional": "b3lyp",
        "grac_A": 0.08,
        "grac_B": 0.09,
    }

    args, kwargs = recorder.calls["calc_sapt_energy"]
    assert args == ("dimer",)
    assert kwargs == {"results": str(out_dir / "sapt_results.csv"), "response": "both"}

    args, kwargs = recorder.calls["calc_property"]
    assert args == ("dimer", "dipole")
    assert kwargs == {
        "results": str(out_dir / "property_results.csv"),
        "response": False,
    }

    args, kwargs = recorder.calls["calc_densities"]
    assert kwargs == {
        "orbital_basis": "MO",
        "save_cubes": False,
        "cubes_dir": str(out_dir),
        "save_matrices": True,
        "matrices_dir": str(out_dir),
        "matrix_format": "npz",
        "save_trexio": False,
        "trexio_dir": str(out_dir),
        "metadata": {
            "run": {
                "config_file": None,
                "config": config.to_dict(),
                "basis_overrides_applied": {},
            }
        },
    }

    # Psi4 and propSAPT were set up from the configuration
    assert psi4.core.get_num_threads() == 2
    assert psi4.core.get_global_option("DF_BASIS_SAPT") == "AUG-CC-PVTZ-RI"
    assert psi4.core.get_global_option("SAVE_JK")
    assert get_propsapt_options()["r_convergence"] == 1.0e-9
    assert (out_dir / "output_propsapt_dft_b3lyp.dat").is_file()

    record = json.loads((out_dir / "propsapt_run.json").read_text(encoding="utf-8"))
    assert record["geometry_file"] == str(geometry_file.resolve())
    assert record["tasks"] == ["sapt_energy", "property", "densities"]
    assert record["config"]["functional"] == "b3lyp"
    assert record["psi4_options"]["DF_BASIS_SAPT"] == "aug-cc-pvtz-ri"
    assert record["prop_sapt_version"] == __version__
    assert record["fragments"] == parse_fragments(HE_H2).to_dict()


def test_rhf_dimer_gets_no_dft_settings(recorder, geometry_file, tmp_path):
    config = resolve_config(
        {"output": {"dir": str(tmp_path)}, "tasks": ["sapt_energy"]}
    )

    runner.run(config, geometry_file)

    _, kwargs = recorder.calls["Dimer"]
    assert kwargs == {"reference": "RHF"}


def test_run_resets_options_left_by_earlier_calculations(
    recorder, geometry_file, tmp_path
):
    psi4.set_options({"freeze_core": True, "DF_BASIS_SCF": "cc-pvdz-jkfit"})

    runner.run(
        resolve_config({"output": {"dir": str(tmp_path)}, "tasks": ["sapt_energy"]}),
        geometry_file,
    )

    assert psi4.core.get_global_option("FREEZE_CORE") == "FALSE"
    assert not psi4.core.has_global_option_changed("DF_BASIS_SCF")


def test_basis_overrides_apply_only_to_options_in_use():
    psi4.core.clean_options()
    molecule = psi4.geometry(HE_H2)
    molecule.update_geometry()
    options = {"basis": "aug-cc-pvtz", "DF_BASIS_SAPT": "aug-cc-pvqz-ri"}
    psi4.set_options(options)

    overrides = {
        "DF_BASIS_SAPT": {"He": "aug-cc-pvtz-ri"},
        "DF_BASIS_MP2": {"He": "cc-pvdz-ri"},
    }
    applied = apply_basis_overrides(options, overrides)

    assert applied == ["DF_BASIS_SAPT"]
    assert not psi4.core.has_global_option_changed("DF_BASIS_MP2")

    def ri_basis_size(mol, target):
        basis = psi4.core.BasisSet.build(
            mol, target=target, other=target, fitrole="RIFIT"
        )
        return basis.nbf()

    helium = molecule.extract_subsets(1)
    hydrogen = molecule.extract_subsets(2)
    expected = ri_basis_size(helium, "aug-cc-pvtz-ri") + ri_basis_size(
        hydrogen, "aug-cc-pvqz-ri"
    )

    # the way helper_SAPT_DF builds the DF_BASIS_SAPT basis
    target = psi4.core.get_global_option("DF_BASIS_SAPT")
    assert ri_basis_size(molecule, target) == expected
    assert expected < ri_basis_size(molecule, "aug-cc-pvqz-ri")


def test_basis_dirs_make_extra_basis_files_available(tmp_path, monkeypatch):
    # a basis set Psi4's library does not have, as if downloaded from the Basis Set Exchange
    library_file = Path(psi4.core.get_datadir()) / "basis" / "def2-universal-jkfit.gbs"
    (tmp_path / "basis").mkdir()
    shutil.copy(library_file, tmp_path / "basis" / "propsapt-test-jkfit.gbs")

    monkeypatch.setenv("PSIPATH", "/some/existing/path")
    config = resolve_config(
        {
            "basis_dirs": "basis",
            "df_basis_scf": "propsapt-test-jkfit",
            "tasks": ["sapt_energy"],
        },
        base_dir=tmp_path,
    )
    runner.setup_psi4(config, str(tmp_path / "output.dat"))
    runner.setup_psi4(config, str(tmp_path / "output.dat"))  # a second run adds nothing

    assert os.environ["PSIPATH"] == os.pathsep.join(
        [str(tmp_path / "basis"), "/some/existing/path"]
    )

    molecule = psi4.geometry(HE_H2)
    molecule.update_geometry()

    def jkfit_basis_size(target):
        return psi4.core.BasisSet.build(
            molecule, "DF_BASIS_SCF", target, "JKFIT", "aug-cc-pvtz"
        ).nbf()

    # the way helper_SAPT_DF builds the DF_BASIS_SCF basis
    target = psi4.core.get_option("SCF", "DF_BASIS_SCF")
    assert target == "PROPSAPT-TEST-JKFIT"
    assert jkfit_basis_size(target) == jkfit_basis_size("def2-universal-jkfit")


def test_he_h2_run_reproduces_sapt_components(geometry_file, tmp_path):
    config = resolve_config(
        {
            **HE_H2_CONFIG,
            "output": {"dir": str(tmp_path / "{geometry}")},
            "tasks": ["sapt_energy", "monomer_references"],
        }
    )

    results = runner.run(config, geometry_file)

    sapt = results["sapt_energy"]
    for key in [
        "ELST1",
        "EXCH1",
        "EXCH1(S^2)",
        "IND2,R",
        "EXCH-IND2,R",
        "EXCH-IND2,R(S^2)",
        "DISP2",
        "EXCH-DISP2",
        "EXCH-DISP2(S^2)",
    ]:
        assert sapt[key] == pytest.approx(EXPECTED_RESULTS[key], abs=THRESHOLD), key

    out_dir = tmp_path / "he-h2"
    for filename in [
        "sapt_results.csv",
        "dm_hf_A.cube",
        "dm_hf_B.cube",
        "monomer_a.fchk",
        "monomer_b.fchk",
        "output_hf.dat",
        "propsapt_run.json",
    ]:
        assert (out_dir / filename).is_file(), filename


@pytest.fixture(scope="module")
def he_h2_energies():
    """ELST1 and DISP2 of He-H2 built by the runner, without basis overrides."""

    return energies_with_overrides(None)


def energies_with_overrides(basis_overrides) -> np.ndarray:
    """Builds the He-H2 dimer with the given basis overrides and returns ELST1 and DISP2
    with the size of the SAPT fitting basis."""

    config = resolve_config(
        {**HE_H2_CONFIG, "basis_overrides": basis_overrides, "tasks": ["sapt_energy"]}
    )
    runner.setup_psi4(config, "output.dat")
    dimer = runner.build_dimer(config, HE_H2)

    return np.array(
        [calc_elst1_energy(dimer), calc_disp2_energy(dimer)["DISP2"], dimer.nao_aux]
    )


def test_sapt_fitting_basis_override_to_the_default_changes_nothing(he_h2_energies):
    energies = energies_with_overrides({"He": {"DF_BASIS_SAPT": "aug-cc-pvqz-ri"}})

    np.testing.assert_allclose(energies, he_h2_energies, rtol=0.0, atol=1.0e-10)


def test_sapt_fitting_basis_override_reaches_the_dimer(he_h2_energies):
    energies = energies_with_overrides({"He": {"DF_BASIS_SAPT": "aug-cc-pvtz-ri"}})

    # a smaller fitting basis on He only
    assert energies[2] < he_h2_energies[2]
    # DISP2 is computed with the fitted integrals, so it moves with the fitting basis
    assert abs(energies[1] - he_h2_energies[1]) > 1.0e-9
