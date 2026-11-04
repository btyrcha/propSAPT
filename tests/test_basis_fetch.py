"""Tests of `prop-sapt basis fetch` (`prop_sapt.basis_fetch`).

Most tests use a stand-in source serving the He block of Psi4's def2-universal-jkfit in the
style of ccrepo_utility's Psi4 writer, so they need neither optional package. The tests of
the real sources run only where basis_set_exchange or ccrepo_utility is installed.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import psi4

import prop_sapt.runner as runner
from .he_h2 import HE_H2, OPTIONS
from prop_sapt import reset_propsapt_options
from prop_sapt.basis_fetch import SOURCES, BasisFetchError, fetch_basis_file
from prop_sapt.cli import main
from prop_sapt.config import resolve_config

LIBRARY_BASIS_DIR = Path(psi4.core.get_datadir()) / "basis"


@pytest.fixture(autouse=True, scope="module")
def restore_psi4_state():
    """Restores the global Psi4 and propSAPT state for the test modules after this one."""

    threads = psi4.core.get_num_threads()

    yield

    psi4.core.clean_options()
    psi4.set_options(OPTIONS)
    psi4.set_memory("2 GB")
    psi4.set_num_threads(threads)
    psi4.core.set_output_file("output.dat", True)
    reset_propsapt_options()


def library_block(basis: str, element: str) -> list[str]:
    """Returns the lines of one element's entry in a basis set of Psi4's library."""

    lines = (
        (LIBRARY_BASIS_DIR / f"{basis}.gbs").read_text(encoding="utf-8").splitlines()
    )
    entry = re.compile(rf"^\s*{element}\s+0\s*$", re.IGNORECASE)
    start = next(i for i, line in enumerate(lines) if entry.match(line))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "****")

    return lines[start:end]


def ccrepo_style_helium_jkfit() -> str:
    """The He block of def2-universal-jkfit as ccrepo_utility writes Psi4 files: no
    spherical/cartesian line, and D exponents."""

    block = [
        re.sub(r"[eE]([+-])", r"D\1", line)
        for line in library_block("def2-universal-jkfit", "He")
    ]

    return "****\n" + "\n".join(block) + "\n****"


@pytest.fixture()
def fake_source(monkeypatch):
    """Registers a stand-in basis set source "fake" that serves He only."""

    calls = []

    def fetch(name, elements):
        calls.append((name, list(elements)))
        return ccrepo_style_helium_jkfit(), "fake source 1.0"

    monkeypatch.setitem(SOURCES, "fake", fetch)

    return calls


def he_h2_molecule() -> psi4.core.Molecule:
    molecule = psi4.geometry(HE_H2)
    molecule.update_geometry()
    return molecule


def jkfit_size(molecule, target: str) -> int:
    """Size of a JKFIT basis, built the way helper_SAPT_DF builds DF_BASIS_SCF."""

    return psi4.core.BasisSet.build(
        molecule, "DF_BASIS_SCF", target, "JKFIT", "aug-cc-pvtz"
    ).nbf()


def test_fetch_writes_named_file_with_provenance(fake_source, tmp_path):
    path, basis_name = fetch_basis_file(
        "fake", "He-JKFIT", [" he "], directory=tmp_path / "basis"
    )

    assert basis_name == "fake-He-JKFIT"
    assert path == tmp_path / "basis" / "fake-he-jkfit.gbs"
    assert fake_source == [("He-JKFIT", ["He"])]

    text = path.read_text(encoding="utf-8")
    assert text.startswith(
        "! Basis set fake-He-JKFIT, written by `prop-sapt basis fetch fake He-JKFIT"
    )
    assert "! Source: fake source 1.0" in text
    assert "elements: He" in text
    assert ccrepo_style_helium_jkfit() in text


def test_fetch_refuses_to_overwrite_unless_forced(fake_source, tmp_path):
    path, _ = fetch_basis_file("fake", "He-JKFIT", ["He"], directory=tmp_path)
    path.write_text("edited", encoding="utf-8")

    with pytest.raises(BasisFetchError, match="already exists"):
        fetch_basis_file("fake", "He-JKFIT", ["He"], directory=tmp_path)
    assert path.read_text(encoding="utf-8") == "edited"

    fetch_basis_file("fake", "He-JKFIT", ["He"], directory=tmp_path, force=True)
    assert path.read_text(encoding="utf-8").startswith("! Basis set fake-He-JKFIT")


def test_fetch_refuses_names_of_library_basis_sets(fake_source, tmp_path):
    # a He-only aug-cc-pvtz-jkfit.gbs would hide the library file for H and every other element
    with pytest.raises(BasisFetchError, match="library already has a basis set"):
        fetch_basis_file(
            "fake", "He-JKFIT", ["He"], directory=tmp_path, as_name="aug-cc-pVTZ-JKFIT"
        )

    assert fake_source == []
    assert not any(tmp_path.iterdir())


def test_fetch_refuses_missing_elements(fake_source, tmp_path):
    with pytest.raises(BasisFetchError, match="not available for Li"):
        fetch_basis_file("fake", "He-JKFIT", ["He", "Li"], directory=tmp_path)

    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"source": "nowhere"}, "Unknown basis set source"),
        ({"elements": ["Xx"]}, "not an element symbol"),
        ({"elements": [" ", ""]}, "No elements given"),
        ({"as_name": "bse:he-jkfit"}, "can only contain"),
    ],
)
def test_fetch_rejects_invalid_requests(fake_source, tmp_path, kwargs, message):
    request = {
        "source": "fake",
        "name": "He-JKFIT",
        "elements": ["He"],
        "directory": tmp_path,
    }

    with pytest.raises(BasisFetchError, match=message):
        fetch_basis_file(**{**request, **kwargs})


def test_fetched_file_fills_a_missing_element_through_overrides(
    fake_source, tmp_path, monkeypatch
):
    # aug-cc-pvtz-jkfit has no He: the fetched file supplies it
    monkeypatch.delenv("PSIPATH", raising=False)
    _, basis_name = fetch_basis_file(
        "fake", "He-JKFIT", ["He"], directory=tmp_path / "basis"
    )
    raw = {
        "df_basis_scf": "aug-cc-pvtz-jkfit",
        "basis_dirs": "basis",
        "tasks": ["sapt_energy"],
    }

    runner.setup_psi4(
        resolve_config(raw, base_dir=tmp_path), str(tmp_path / "output.dat")
    )
    molecule = he_h2_molecule()
    with pytest.raises(Exception, match="Unable to find a basis set for atom 1"):
        jkfit_size(molecule, psi4.core.get_option("SCF", "DF_BASIS_SCF"))

    raw["basis_overrides"] = {"He": {"DF_BASIS_SCF": basis_name}}
    runner.setup_psi4(
        resolve_config(raw, base_dir=tmp_path), str(tmp_path / "output.dat")
    )

    expected = jkfit_size(
        molecule.extract_subsets(2), "aug-cc-pvtz-jkfit"
    ) + jkfit_size(molecule.extract_subsets(1), "def2-universal-jkfit")
    assert jkfit_size(molecule, psi4.core.get_option("SCF", "DF_BASIS_SCF")) == expected


def test_cli_basis_fetch(fake_source, tmp_path, capsys):
    status = main(
        [
            "basis",
            "fetch",
            "fake",
            "He-JKFIT",
            "--elements",
            "He",
            "--dir",
            str(tmp_path),
        ]
    )

    assert status == 0
    out = capsys.readouterr().out
    assert f"Wrote {tmp_path / 'fake-he-jkfit.gbs'}" in out
    assert "basis_overrides: {He: {DF_BASIS_SCF: fake-He-JKFIT}}" in out

    # a second fetch into the same file is refused
    status = main(
        [
            "basis",
            "fetch",
            "fake",
            "He-JKFIT",
            "--elements",
            "He",
            "--dir",
            str(tmp_path),
        ]
    )
    assert status == 2
    assert "already exists" in capsys.readouterr().err


def test_cli_basis_fetch_rejects_unknown_sources(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["basis", "fetch", "nowhere", "cc-pvdz", "--elements", "He"])

    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    "source, modules, hint",
    [
        ("bse", ["basis_set_exchange"], "pip install basis_set_exchange"),
        (
            "ccrepo",
            ["ccrepo", "ccrepo.fetch"],
            "ccrepo_utility, but ccrepo_utility is not installed",
        ),
    ],
)
def test_cli_basis_fetch_explains_missing_packages(
    tmp_path, capsys, monkeypatch, source, modules, hint
):
    for module in modules:
        monkeypatch.setitem(sys.modules, module, None)  # makes the import fail

    status = main(
        [
            "basis",
            "fetch",
            source,
            "cc-pvdz",
            "--elements",
            "He",
            "--dir",
            str(tmp_path),
        ]
    )

    assert status == 2
    assert hint in capsys.readouterr().err


def scf_energy(basis_setup) -> float:
    """SCF energy of He-H2 with the basis set up by `basis_setup`."""

    psi4.core.clean_options()
    psi4.set_options(
        {"scf_type": "pk", "e_convergence": 1.0e-10, "d_convergence": 1.0e-10}
    )
    basis_setup()

    return psi4.energy("scf", molecule=he_h2_molecule())


@pytest.mark.skipif(
    importlib.util.find_spec("basis_set_exchange") is None,
    reason="basis_set_exchange not installed",
)
def test_bse_source(tmp_path, monkeypatch):
    path, basis_name = fetch_basis_file(
        "bse", "def2-universal-jkfit", ["He", "H"], directory=tmp_path
    )
    monkeypatch.setenv("PSIPATH", str(tmp_path))

    assert "Basis Set Exchange" in path.read_text(encoding="utf-8")
    assert jkfit_size(he_h2_molecule(), basis_name) == jkfit_size(
        he_h2_molecule(), "def2-universal-jkfit"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("ccrepo") is None, reason="ccrepo_utility not installed"
)
def test_ccrepo_source(tmp_path, monkeypatch):
    path, basis_name = fetch_basis_file(
        "ccrepo", "cc-pVDZ", ["He", "H"], directory=tmp_path
    )
    monkeypatch.setenv("PSIPATH", str(tmp_path))

    assert "ccRepo" in path.read_text(encoding="utf-8")
    assert scf_energy(lambda: psi4.set_options({"basis": basis_name})) == pytest.approx(
        scf_energy(lambda: psi4.set_options({"basis": "cc-pvdz"})), abs=1.0e-9
    )
