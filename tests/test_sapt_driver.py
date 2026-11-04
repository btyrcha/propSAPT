"""Saving of the `calc_sapt_energy` results, with the energy terms stubbed out."""

import pandas as pd
import pytest

from prop_sapt import sapt_driver


@pytest.fixture
def stub_terms(monkeypatch, tmp_path):
    """Replaces the energy terms with 1.0 each, so no calculation runs, and works in
    `tmp_path`."""

    for name in ("calc_elst1_energy", "calc_exch1_energy", "calc_exch1_s2_energy"):
        monkeypatch.setattr(sapt_driver, name, lambda dimer: 1.0)

    induction = pd.Series({"IND2,R": 1.0, "EXCH-IND2,R": 1.0, "EXCH-IND2,R(S^2)": 1.0})
    monkeypatch.setattr(sapt_driver, "calc_ind2_r_energy", lambda dimer: induction)

    dispersion = pd.Series({"DISP2": 1.0, "EXCH-DISP2": 1.0, "EXCH-DISP2(S^2)": 1.0})
    monkeypatch.setattr(sapt_driver, "calc_disp2_energy", lambda dimer: dispersion)

    monkeypatch.setattr(sapt_driver, "print_sapt_summary", lambda results, **kw: None)
    monkeypatch.chdir(tmp_path)


def test_results_saved_to_sapt_results_csv_by_default(stub_terms, tmp_path):
    """Like the other entry points, it saves its results unless told not to, under a
    name no other default uses (`calc_property` writes results.csv)."""

    results = sapt_driver.calc_sapt_energy(None)

    saved = pd.read_csv(tmp_path / "sapt_results.csv", index_col=0).iloc[:, 0]
    assert results["TOTAL"] == 6.0
    pd.testing.assert_series_equal(saved, results, check_names=False)


def test_results_true_saves_to_the_default_file(stub_terms, tmp_path):
    sapt_driver.calc_sapt_energy(None, results=True)

    assert [path.name for path in tmp_path.glob("*.csv")] == ["sapt_results.csv"]


def test_results_path(stub_terms, tmp_path):
    sapt_driver.calc_sapt_energy(None, results=str(tmp_path / "energies.csv"))

    assert [path.name for path in tmp_path.glob("*.csv")] == ["energies.csv"]


@pytest.mark.parametrize("results", [False, None])
def test_results_false_saves_nothing(stub_terms, tmp_path, results):
    sapt_driver.calc_sapt_energy(None, results=results)

    assert list(tmp_path.glob("*.csv")) == []


def test_invalid_response_names_every_choice(stub_terms):
    with pytest.raises(ValueError, match="True, False or 'both'"):
        sapt_driver.calc_sapt_energy(None, response="coupled")
