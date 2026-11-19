"""The `prop_sapt` logger and its handler writing into Psi4's output file."""

import importlib
import logging

import psi4
import pytest

import prop_sapt.log

from .test_cpscf_solver import DenseHessian, right_hand_sides, solve


@pytest.fixture
def psi4_output(tmp_path):
    """Psi4's output file for one test; returns a function reading it. The test session's
    `output.dat` is reopened in append mode afterwards, as `conftest.py` opens it."""

    path = tmp_path / "output.dat"
    psi4.core.set_output_file(str(path), False)

    def read() -> str:
        psi4.core.flush_outfile()
        return path.read_text()

    yield read

    psi4.core.set_output_file("output.dat", True)


@pytest.fixture
def prop_sapt_level():
    """Restores the level of the `prop_sapt` logger after a test."""

    logger = logging.getLogger("prop_sapt")
    level = logger.level
    yield logger
    logger.setLevel(level)


def _handlers():
    return [
        handler
        for handler in logging.getLogger("prop_sapt").handlers
        if getattr(handler, "_prop_sapt_psi4_output", False)
    ]


def test_module_records_go_to_the_psi4_output_file(psi4_output):
    """A record of a module logger lands in Psi4's output file, one line per message."""

    logging.getLogger("prop_sapt.cpscf_solver").info("solver message")
    logging.getLogger("prop_sapt").info("first line\nsecond line")

    assert psi4_output().endswith("solver message\nfirst line\nsecond line\n")


def test_warnings_carry_their_level(psi4_output):
    logging.getLogger("prop_sapt.checks").warning("not symmetric")

    assert psi4_output().endswith("Warning: not symmetric\n")


def test_level_quiets_the_output(psi4_output, prop_sapt_level):
    prop_sapt_level.setLevel(logging.WARNING)
    logging.getLogger("prop_sapt.densities").info("quiet")
    logging.getLogger("prop_sapt.densities").warning("loud")

    text = psi4_output()
    assert "quiet" not in text
    assert text.endswith("Warning: loud\n")


def test_info_is_logged_by_default():
    assert logging.getLogger("prop_sapt").getEffectiveLevel() == logging.INFO


def test_reload_keeps_one_handler(psi4_output):
    assert len(_handlers()) == 1

    importlib.reload(prop_sapt.log)

    assert len(_handlers()) == 1
    logging.getLogger("prop_sapt").info("once")
    assert psi4_output().count("once") == 1


def _solver_records(caplog):
    return [
        record for record in caplog.records if record.name == "prop_sapt.cpscf_solver"
    ]


def test_cpscf_progress_is_logged(caplog):
    """The CPSCF table is made of INFO records of the `prop_sapt.cpscf_solver` logger,
    which reach `caplog` by propagation, each without blank lines around it."""

    with caplog.at_level(logging.INFO, logger="prop_sapt"):
        solve(DenseHessian(), right_hand_sides(2))

    records = _solver_records(caplog)
    messages = [record.getMessage() for record in records]
    assert "CPSCF: preconditioned conjugate gradient" in messages[0]
    assert {record.levelno for record in records} == {logging.INFO}
    assert not [m for m in messages if m.startswith("\n") or m.endswith("\n")]


def test_unconverged_solve_logs_a_warning(caplog):
    with (
        pytest.warns(RuntimeWarning),
        caplog.at_level(logging.INFO, logger="prop_sapt"),
    ):
        solve(DenseHessian(), right_hand_sides(2), maxiter=2)

    warned = [r for r in _solver_records(caplog) if r.levelno == logging.WARNING]
    assert len(warned) == 1
    assert "did not converge" in warned[0].getMessage()
