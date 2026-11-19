"""The `prop_sapt` logger and the handler that writes its records into Psi4's output
file.

Every module of the package logs through `logging.getLogger(__name__)`, a child of the
`prop_sapt` logger, and this module is the only one that writes to Psi4's output. By
default the records go to Psi4's output file, like propSAPT's output always has. The
standard `logging` calls quiet or redirect them, e.g.
`logging.getLogger("prop_sapt").setLevel(logging.WARNING)` keeps only the warnings, and
`logging.getLogger("prop_sapt.cpscf_solver")` is the logger of the CPSCF iterations.
"""

import logging

import psi4

logger = logging.getLogger("prop_sapt")


class Psi4OutputFormatter(logging.Formatter):
    """The message alone, after its level ("Warning: ...") for warnings and errors."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.capitalize()}: {text}"

        return text


class Psi4OutputHandler(logging.Handler):
    """Writes each log record into Psi4's output file, ended by a newline."""

    # marks the handler of this module across reloads, which create a new class
    _prop_sapt_psi4_output = True

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self.setFormatter(Psi4OutputFormatter())

    def emit(self, record: logging.LogRecord):
        try:
            psi4.core.print_out(self.format(record) + "\n")
        except Exception:
            self.handleError(record)


def _install_psi4_output():
    """Sends the `prop_sapt` records to Psi4's output file, once, and logs INFO and above
    unless the logger's level was set before the package was imported."""

    if not any(
        getattr(handler, "_prop_sapt_psi4_output", False) for handler in logger.handlers
    ):
        logger.addHandler(Psi4OutputHandler())

    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


_install_psi4_output()
