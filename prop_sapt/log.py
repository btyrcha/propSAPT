"""The `prop_sapt` logger, and where its records go.

Every module of the package logs through `logging.getLogger(__name__)`, a child of the
`prop_sapt` logger. By default the records go to Psi4's output file, like propSAPT's
output always has, through the Psi4 backend's handler
(`prop_sapt.backends.psi4_backend.Psi4OutputHandler`), the only code writing to Psi4's
output. The standard `logging` calls quiet or redirect them, e.g.
`logging.getLogger("prop_sapt").setLevel(logging.WARNING)` keeps only the warnings, and
`logging.getLogger("prop_sapt.cpscf_solver")` is the logger of the CPSCF iterations.
"""

import logging

from .backends.psi4_backend import Psi4OutputFormatter, Psi4OutputHandler

logger = logging.getLogger("prop_sapt")


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
