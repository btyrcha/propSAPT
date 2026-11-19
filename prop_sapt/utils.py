import logging
import os
import time
from functools import wraps
from collections.abc import Callable
from importlib import metadata
import tracemalloc

logger = logging.getLogger(__name__)


def trace_memory_peak(func: Callable):
    """
    Decorator. Trace peak memory usage of 'func'.

    Args:
        func (Callable): A function to trace peak memory usage.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        result = func(*args, **kwargs)
        _, peak_size = tracemalloc.get_traced_memory()

        logger.info(f"Function: {func}")
        logger.info(f"...peak memory usage: {peak_size*1e-6:.2f} MB")

        tracemalloc.reset_peak()
        tracemalloc.stop()

        return result

    return wrapper


def package_version(name: str) -> str:
    """Returns the installed version of a package, or "unknown".

    Args:
        name (str): Distribution name of the package, e.g. "prop_sapt".

    Returns:
        str: The version, or "unknown" if the package is not installed.
    """

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def changed_psi4_options() -> dict | None:
    """The Psi4 options changed from their defaults, None if Psi4 cannot list them."""

    try:
        from psi4.driver.p4util import prepare_options_for_set_options
    except ImportError:
        return None

    return prepare_options_for_set_options()


def prepare_path(file_path: str) -> str:
    """
    Create directories (and subdirectories) from `file_path` if they don't exist.

    Args:
        file_path (str): Path to check or create.

    Returns:
        str: The `file_path` given.
    """

    directories = os.path.dirname(file_path)

    if not directories == "" and not os.path.exists(directories):
        os.makedirs(directories)

    return file_path


class CalcTimer(object):
    """
    Class to time calculations.
    """

    def __init__(self, name: str):
        """Initializes timer with a name.

        Args:
            name (str): Name of the timed operation.
        """

        self.name = name
        self.start = 0.0

    def __enter__(self):
        """Starts the timer.

        Returns:
            CalcTimer: Returns the CalcTimer instance to allow for use in a `with` statement.
        """

        self.start = time.time()
        logger.info(f"Starting {self.name}...")

        return self

    def __exit__(self, *args):
        """
        Stops the timer and logs the time taken for the operation.
        """

        t = time.time() - self.start
        logger.info(f"...{self.name} took a total of {t: .2f} seconds.")


def format_energy(name: str, value: float) -> str:
    """The line of an energy term: its name and its value in mEh, kcal/mol and kJ/mol.

    Args:
        name (str): Name of the energy term.
        value (float): Value of the energy term in Hartree.
    """

    name_padding = " " * (20 - len(name))
    return (
        name
        + name_padding
        + f"{value* 1000: 16.8f} mEh  {value* 627.509: 16.8f} kcal/mol  {value* 2625.50: 16.8f} kJ/mol"
    )


def energy_printer(name: str, value: float, output: str = "psi4", **kwargs):
    """Prints out the energy value in mH and kcal/mol.

    Args:
        name (str): Name of the energy term.
        value (float): Value of the energy term in Hartree.
        output (str, optional): Output method. Defaults to "psi4": the propSAPT log,
            which goes into Psi4's output file (see `prop_sapt.log`).

    Raises:
        ValueError: If an unknown output type is provided.
        ValueError: If the output filename is not provided when output is 'file'.
    """

    energy_output_string = format_energy(name, value)

    if output == "psi4":
        logger.info(energy_output_string)

    elif output == "stdout":
        print(energy_output_string)

    elif output == "file":
        output_fname = kwargs.get("output_fname")

        if output_fname is None:
            raise ValueError("Output filename must be provided when output is 'file'.")

        with open(output_fname, "a", encoding="utf-8") as f:
            f.write(energy_output_string + "\n")

    else:
        raise ValueError(
            f"Unknown output type: {output}. Supported types are 'psi4', 'stdout', and 'file'."
        )
