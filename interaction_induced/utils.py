import os
import time
from functools import wraps
from collections.abc import Callable
import tracemalloc
import psi4


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

        psi4.core.print_out(f"\nFunciton: {func}\n")
        psi4.core.print_out(f"...peak meamory usage: {peak_size*1e-6:.2f} MB\n\n")

        tracemalloc.reset_peak()
        tracemalloc.stop()

        return result

    return wrapper


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
        self.start = time.time()
        psi4.core.print_out(f"\nStarting {name}...")

    def stop(self):
        """
        Stops the timer. And prints out the time taken for the operation into the output file.
        """

        t = time.time() - self.start
        psi4.core.print_out(f"...{self.name} took a total of {t: .2f} seconds.")


def energy_printer(name: str, value: float):
    """Prints out the energy value in mH and kcal/mol into the output file.

    Args:
        name (str): Name of the energy term.
        value (float): Value of the energy term in Hartree.
    """

    spacer = " " * (20 - len(name))
    psi4.core.print_out(
        name + spacer + f"{value* 1000: 16.8f} mH  {value* 627.509: 16.8f} kcal/mol"
    )
