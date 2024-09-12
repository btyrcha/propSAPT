import tracemalloc
import psi4


def trace_memory_peak(func):
    """
    Trace peak memory usage of 'func'.
    """

    def wrapper(*args, **kwargs):
        tracemalloc.start()
        func(*args, **kwargs)
        _, peak_size = tracemalloc.get_traced_memory()
        psi4.core.print_out(f"\nFunciton: {func}\n")
        psi4.core.print_out(f"...peak meamory usage: {peak_size*1e-6:.2f} MB\n")
        tracemalloc.reset_peak()
        tracemalloc.stop()

    return wrapper
