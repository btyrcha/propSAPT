"""Preconditioned conjugate-gradient solver of the CPSCF equations.

`conjugate_gradient` solves H x = b for several right-hand sides b at once, where H is
symmetric positive definite and known only through its products with vectors: for the
CPSCF equations, the orbital Hessian of a monomer (see
`helper_SAPT.orbital_hessian_product`). Each iteration takes the products of all
unconverged vectors in one call, which for the CPSCF equations is one J/K build for the
whole batch. Each vector converges on its own.
"""

import time
import warnings
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import psi4

HessianProduct = Callable[[list[np.ndarray]], list[np.ndarray]]


def conjugate_gradient(
    rhs: Sequence[np.ndarray],
    hessian_product: HessianProduct,
    diagonal: np.ndarray,
    r_convergence: float,
    maxiter: int,
    title: str = "CPSCF",
) -> list[np.ndarray]:
    """
    Solves H x = b for every b in `rhs` with the preconditioned conjugate-gradient method.

    The preconditioner divides by `diagonal`, and the start is the preconditioned
    right-hand side, b / `diagonal`: for the CPSCF equations, the uncoupled solution. A
    vector has converged when its relative residual ||b - H x|| / ||b|| is below
    `r_convergence`. A zero right-hand side has the solution zero. The progress is
    printed to Psi4's output file.

    Args:
        rhs (Sequence[np.ndarray]): Right-hand sides b, arrays of the shape of
            `diagonal`. They are not changed.
        hessian_product (Callable[[list[np.ndarray]], list[np.ndarray]]): Takes a list of
            arrays x and returns the products H x, in the same order.
        diagonal (np.ndarray): Positive approximation of the diagonal of H, e.g. the
            orbital energy differences.
        r_convergence (float): Relative residual below which a vector has converged.
        maxiter (int): Maximum number of iterations, each one `hessian_product` call.
        title (str, optional): Name of the solve in the output. Defaults to "CPSCF".

    Returns:
        list[np.ndarray]: The solutions x, in the order of `rhs`.

    Warns:
        RuntimeWarning: Raised if a vector has not converged in `maxiter` iterations, or
            has met a direction in which H is not positive definite. Its last iterate is
            returned.
    """

    t_start = time.time()
    b = [np.asarray(vector, dtype=np.float64) for vector in rhs]
    for vector in b:
        if vector.shape != diagonal.shape:
            raise ValueError(
                f"{title}: right-hand side of shape {vector.shape}, "
                f"expected {diagonal.shape}."
            )
    if not b:
        return []

    b_norm = [np.linalg.norm(vector) for vector in b]
    residual = [0.0] * len(b)
    x = [vector / diagonal for vector in b]
    r, z, p, rz = {}, {}, {}, {}

    active = [i for i, norm in enumerate(b_norm) if norm > 0.0]
    for i, product in zip(active, _products(hessian_product, x, active, title)):
        r[i] = b[i] - product
        residual[i] = np.linalg.norm(r[i]) / b_norm[i]
    active = [i for i in active if residual[i] >= r_convergence]

    for i in active:
        z[i] = r[i] / diagonal
        p[i] = z[i]
        rz[i] = np.vdot(r[i], z[i])

    _print_header(title, len(b), r_convergence, maxiter)
    _print_row("Guess", residual, len(active), t_start)

    # vectors stopped at a direction of non-positive curvature
    stopped = []
    iteration = 0
    while active and iteration < maxiter:
        iteration += 1
        unconverged = []
        for i, product in zip(active, _products(hessian_product, p, active, title)):
            curvature = np.vdot(p[i], product)
            if not curvature > 0.0:
                stopped.append(i)
                continue

            alpha = rz[i] / curvature
            x[i] = x[i] + alpha * p[i]
            r[i] = r[i] - alpha * product
            residual[i] = np.linalg.norm(r[i]) / b_norm[i]
            if residual[i] < r_convergence:
                continue

            z[i] = r[i] / diagonal
            rz_new = np.vdot(r[i], z[i])
            p[i] = z[i] + (rz_new / rz[i]) * p[i]
            rz[i] = rz_new
            unconverged.append(i)

        active = unconverged
        _print_row(f"{iteration:d}", residual, len(active), t_start)

    psi4.core.print_out("   " + "-" * 56 + "\n")

    if active:
        _warn(
            f"{title}: {len(active)} of {len(b)} vectors did not converge in "
            f"{maxiter} iterations (largest relative residual "
            f"{max(residual[i] for i in active):.2e}, convergence {r_convergence:.1e})."
        )
    if stopped:
        _warn(
            f"{title}: {len(stopped)} of {len(b)} vectors stopped at a direction in "
            "which the Hessian is not positive definite (largest relative residual "
            f"{max(residual[i] for i in stopped):.2e})."
        )

    return x


def _products(
    hessian_product: HessianProduct,
    vectors: Sequence[np.ndarray] | Mapping[int, np.ndarray],
    active: list[int],
    title: str,
) -> list[np.ndarray]:
    """The products of H with the `active` ones of `vectors`, in one call."""

    if not active:
        return []

    products = hessian_product([vectors[i] for i in active])
    if len(products) != len(active):
        raise ValueError(
            f"{title}: the Hessian product returned {len(products)} vectors "
            f"for {len(active)}."
        )

    return products


def _print_header(title: str, count: int, r_convergence: float, maxiter: int):
    psi4.core.print_out(
        f"\n   {title}: preconditioned conjugate gradient\n"
        f"    Equations           = {count:11d}\n"
        f"    Convergence         = {r_convergence:11.3e}\n"
        f"    Maxiter             = {maxiter:11d}\n\n"
        f"    {'Iter':>5} {'Residual mean':>14} {'Residual max':>13} "
        f"{'Remaining':>10} {'Time [s]':>9}\n"
        "   " + "-" * 56 + "\n"
    )


def _print_row(label: str, residual: list[float], remaining: int, t_start: float):
    psi4.core.print_out(
        f"    {label:>5} {np.mean(residual):14.3e} {np.max(residual):13.3e} "
        f"{remaining:10d} {time.time() - t_start:9.1f}\n"
    )


def _warn(message: str):
    psi4.core.print_out(f"\n   Warning: {message}\n")
    warnings.warn(message, RuntimeWarning, stacklevel=3)
