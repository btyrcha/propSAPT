"""propSAPT's conjugate-gradient solver of the CPSCF equations, on small dense systems
with no SCF."""

import numpy as np
import pytest

from prop_sapt.cpscf_solver import conjugate_gradient

SHAPE = (3, 5)


# elements of the arrays of `SHAPE` that H does not couple to the others
DECOUPLED = 3


class DenseHessian:
    """A symmetric positive definite H on arrays of `SHAPE`, diagonally dominant like an
    orbital Hessian, that records how many vectors each product call gets. It does not
    couple the first `DECOUPLED` elements, so it is diagonal on them."""

    def __init__(self, seed=3, sign=1.0):
        rng = np.random.default_rng(seed)
        n = np.prod(SHAPE)
        self.diagonal = rng.uniform(0.5, 3.0, SHAPE)
        coupling = 0.2 * rng.standard_normal((n, n))
        coupling[:DECOUPLED] = 0.0
        self.matrix = sign * (np.diag(self.diagonal.ravel()) + coupling @ coupling.T)
        self.calls = []

    def __call__(self, vectors):
        self.calls.append(len(vectors))
        return [(self.matrix @ v.ravel()).reshape(SHAPE) for v in vectors]

    def solve(self, b):
        return np.linalg.solve(self.matrix, b.ravel()).reshape(SHAPE)


def right_hand_sides(count, seed=5):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(SHAPE) for _ in range(count)]


def solve(hessian, rhs, r_convergence=1e-12, maxiter=100):
    return conjugate_gradient(rhs, hessian, hessian.diagonal, r_convergence, maxiter)


def test_solutions():
    hessian = DenseHessian()
    rhs = right_hand_sides(3)

    solutions = solve(hessian, rhs)

    assert len(solutions) == 3
    for x, b in zip(solutions, rhs):
        assert x.shape == SHAPE
        np.testing.assert_allclose(x, hessian.solve(b), rtol=0, atol=1e-10)


def test_converges_to_the_threshold():
    """Every solution has a relative residual below the threshold."""

    hessian = DenseHessian()
    rhs = right_hand_sides(3)

    for r_convergence in (1e-4, 1e-8):
        for x, b in zip(solve(hessian, rhs, r_convergence), rhs):
            residual = np.linalg.norm(b - hessian([x])[0]) / np.linalg.norm(b)
            assert residual < r_convergence


def test_batch_matches_single_solves():
    """Solving together changes no vector's iterates, and each product call gets the
    vectors that have not converged: the first all of them, the k-th as many as the
    single solves that take more than k calls."""

    # the first is solved by the start, b / diagonal
    easy = np.zeros(SHAPE)
    easy.flat[:DECOUPLED] = 1.0
    rhs = [easy] + right_hand_sides(2)
    single_calls = []
    single = []
    for b in rhs:
        hessian = DenseHessian()
        single.append(solve(hessian, [b], r_convergence=1e-10)[0])
        single_calls.append(len(hessian.calls))

    hessian = DenseHessian()
    batch = solve(hessian, rhs, r_convergence=1e-10)

    for x_batch, x_single in zip(batch, single):
        np.testing.assert_array_equal(x_batch, x_single)
    assert single_calls[0] == 1 < min(single_calls[1:])
    expected = [sum(n > k for n in single_calls) for k in range(max(single_calls))]
    assert hessian.calls == expected


def test_right_hand_sides_unchanged():
    rhs = right_hand_sides(2)
    copies = [b.copy() for b in rhs]

    solve(DenseHessian(), rhs)

    for b, copy in zip(rhs, copies):
        np.testing.assert_array_equal(b, copy)


def test_zero_right_hand_side():
    """A zero right-hand side has the solution zero and takes no products."""

    hessian = DenseHessian()
    rhs = right_hand_sides(1) + [np.zeros(SHAPE)]

    x, x_zero = solve(hessian, rhs)

    np.testing.assert_array_equal(x_zero, np.zeros(SHAPE))
    np.testing.assert_allclose(x, hessian.solve(rhs[0]), rtol=0, atol=1e-10)
    assert all(count == 1 for count in hessian.calls)


def test_no_right_hand_sides():
    hessian = DenseHessian()

    assert solve(hessian, []) == []
    assert hessian.calls == []


def test_wrong_shape_rejected():
    with pytest.raises(ValueError, match="right-hand side of shape"):
        solve(DenseHessian(), [np.ones((5, 3))])


def test_not_converged_warns():
    with pytest.warns(RuntimeWarning, match="2 of 2 vectors did not converge in 2"):
        solve(DenseHessian(), right_hand_sides(2), maxiter=2)


def test_not_positive_definite_warns():
    """A vector that meets a direction of negative curvature stops, with a warning."""

    hessian = DenseHessian(sign=-1.0)

    with pytest.warns(RuntimeWarning, match="not positive definite"):
        solve(hessian, right_hand_sides(1))

    assert len(hessian.calls) == 2
