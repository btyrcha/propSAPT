"""
The matched-integrals switch that `test_calc_property.py` relies on.

`tests/matched_integrals.py` replaces `helper_SAPT.compute_jk` by a J/K built from the
`DF_BASIS_SAPT` tensors. A wrong index convention there would not fail loudly -- it
would only move the finite-field reference -- so the conventions are pinned here
against explicit four-index integrals. Pure NumPy apart from the patch/restore check;
no quantum chemistry.
"""

import numpy as np
import pytest

from prop_sapt.helper_SAPT_DF import helper_SAPT

from . import matched_integrals as integrals

NBF, NAUX = 7, 13


class FittedHelper:
    """Just enough of a `helper_SAPT` for `df_compute_jk`: the fitted tensors."""

    def __init__(self, Qpq):
        self.Qpq = Qpq


@pytest.fixture
def rng():
    return np.random.default_rng(20260911)


@pytest.fixture
def helper(rng):
    Qpq = rng.standard_normal((NAUX, NBF, NBF))
    return FittedHelper(Qpq + Qpq.transpose(0, 2, 1))


def reference_jk(helper, density):
    """Psi4's conventions spelled out on the four-index DF integrals."""

    eri = np.einsum("Qpq,Qrs->pqrs", helper.Qpq, helper.Qpq)
    J = np.einsum("mnls,ls->mn", eri, density)
    K = np.einsum("mlns,ls->mn", eri, density)
    return J, K


def test_j_and_k_follow_psi4_conventions(helper, rng):
    """D = Cl Cr^T, J_mn = (mn|ls) D_ls, K_mn = (ml|ns) D_ls -- with a non-symmetric D,
    so that a transposed K would not pass."""

    Cl = rng.standard_normal((NBF, 3))
    Cr = rng.standard_normal((NBF, 3))

    J, K = integrals.df_compute_jk(helper, Cl, Cr)
    J_ref, K_ref = reference_jk(helper, Cl @ Cr.T)

    np.testing.assert_allclose(J, J_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(K, K_ref, rtol=0, atol=1e-10)
    assert not np.allclose(K, K.T), "the check needs a K that is not symmetric"


@pytest.mark.parametrize("n_left, n_right", [(4, 2), (2, 4)])
def test_tensor_is_folded_into_the_narrower_side(helper, rng, n_left, n_right):
    """Both foldings of `tensor` must amount to D = Cl T Cr^T, as in the original."""

    Cl = rng.standard_normal((NBF, n_left))
    Cr = rng.standard_normal((NBF, n_right))
    fold = rng.standard_normal((n_left, n_right))

    J, K = integrals.df_compute_jk(helper, Cl, Cr, tensor=fold)
    J_ref, K_ref = reference_jk(helper, Cl @ fold @ Cr.T)

    np.testing.assert_allclose(J, J_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(K, K_ref, rtol=0, atol=1e-10)


def test_lists_in_lists_out(helper, rng):
    lefts = [rng.standard_normal((NBF, 2)), rng.standard_normal((NBF, 3))]
    rights = [rng.standard_normal((NBF, 2)), rng.standard_normal((NBF, 3))]

    J_list, K_list = integrals.df_compute_jk(helper, lefts, rights)

    assert isinstance(J_list, list) and len(J_list) == 2
    for Cl, Cr, J, K in zip(lefts, rights, J_list, K_list):
        J_one, K_one = integrals.df_compute_jk(helper, Cl, Cr)
        np.testing.assert_array_equal(J, J_one)
        np.testing.assert_array_equal(K, K_one)


def test_use_integrals_patches_the_class_and_puts_it_back():
    original = helper_SAPT.compute_jk

    with integrals.use_integrals("matched"):
        assert helper_SAPT.compute_jk is integrals.df_compute_jk
    assert helper_SAPT.compute_jk is original

    with pytest.raises(RuntimeError):
        with integrals.use_integrals("matched"):
            raise RuntimeError("an interrupted calculation")
    assert helper_SAPT.compute_jk is original

    with integrals.use_integrals("production"):
        assert helper_SAPT.compute_jk is original

    with pytest.raises(ValueError):
        with integrals.use_integrals("exact"):
            pass
