"""
Which two-electron integrals propSAPT runs on, for the finite-field comparison.

propSAPT's energy and property paths are not fed entirely the same integrals. The J/K
terms of the energies -- ELST1, EXCH1, and every omega through ``J_A``/``J_B`` -- come
from Psi4's JK object via ``helper_SAPT.compute_jk``, which under ``scf_type direct``
is ``DirectJK`` and therefore exact. So do the first-order exchange potentials
(``omega_exch*`` in ``sinfinity.py``), but the second-order contractions still take
some J/K-type integrals from the ``DF_BASIS_SAPT`` tensors. Where the paths differ, the
analytic property is not the derivative of the energy the finite-field path
differentiates, and the gap -- the fitting error of the property -- lands on top of
whatever is being validated: ~1e-9 on the second-order terms at the ``fixture``
settings, up to 7e-8 for the HF dimer at aug-cc-pVDZ. On first-order exchange it was
6.1e-8 at the ``fixture`` settings and 1.9e-6 for the HF dimer, before the exchange
potentials moved onto the JK object.

``matched`` reroutes ``compute_jk`` through the ``DF_BASIS_SAPT`` tensors (``Qpq``)
for both paths, so that every J/K -- energy, exchange potential or contraction --
comes from the same integrals, and what remains is only whether the contractions are
the derivative of the energy expressions: the first-order terms agree to 1e-11 to
1e-10. ``production`` leaves propSAPT as it is and measures the gap itself.

What ``matched`` does not reach is the monomer SCF and CPSCF, which keep Psi4's own
JK: the finite-field path relaxes the orbitals with it and the CPSCF Hessian applies
it, while the second-order contractions take the monomers' own integrals from the
fitting tensors. Density fitting the SCF on the same basis as well moved the
second-order deviations by at most 2e-9 on He-H2, and raised the first-order floor
to ~1e-9, so the SCF is left conventional.
"""

import importlib
from contextlib import contextmanager

import numpy as np

MODES = ("matched", "production")

DESCRIPTIONS = {
    "matched": "both paths on the DF_BASIS_SAPT integrals (compute_jk rerouted)",
    "production": "as shipped: J/K from Psi4's JK object, second-order "
    "contractions partly on DF_BASIS_SAPT",
}

#: resolved when the switch is entered, not at import: the harness swaps whole copies
#: of the package in and out, and the class to patch is the active copy's
_HELPER_MODULE = "prop_sapt.helper_SAPT_DF"


def df_compute_jk(self, Cleft, Cright, tensor=None):
    """``helper_SAPT.compute_jk`` evaluated on the ``DF_BASIS_SAPT`` tensors.

    Same interface and conventions as the original: ``D = Cl Cr^T``,
    ``J_mn = (mn|ls) D_ls`` and ``K_mn = (ml|ns) D_ls``, with ``tensor`` folded into
    whichever side has fewer columns, and lists in giving lists out.
    """

    lefts, rights, tensors = Cleft, Cright, tensor
    single = False
    if not isinstance(lefts, (list, tuple)):
        lefts, single = [lefts], True
    if not isinstance(rights, (list, tuple)):
        rights, single = [rights], True
    if tensors is not None and not isinstance(tensors, (list, tuple)):
        tensors, single = [tensors], True
    if len(lefts) != len(rights):
        raise ValueError("Cleft list is not the same length as Cright list")

    Qpq = self.Qpq
    nbf = Qpq.shape[1]

    J_list, K_list = [], []
    for num, (Cl, Cr) in enumerate(zip(lefts, rights)):
        if Cl.shape[1] == 0 or Cr.shape[1] == 0:
            J_list.append(np.zeros((nbf, nbf)))
            K_list.append(np.zeros((nbf, nbf)))
            continue

        if tensors is not None:
            fold = tensors[num]
            if fold.shape != (Cl.shape[1], Cr.shape[1]):
                raise ValueError(
                    f"tensor shape {fold.shape} does not match Cl ({Cl.shape[1]}) "
                    f"/ Cr ({Cr.shape[1]})"
                )
            if Cl.shape[1] < Cr.shape[1]:
                Cl = Cl @ fold
            else:
                Cr = Cr @ fold.T

        density = Cl @ Cr.T
        gamma = np.tensordot(Qpq, density, axes=([1, 2], [0, 1]))
        J_list.append(np.tensordot(gamma, Qpq, axes=(0, 0)))

        QCl = Qpq @ Cl
        QCr = Qpq @ Cr
        K_list.append(np.tensordot(QCl, QCr, axes=([0, 2], [0, 2])))

    if single:
        return J_list[0], K_list[0]
    return J_list, K_list


@contextmanager
def use_integrals(mode):
    """Run the enclosed propSAPT calls on ``mode``'s integrals.

    Patches the class, so every ``Dimer`` built inside the block is affected and none
    built outside it is. Enter it after the harness has activated a package, so the
    copy being patched is the one being evaluated.
    """

    if mode not in MODES:
        raise ValueError(f"unknown integrals mode {mode!r}; expected one of {MODES}")
    if mode == "production":
        yield
        return

    helper = importlib.import_module(_HELPER_MODULE).helper_SAPT
    original = helper.__dict__["compute_jk"]
    helper.compute_jk = df_compute_jk
    try:
        yield
    finally:
        helper.compute_jk = original
