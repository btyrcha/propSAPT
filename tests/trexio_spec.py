"""
An evaluator of the orbitals in a TREXIO file, written from the TREXIO specification and
independent of Psi4: the tests of `prop_sapt.trexio_export` compare it with Psi4.

- **AOs:** an AO is ``N'_i P(r - R) N_s sum_k f_k a_k exp(-g_k |r - R|^2)``, built from
  ``ao.normalization``, ``basis.shell_factor``, ``basis.prim_factor``,
  ``basis.coefficient`` and ``basis.exponent``.
- **Cartesian shells:** P is x^a y^b z^c, in alphabetical order.
- **Spherical shells:** P is the real solid harmonics, in the order m = 0, +1, -1, +2, -2,
  ..., Racah-normalised and with the spec's phases (S_1^{+1} = x, S_1^{-1} = y).
- **Where the harmonics come from:** they are generated from Helgaker, Jorgensen and Olsen,
  *Molecular Electronic-Structure Theory*, eqs. 6.4.47-6.4.50. Up to l = 4 these agree to
  4e-16 with trexio-tools' tables of the spec (``converters/cart_sphe.py`` times
  ``normalization``).
"""

from math import comb, factorial, sqrt

import numpy as np


def cartesian_powers(l: int) -> list[tuple[int, int, int]]:
    """The (a, b, c) of the Cartesian AOs of a shell, in alphabetical order."""

    return [(a, b, l - a - b) for a in range(l, -1, -1) for b in range(l - a, -1, -1)]


def spherical_order(l: int) -> list[int]:
    """The m of the spherical AOs of a shell: 0, +1, -1, ..., +l, -l."""

    return [0] + [sign * k for k in range(1, l + 1) for sign in (1, -1)]


def solid_harmonic(l: int, m: int) -> dict[tuple[int, int, int], float]:
    """The real solid harmonic S_lm as Cartesian coefficients {(a, b, c): coefficient}."""

    am = abs(m)
    norm = sqrt(2 * factorial(l + am) * factorial(l - am) / (2 if m == 0 else 1)) / (
        2**am * factorial(l)
    )
    two_vm = 0 if m >= 0 else 1
    coefficients = {}
    for t in range((l - am) // 2 + 1):
        for u in range(t + 1):
            for two_v in range(two_vm, am + 1, 2):
                c = (
                    (-1) ** (t + (two_v - two_vm) // 2)
                    * 0.25**t
                    * comb(l, t)
                    * comb(l - t, am + t)
                    * comb(t, u)
                    * comb(am, two_v)
                )
                power = (2 * t + am - 2 * u - two_v, 2 * u + two_v, l - 2 * t - am)
                coefficients[power] = coefficients.get(power, 0.0) + norm * c
    return coefficients


def read_file(path) -> dict:
    """The arrays of a TREXIO file that its orbitals need."""

    import trexio

    handle = trexio.File(str(path), mode="r", back_end=trexio.TREXIO_AUTO)
    try:
        data = {
            "coord": trexio.read_nucleus_coord(handle),
            "charge": trexio.read_nucleus_charge(handle),
            "label": trexio.read_nucleus_label(handle),
            "shell_center": trexio.read_basis_nucleus_index(handle),
            "shell_am": trexio.read_basis_shell_ang_mom(handle),
            "shell_factor": trexio.read_basis_shell_factor(handle),
            "prim_shell": np.asarray(trexio.read_basis_shell_index(handle)),
            "exponent": trexio.read_basis_exponent(handle),
            "coefficient": trexio.read_basis_coefficient(handle)
            * trexio.read_basis_prim_factor(handle),
            "cartesian": trexio.read_ao_cartesian(handle),
            "ao_shell": trexio.read_ao_shell(handle),
            "ao_normalization": trexio.read_ao_normalization(handle),
            "description": trexio.read_metadata_description(handle),
        }
        if trexio.has_mo_num(handle):
            data["mo_type"] = trexio.read_mo_type(handle)
            data["mo_coefficient"] = trexio.read_mo_coefficient(handle)
            data["mo_occupation"] = trexio.read_mo_occupation(handle)
            data["mo_spin"] = trexio.read_mo_spin(handle)
    finally:
        handle.close()

    return data


def ao_values(data: dict, points: np.ndarray) -> np.ndarray:
    """The AOs of a file read by `read_file` at `points` (bohr), shape (npoints, nao)."""

    values = np.zeros((len(points), len(data["ao_shell"])))
    ao = 0
    for shell, l in enumerate(data["shell_am"]):
        d = points - data["coord"][data["shell_center"][shell]]
        r2 = np.einsum("pi,pi->p", d, d)
        prims = np.where(data["prim_shell"] == shell)[0]
        radial = data["shell_factor"][shell] * (
            np.exp(-np.outer(r2, data["exponent"][prims])) @ data["coefficient"][prims]
        )
        if data["cartesian"]:
            polynomials = [{power: 1.0} for power in cartesian_powers(l)]
        else:
            polynomials = [solid_harmonic(l, m) for m in spherical_order(l)]
        for polynomial in polynomials:
            assert data["ao_shell"][ao] == shell
            angular = sum(
                c * d[:, 0] ** a * d[:, 1] ** b * d[:, 2] ** e
                for (a, b, e), c in polynomial.items()
            )
            values[:, ao] = data["ao_normalization"][ao] * radial * angular
            ao += 1

    return values


def orbital_values(data: dict, points: np.ndarray) -> np.ndarray:
    """The orbitals of a file at `points`, shape (npoints, nmo)."""

    return ao_values(data, points) @ data["mo_coefficient"].T


def difference_density(data: dict, points: np.ndarray) -> np.ndarray:
    """The alpha - beta density of a file at `points`: the density change it holds."""

    signs = np.where(np.asarray(data["mo_spin"]) == 0, 1.0, -1.0)
    return orbital_values(data, points) ** 2 @ (signs * data["mo_occupation"])
