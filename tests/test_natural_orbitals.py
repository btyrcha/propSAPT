"""Tests of the natural orbitals of the density changes (`prop_sapt.natural_orbitals`)."""

import numpy as np
import pytest

from prop_sapt import DENSITY_COMPONENTS, calc_densities, density_natural_orbitals
from prop_sapt.artifacts import DENSITIES_FILENAME, load_densities, save_densities
from prop_sapt.natural_orbitals import NaturalOrbitals, natural_orbitals


@pytest.fixture(scope="module")
def he_h2_densities(prepare_dimer):
    """The AO density changes of He-H2, all keys, computed once."""

    return calc_densities(prepare_dimer)


@pytest.fixture(scope="module")
def he_h2_orbitals(prepare_dimer, he_h2_densities):
    """Their natural orbitals, all keys."""

    return density_natural_orbitals(he_h2_densities, prepare_dimer)


def test_natural_orbitals_rebuild_every_density(he_h2_densities, he_h2_orbitals):
    """sum_k n_k c_k c_k^T is the density change, for every key, also the dimer ones
    (both monomers' orbitals together)."""

    assert set(he_h2_orbitals) == set(he_h2_densities)
    for key, matrix in he_h2_densities.items():
        np.testing.assert_allclose(
            he_h2_orbitals[key].density(), matrix, rtol=0, atol=1e-12, err_msg=key
        )


def test_natural_orbitals_are_orthonormal_and_sorted(prepare_dimer, he_h2_orbitals):
    """A monomer's natural orbitals are orthonormal and sorted by decreasing occupation;
    their occupations sum to Tr(dP S) = 0."""

    overlap = prepare_dimer.S
    for key, orbitals in he_h2_orbitals.items():
        assert np.all(np.diff(orbitals.occupations) <= 0.0), key
        assert abs(orbitals.occupations.sum()) < 1e-12, key
        if key.endswith(("_A", "_B")):
            assert len(orbitals) == prepare_dimer.nmo
            assert set(orbitals.monomers) == {key[-1]}
            gram = orbitals.coefficients.T @ overlap @ orbitals.coefficients
            np.testing.assert_allclose(
                gram, np.eye(len(orbitals)), rtol=0, atol=1e-9, err_msg=key
            )


def test_dimer_key_lists_both_monomers(prepare_dimer, he_h2_orbitals):
    """A dimer key has the orbitals of A and of B, by decreasing occupation."""

    for component in DENSITY_COMPONENTS:
        orbitals = he_h2_orbitals[component]
        assert len(orbitals) == 2 * prepare_dimer.nmo
        for monomer in ("A", "B"):
            mine = orbitals.occupations[orbitals.monomers == monomer]
            np.testing.assert_allclose(
                mine,
                he_h2_orbitals[f"{component}_{monomer}"].occupations,
                rtol=0,
                atol=1e-14,
            )


@pytest.mark.parametrize("component", ["pol", "exch"])
@pytest.mark.parametrize("monomer", ["A", "B"])
def test_first_order_orbitals_come_in_pairs(
    prepare_dimer, he_h2_orbitals, component, monomer
):
    """First-order polarisation and exchange mix occupied with virtual orbitals only, so
    their occupations are +/- pairs, one per occupied orbital, and the rest zero."""

    occupations = he_h2_orbitals[f"{component}_{monomer}"].occupations
    ndocc = prepare_dimer.ndocc_A if monomer == "A" else prepare_dimer.ndocc_B
    scale = np.abs(occupations).max()

    np.testing.assert_allclose(
        occupations, -occupations[::-1], rtol=0, atol=1e-12 * scale
    )
    assert np.sum(np.abs(occupations) > 1e-10 * scale) == 2 * ndocc


def test_cutoff_drops_small_occupations(prepare_dimer, he_h2_densities, he_h2_orbitals):
    """`cutoff` drops the orbitals with |n| up to it, and keeps the order."""

    kept = density_natural_orbitals(
        he_h2_densities, prepare_dimer, keys=["Disp_A"], cutoff=1e-6
    )["Disp_A"]
    full = he_h2_orbitals["Disp_A"]

    assert 0 < len(kept) < len(full)
    np.testing.assert_array_equal(
        kept.occupations, full.occupations[np.abs(full.occupations) > 1e-6]
    )


def test_bundle_gives_the_same_orbitals(
    tmp_path, prepare_dimer, he_h2_densities, he_h2_orbitals
):
    """A bundle's monomer wavefunctions give the same natural orbitals as the dimer."""

    path = save_densities(tmp_path / DENSITIES_FILENAME, he_h2_densities, prepare_dimer)
    from_bundle = density_natural_orbitals(load_densities(path), keys=["pol", "Disp_B"])

    for key, orbitals in from_bundle.items():
        np.testing.assert_allclose(
            orbitals.occupations,
            he_h2_orbitals[key].occupations,
            rtol=0,
            atol=1e-14,
            err_msg=key,
        )
        np.testing.assert_allclose(
            orbitals.density(), he_h2_densities[key], rtol=0, atol=1e-12, err_msg=key
        )


def test_bundle_without_wavefunctions_is_refused(
    tmp_path, prepare_dimer, he_h2_densities
):
    path = save_densities(
        tmp_path / DENSITIES_FILENAME,
        he_h2_densities,
        prepare_dimer,
        wavefunctions=False,
    )

    with pytest.raises(ValueError, match="no wavefunction of monomer"):
        density_natural_orbitals(load_densities(path))


def test_density_natural_orbitals_errors(he_h2_densities, prepare_dimer):
    with pytest.raises(TypeError, match="`dimer` is needed"):
        density_natural_orbitals(he_h2_densities)

    with pytest.raises(KeyError, match="needs 'pol_B'"):
        density_natural_orbitals(
            {"pol_A": he_h2_densities["pol_A"]}, prepare_dimer, keys=["pol"]
        )


def test_density_outside_the_orbitals_is_refused(prepare_dimer, he_h2_densities):
    """With fewer orbitals than basis functions, a matrix outside their span is not a
    density change of the monomer."""

    orbitals = prepare_dimer.C_A[:, :-5]
    rng = np.random.default_rng(3)
    outside = rng.normal(size=orbitals.shape[:1] * 2)
    outside = outside + outside.T

    with pytest.raises(ValueError, match="does not lie in the span"):
        natural_orbitals(outside, orbitals, prepare_dimer.S)

    with pytest.raises(ValueError, match="Shapes do not match"):
        natural_orbitals(he_h2_densities["pol_A"][:-1], orbitals, prepare_dimer.S)


def test_select_keeps_fields_together():
    orbitals = NaturalOrbitals(
        np.array([0.3, 1e-9, -0.3]),
        np.eye(3),
        np.array(["A", "B", "A"]),
    )

    kept = orbitals.select(1e-6)

    np.testing.assert_array_equal(kept.occupations, [0.3, -0.3])
    np.testing.assert_array_equal(kept.coefficients, np.eye(3)[:, [0, 2]])
    np.testing.assert_array_equal(kept.monomers, ["A", "A"])
