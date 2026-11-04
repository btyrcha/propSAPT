"""Tests of the self-describing density bundles (`prop_sapt.artifacts`)."""

import json
import warnings

import numpy as np
import psi4
import pytest

import prop_sapt.densities as densities_module
from .he_h2 import HE_H2
from prop_sapt import (
    DENSITY_COMPONENTS,
    __version__,
    calc_densities,
    calc_density_matrix,
    load_densities,
)
from prop_sapt.cubes import read_cube_file
from prop_sapt.artifacts import (
    CONVENTIONS,
    DENSITIES_FILENAME,
    FORMAT_VERSION,
    ArtifactError,
    _flatten_tree,
    _unflatten_tree,
    save_densities,
)
from prop_sapt.checks import InvariantWarning


# The default JKFIT basis of aug-cc-pVTZ, which has no He set: Psi4 fills in from
# def2-universal-JKFIT
JKFIT_BLEND = {"AUG-CC-PVTZ-JKFIT", "DEF2-UNIVERSAL-JKFIT"}


def _symmetric(nbf, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(nbf, nbf))
    return matrix + matrix.T


@pytest.fixture(scope="module")
def matrices(prepare_dimer):
    """Synthetic AO matrices keyed like the output of `calc_densities`."""

    nbf = prepare_dimer.nbf
    rho = {
        "pol_A": _symmetric(nbf, 1),
        "pol_B": _symmetric(nbf, 2),
        "exch-ind_S2_A": _symmetric(nbf, 3),
        "exch-ind_S2_B": _symmetric(nbf, 4),
    }
    rho["pol"] = rho["pol_A"] + rho["pol_B"]
    rho["exch-ind_S2"] = rho["exch-ind_S2_A"] + rho["exch-ind_S2_B"]

    return rho


@pytest.fixture()
def bundle(prepare_dimer, matrices, tmp_path):
    """A bundle of the synthetic matrices."""

    return save_densities(tmp_path / DENSITIES_FILENAME, matrices, prepare_dimer)


def _rewrite(path, metadata_changes=None, replace=None, drop=()):
    """Rewrites a bundle with changed metadata entries and replaced or dropped arrays."""

    with np.load(path, allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files if name not in drop}

    if metadata_changes:
        metadata = json.loads(arrays["metadata"].item())
        metadata.update(metadata_changes)
        arrays["metadata"] = np.array(json.dumps(metadata))
    arrays.update(replace or {})

    np.savez_compressed(path, **arrays)


def _assert_same_tree(actual, expected, path="wfn"):
    """Compares trees of dicts, lists, tuples and arrays leaf by leaf."""

    assert type(actual) is type(expected), path
    if isinstance(expected, dict):
        assert list(actual) == list(expected), path
        for key in expected:
            _assert_same_tree(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_same_tree(a, e, f"{path}/{i}")
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype, path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    else:
        assert actual == expected, path


def test_round_trip(prepare_dimer, matrices, bundle):
    densities = load_densities(bundle)

    assert list(densities.matrices) == list(matrices)
    for key, matrix in matrices.items():
        np.testing.assert_array_equal(densities.matrices[key], matrix)
    np.testing.assert_array_equal(densities.total("pol_A"), 2 * matrices["pol_A"])

    assert densities.components == ("pol", "exch-ind_S2")
    assert densities.fragments == prepare_dimer.fragments
    assert densities.path == bundle

    metadata = densities.metadata
    assert metadata["format_version"] == FORMAT_VERSION
    assert metadata["prop_sapt_version"] == __version__
    for key, value in CONVENTIONS.items():
        assert metadata[key] == value
    assert metadata["method"] == {
        "reference": "RHF",
        "functional": None,
        "grac_A": None,
        "grac_B": None,
    }
    assert metadata["basis"]["nbf"] == prepare_dimer.nbf
    assert metadata["basis"]["blend"] == "AUG-CC-PVTZ"
    assert metadata["fitting_bases"]["DF_BASIS_SAPT"] == "AUG-CC-PVQZ-RI"
    assert set(metadata["fitting_bases"]["DF_BASIS_SCF"].split(" + ")) == JKFIT_BLEND
    assert metadata["psi4_options"]["SCF_TYPE"] == "DIRECT"
    assert metadata["geometry_input"] == HE_H2
    assert metadata["wavefunctions"] == ["A", "B"]


def test_records_the_settings_the_dimer_was_built_with(
    prepare_dimer, matrices, tmp_path
):
    """The fitting bases and Psi4 options of a bundle are those the dimer was built
    with, not those set when it is saved."""

    saved_options = psi4.p4util.OptionsState(
        ["BASIS"], ["DF_BASIS_SAPT"], ["DF_BASIS_SCF"], ["SCF_TYPE"]
    )
    try:
        psi4.set_options(
            {
                "basis": "cc-pvdz",
                "df_basis_sapt": "cc-pvdz-ri",
                "df_basis_scf": "cc-pvdz-jkfit",
                "scf_type": "df",
            }
        )
        path = save_densities(tmp_path / "bundle.npz", matrices, prepare_dimer)
    finally:
        saved_options.restore()

    metadata = load_densities(path).metadata
    assert metadata["basis"]["blend"] == "AUG-CC-PVTZ"
    assert metadata["fitting_bases"]["DF_BASIS_SAPT"] == "AUG-CC-PVQZ-RI"
    assert set(metadata["fitting_bases"]["DF_BASIS_SCF"].split(" + ")) == JKFIT_BLEND
    for option, value in {
        "BASIS": "AUG-CC-PVTZ",
        "DF_BASIS_SAPT": "AUG-CC-PVQZ-RI",
        "SCF_TYPE": "DIRECT",
    }.items():
        assert metadata["psi4_options"][option] == value


def test_directory_and_suffix(prepare_dimer, matrices, tmp_path):
    path = save_densities(tmp_path / "sub" / "densities", matrices, prepare_dimer)

    assert path == tmp_path / "sub" / DENSITIES_FILENAME
    assert list(load_densities(tmp_path / "sub").matrices) == list(matrices)


def test_nothing_is_pickled(bundle):
    with np.load(bundle, allow_pickle=False) as npz:
        for name in npz.files:
            assert npz[name].dtype.kind != "O", name


def test_wavefunctions(prepare_dimer, bundle):
    densities = load_densities(bundle)

    for monomer, wfn in (("A", prepare_dimer.wfnA), ("B", prepare_dimer.wfnB)):
        _assert_same_tree(densities.wavefunction_data(monomer), wfn.to_file())

        # twice: `Wavefunction.from_file` changes the data it is given
        for _ in range(2):
            restored = densities.wavefunction(monomer)
            np.testing.assert_array_equal(restored.Ca().np, wfn.Ca().np)
            np.testing.assert_array_equal(restored.Da().np, wfn.Da().np)
            np.testing.assert_array_equal(restored.epsilon_a().np, wfn.epsilon_a().np)
            assert restored.energy() == wfn.energy()

    with pytest.raises(KeyError, match="No wavefunction of monomer 'C'"):
        densities.wavefunction_data("C")


def test_wavefunction_with_unknown_basis(prepare_dimer, bundle, monkeypatch):
    densities = load_densities(bundle)
    data = densities.wavefunction_data("A")
    data["string"]["basisname"] = "PROPSAPT_NO_SUCH_BASIS"
    monkeypatch.setattr(type(densities), "wavefunction_data", lambda self, m: data)

    with pytest.raises(RuntimeError, match="could not rebuild the wavefunction"):
        densities.wavefunction("A")


def test_without_wavefunctions(prepare_dimer, matrices, tmp_path):
    path = save_densities(
        tmp_path / "bundle.npz", matrices, prepare_dimer, wavefunctions=False
    )

    densities = load_densities(path)

    assert densities.metadata["wavefunctions"] == []
    with pytest.raises(KeyError):
        densities.wavefunction("A")


def test_extra_metadata(prepare_dimer, matrices, tmp_path):
    path = save_densities(
        tmp_path / "bundle.npz",
        matrices,
        prepare_dimer,
        metadata={"extra": {"project": "test", "path": tmp_path}},
    )

    assert load_densities(path).metadata["extra"] == {
        "project": "test",
        "path": str(tmp_path),
    }

    with pytest.raises(ValueError, match="written by propSAPT"):
        save_densities(path, matrices, prepare_dimer, metadata={"spin": "total"})


def test_save_checks_matrices(prepare_dimer, tmp_path):
    with pytest.raises(ValueError, match="No density matrices"):
        save_densities(tmp_path / "bundle.npz", {}, prepare_dimer)

    with pytest.raises(ValueError, match="has shape"):
        save_densities(tmp_path / "bundle.npz", {"pol_A": np.eye(3)}, prepare_dimer)


def test_save_warns_for_non_symmetric_matrices(prepare_dimer, matrices, tmp_path):
    """A non-symmetric matrix warns, naming it and pointing at the caller, and is
    saved anyway; loading does not check."""

    skewed = dict(matrices)
    skewed["exch-ind_S2_A"] = matrices["exch-ind_S2_A"].copy()
    skewed["exch-ind_S2_A"][0, 1] += 1e-6

    with pytest.warns(InvariantWarning, match="'exch-ind_S2_A'") as record:
        path = save_densities(tmp_path / "bundle.npz", skewed, prepare_dimer)

    assert len(record) == 1
    assert "'pol_A'" not in str(record[0].message)
    assert record[0].filename == __file__

    with warnings.catch_warnings():
        warnings.simplefilter("error", InvariantWarning)
        densities = load_densities(path)
    np.testing.assert_array_equal(
        densities.matrices["exch-ind_S2_A"], skewed["exch-ind_S2_A"]
    )


def test_checks_against_dimer_geometry_and_basis(prepare_dimer, bundle):
    load_densities(bundle, prepare_dimer)
    load_densities(bundle, geometry=HE_H2)
    load_densities(bundle, geometry=prepare_dimer.fragments)
    load_densities(bundle, basisset=prepare_dimer.dimer_wfn.basisset())

    shifted = HE_H2.replace("1.775500000", "1.775600000")
    with pytest.raises(ArtifactError, match=r"coordinates differ by up to 1\.000e-04"):
        load_densities(bundle, geometry=shifted)
    load_densities(bundle, geometry=shifted, atol=1.0e-3)

    other_basis = psi4.core.BasisSet.build(prepare_dimer.dimer, "ORBITAL", "CC-PVDZ")
    with pytest.raises(ArtifactError, match="'AUG-CC-PVTZ' vs expected 'CC-PVDZ'"):
        load_densities(bundle, basisset=other_basis)


def test_rejects_non_bundles(tmp_path):
    npy = tmp_path / "matrix.npy"
    np.save(npy, np.eye(2))
    with pytest.raises(ArtifactError, match="not an .npz archive"):
        load_densities(npy)

    npz = tmp_path / "plain.npz"
    np.savez(npz, a=np.eye(2))
    with pytest.raises(ArtifactError, match="no metadata"):
        load_densities(npz)


@pytest.mark.parametrize(
    "metadata_changes, message",
    [
        ({"format": "other"}, "not a propSAPT density bundle"),
        ({"format_version": FORMAT_VERSION + 1}, "reads format versions up to"),
        ({"format_version": "1"}, "invalid format version"),
        ({"spin": "total"}, "unknown convention spin='total'"),
        ({"ao_order": "cca"}, "unknown convention ao_order"),
        ({"total_density_factor": 1}, "unknown convention total_density_factor"),
        ({"components": None, "matrices": ["pol_A"]}, "unlisted"),
        ({"geometry": {"units": "angstrom"}}, "invalid geometry metadata"),
    ],
)
def test_rejects_inconsistent_metadata(bundle, metadata_changes, message):
    _rewrite(bundle, metadata_changes=metadata_changes)

    with pytest.raises(ArtifactError, match=message):
        load_densities(bundle)


def test_rejects_missing_metadata_entry(bundle):
    with np.load(bundle, allow_pickle=False) as npz:
        metadata = json.loads(npz["metadata"].item())
    del metadata["basis"]
    _rewrite(bundle, replace={"metadata": np.array(json.dumps(metadata))})

    with pytest.raises(ArtifactError, match="no 'basis' metadata entry"):
        load_densities(bundle)


def test_rejects_wrong_matrices(prepare_dimer, bundle):
    nbf = prepare_dimer.nbf

    _rewrite(bundle, replace={"delta_dm_pol_A": np.eye(nbf - 1)})
    with pytest.raises(ArtifactError, match=r"matrix 'pol_A' has shape"):
        load_densities(bundle)

    _rewrite(bundle, replace={"delta_dm_pol_A": np.eye(nbf)}, drop=("delta_dm_pol",))
    with pytest.raises(ArtifactError, match=r"missing \['pol'\]"):
        load_densities(bundle)


def test_rejects_unlisted_matrix_and_pickled_entries(prepare_dimer, bundle):
    _rewrite(bundle, replace={"delta_dm_other": np.eye(prepare_dimer.nbf)})
    with pytest.raises(ArtifactError, match=r"unlisted \['other'\]"):
        load_densities(bundle)

    _rewrite(
        bundle,
        replace={"delta_dm_pol_A": np.array([{"a": 1}], dtype=object)},
        drop=("delta_dm_other",),
    )
    with pytest.raises(ArtifactError, match="entry 'delta_dm_pol_A' cannot be read"):
        load_densities(bundle)


def test_flatten_round_trip(tmp_path):
    tree = {
        "array": np.arange(6.0).reshape(2, 3),
        "strings": np.array(["He", "H"]),
        "nested": {"none": None, "flag": True, "count": np.int64(3), "x": 0.1},
        "tuple": (1, 2, np.array([True, False])),
        "list": [1.5, "text", np.array([1, 2])],
        "key with space": np.ones(1),
    }

    manifest, arrays = _flatten_tree(tree, "tree")
    np.savez(tmp_path / "tree.npz", manifest=np.array(json.dumps(manifest)), **arrays)

    with np.load(tmp_path / "tree.npz", allow_pickle=False) as npz:
        restored = _unflatten_tree(json.loads(npz["manifest"].item()), npz)

    expected = dict(tree, nested=dict(tree["nested"], count=3))
    _assert_same_tree(restored, expected)


@pytest.mark.parametrize(
    "tree, message",
    [
        ({"a": object()}, "holds a object"),
        ({"a": {1, 2}}, "holds a set"),
        ({"a": np.array([None])}, "array of object"),
        ({1: 2}, "Keys should be strings"),
        ({"__array__": "x"}, "reserved"),
    ],
)
def test_flatten_refuses_what_needs_pickling(tree, message):
    with pytest.raises(TypeError, match=message):
        _flatten_tree(tree, "tree")


# calc_densities writes bundles; the components are computed once and reused


@pytest.fixture(scope="module")
def component_matrices(prepare_dimer):
    """The AO density components of both monomers."""

    return {
        monomer: calc_density_matrix(prepare_dimer, monomer) for monomer in ("A", "B")
    }


@pytest.fixture()
def cached_components(monkeypatch, component_matrices):
    """Makes `calc_densities` reuse the components instead of computing them again."""

    def cached(mol, monomer, orbital_basis="AO"):
        assert orbital_basis == "AO"
        return {key: rho.copy() for key, rho in component_matrices[monomer].items()}

    monkeypatch.setattr(densities_module, "calc_density_matrix", cached)


@pytest.fixture()
def cached_mo_components(monkeypatch, prepare_dimer, component_matrices):
    """Makes `calc_densities` in the MO basis reuse the components, transformed into the
    MO basis of their monomer: C^T S rho S C, the inverse of C rho C^T as C^T S C = 1."""

    overlap = np.asarray(prepare_dimer.mints.ao_overlap())
    coefficients = {"A": prepare_dimer.C_A, "B": prepare_dimer.C_B}
    mo_components = {
        monomer: {
            key: coefficients[monomer].T @ overlap @ rho @ overlap @ coefficients[monomer]
            for key, rho in components.items()
        }
        for monomer, components in component_matrices.items()
    }

    def cached(mol, monomer, orbital_basis="AO"):
        assert orbital_basis == "MO"
        return {key: rho.copy() for key, rho in mo_components[monomer].items()}

    monkeypatch.setattr(densities_module, "calc_density_matrix", cached)

    return mo_components


ALL_KEYS = {
    f"{component}{suffix}"
    for component in DENSITY_COMPONENTS
    for suffix in ("_A", "_B", "")
}
MONOMER_KEYS = {
    f"{component}{suffix}" for component in DENSITY_COMPONENTS for suffix in ("_A", "_B")
}


def test_calc_densities_writes_bundle_and_cubes(
    prepare_dimer, cached_components, tmp_path
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", InvariantWarning)  # the components are symmetric
        rho = calc_densities(
            prepare_dimer,
            save_cubes=True,
            cubes_dir=str(tmp_path / "cubes"),
            save_matrices=True,
            matrices_dir=str(tmp_path / "matrices"),
            metadata={"project": "test"},
        )

    assert set(rho) == ALL_KEYS

    densities = load_densities(tmp_path / "matrices", prepare_dimer, geometry=HE_H2)
    assert densities.components == DENSITY_COMPONENTS
    assert list(densities.matrices) == list(rho)
    for key, matrix in rho.items():
        np.testing.assert_array_equal(densities.matrices[key], matrix)

    assert densities.metadata["extra"] == {"project": "test"}
    assert densities.metadata["cubes"] == {
        "files": [f"../cubes/delta_dm_{key}.cube" for key in rho],
        "density": "total",
    }

    cubes = sorted(path.name for path in (tmp_path / "cubes").iterdir())
    assert cubes == sorted(f"delta_dm_{key}.cube" for key in rho)
    cube = read_cube_file(str(tmp_path / "cubes" / "delta_dm_pol_A.cube"))
    assert (
        cube.comment1
        == "propSAPT delta_dm_pol_A: total density (2 x alpha-spin matrix)"
    )


def test_calc_densities_in_mo_basis_saves_ao_matrices(
    prepare_dimer, component_matrices, cached_mo_components, tmp_path
):
    """In the MO basis only the monomer matrices are returned: each monomer has its own
    MO basis, not orthogonal to the other's, so there is no dimer sum. The matrices are
    saved in the AO basis, transformed with the orbitals of their own monomer."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", InvariantWarning)
        rho = calc_densities(
            prepare_dimer,
            orbital_basis="MO",
            save_cubes=True,
            cubes_dir=str(tmp_path / "cubes"),
            save_matrices=True,
            matrices_dir=str(tmp_path / "matrices"),
        )

    assert set(rho) == MONOMER_KEYS
    for monomer, components in cached_mo_components.items():
        for component, matrix in components.items():
            np.testing.assert_array_equal(rho[f"{component}_{monomer}"], matrix)

    densities = load_densities(tmp_path / "matrices", prepare_dimer)
    assert densities.components == DENSITY_COMPONENTS
    assert list(densities.matrices) == list(rho)
    for monomer, components in component_matrices.items():
        for component, matrix in components.items():
            key = f"{component}_{monomer}"
            np.testing.assert_allclose(
                densities.matrices[key], matrix, rtol=0, atol=1e-10, err_msg=key
            )

    cubes = sorted(path.name for path in (tmp_path / "cubes").iterdir())
    assert cubes == sorted(f"delta_dm_{key}.cube" for key in rho)


def test_calc_densities_npy_format_is_deprecated(
    prepare_dimer, cached_components, tmp_path
):
    with pytest.warns(FutureWarning, match=r'matrix_format="npy"\) is deprecated'):
        rho = calc_densities(
            prepare_dimer,
            save_matrices=True,
            matrices_dir=str(tmp_path),
            matrix_format="npy",
        )

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [f"delta_dm_{key}.npy" for key in rho] + ["wfn_mon_A.npy", "wfn_mon_B.npy"]
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "delta_dm_pol_A.npy"), rho["pol_A"]
    )


@pytest.mark.parametrize(
    "compress, written", [(True, "densities.npz"), (False, "delta_dm_pol.npy")]
)
def test_calc_densities_compress_matrices_is_deprecated(
    prepare_dimer, cached_components, tmp_path, compress, written
):
    with pytest.warns(FutureWarning) as record:
        rho = calc_densities(
            prepare_dimer,
            save_matrices=True,
            matrices_dir=str(tmp_path),
            compress_matrices=compress,
        )

    # False also selects the deprecated .npy format
    messages = [str(warning.message) for warning in record]
    assert any("`compress_matrices` is deprecated" in m for m in messages)
    assert any('matrix_format="npy") is deprecated' in m for m in messages) == (
        not compress
    )

    # the monomer wavefunctions are saved, not added to the returned matrices
    assert set(rho) == ALL_KEYS
    assert (tmp_path / written).is_file()


def test_calc_densities_rejects_bad_formats(prepare_dimer, cached_components, tmp_path):
    with pytest.raises(TypeError, match="use `matrix_format` alone"):
        calc_densities(prepare_dimer, matrix_format="npz", compress_matrices=True)

    with pytest.raises(ValueError, match="`matrix_format` should be one of"):
        calc_densities(prepare_dimer, save_matrices=True, matrix_format="hdf5")
