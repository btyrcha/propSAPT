import pytest

import numpy as np
import psi4

from prop_sapt.cubes import (
    Cube,
    add_cubes,
    cosine_similarity,
    read_cube_file,
    subtract_cubes,
)

GRID_STEP = 1.0  # coarse grid, the tests check file handling not the cube values


@pytest.fixture(scope="module")
def density_A(prepare_dimer):
    """Monomer A density matrix in AO basis."""

    return 2 * np.asarray(prepare_dimer.wfnA.Da())  # RHF spin


def test_save_cube_single_filename_for_many_matrices(
    prepare_dimer, density_A, tmp_path
):
    """A single filename must not be zipped character-wise against a list of cubes."""

    with pytest.raises(ValueError):
        prepare_dimer.save_cube(
            [density_A, density_A],
            filename=str(tmp_path / "density.cube"),
            grid_step=GRID_STEP,
        )

    assert list(tmp_path.iterdir()) == []


def test_save_cube_filename_count_mismatch(prepare_dimer, density_A, tmp_path):
    """Every matrix needs its own filename, extra cubes must not be dropped silently."""

    with pytest.raises(ValueError):
        prepare_dimer.save_cube(
            [density_A] * 3,
            filename=[tmp_path / "a.cube", tmp_path / "b.cube"],
            grid_step=GRID_STEP,
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "as_list", [True, False], ids=["one-element-lists", "bare-matrix-and-path"]
)
def test_save_cube_single_matrix(prepare_dimer, density_A, tmp_path, as_list):
    """A single matrix is saved both as a one-element list and as a bare array."""

    fname = tmp_path / "density.cube"

    if as_list:
        prepare_dimer.save_cube([density_A], filename=[fname], grid_step=GRID_STEP)
    else:
        prepare_dimer.save_cube(density_A, filename=fname, grid_step=GRID_STEP)

    cube = read_cube_file(str(fname))
    assert cube.volumetric_data.shape == (cube.n_x, cube.n_y, cube.n_z)


def _grid(x, y, z):
    """Grid dictionary in the `prepare_grid` format from explicit axis points."""

    grid = {}
    for axis, points in zip("xyz", (x, y, z)):
        points = np.asarray(points, dtype=float)
        grid[axis] = points
        grid[f"n_{axis}"] = points.size
        grid[f"step_{axis}"] = points[1] - points[0] if points.size > 1 else 1.0

    return grid


def _psi4_esp(dimer, density, points):
    """Electronic ESP at `points` straight from Psi4 analytic integrals."""

    mints = psi4.core.MintsHelper(dimer.get_psi4_basisset())
    esp = mints.electrostatic_potential_value(
        psi4.core.Vector.from_array(np.ones(len(points))),
        psi4.core.Matrix.from_array(np.asarray(points, dtype=float)),
        psi4.core.Matrix.from_array(density),
    )

    return np.asarray(esp)


# axes of different lengths, so a transposed grid cannot pass
ESP_GRID = _grid(np.linspace(-4.0, 4.0, 5), np.linspace(-1.0, 1.5, 4), [-0.5, 0.0, 0.5])


def test_esp_cube_grid_order(prepare_dimer, density_A):
    """Every voxel of an ESP cube holds the potential at its own grid point."""

    cube = prepare_dimer.make_cube(density_A, obj_type="esp", grid=ESP_GRID)

    points = np.stack(
        np.meshgrid(ESP_GRID["x"], ESP_GRID["y"], ESP_GRID["z"], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    expected = _psi4_esp(prepare_dimer, density_A, points).reshape(5, 4, 3)

    np.testing.assert_allclose(cube.volumetric_data, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("direction", [(-1, 0, 0), (0, 1, 0), (0, 0, 1)], ids="xyz")
def test_esp_cube_gauss_law(prepare_dimer, density_A, direction):
    """Far from monomer A its electrons look like a point charge of -N_elec."""

    n_elec = 2 * prepare_dimer.ndocc_A
    distance = 25.0
    point = np.array([-2.5, 0.0, 0.0]) + distance * np.array(direction)  # from He

    cube = prepare_dimer.make_cube(
        density_A, obj_type="esp", grid=_grid([point[0]], [point[1]], [point[2]])
    )

    assert cube.volumetric_data.item() == pytest.approx(-n_elec / distance, abs=1e-6)


def test_esp_cube_nuclei(prepare_dimer, density_A):
    """`esp_nuclei` adds sum_A Z_A / |r - R_A|, and stays finite on a nucleus."""

    # He nucleus at (-2.5, 0, 0) is a grid point
    grid = _grid([-3.0, -2.5, -2.0, 2.0], [-0.5, 0.0, 0.5], [0.0, 0.5])

    electronic = prepare_dimer.make_cube(density_A, obj_type="esp", grid=grid)
    total = prepare_dimer.make_cube(
        density_A, obj_type="esp", grid=grid, esp_nuclei=True
    )

    mol = prepare_dimer.get_psi4_molecule()
    nuclei = np.asarray(mol.geometry())
    charges = np.array([mol.Z(i) for i in range(mol.natom())])

    points = np.stack(
        np.meshgrid(grid["x"], grid["y"], grid["z"], indexing="ij"), axis=-1
    )
    dist = np.linalg.norm(points[..., None, :] - nuclei, axis=-1)
    dist[dist < 1e-10] = np.inf
    expected_nuclear = np.sum(charges / dist, axis=-1)

    assert np.all(np.isfinite(total.volumetric_data))
    np.testing.assert_allclose(
        total.volumetric_data - electronic.volumetric_data,
        expected_nuclear,
        rtol=1e-12,
        atol=1e-12,
    )


def test_density_and_orbital_cubes_match_compute_phi(prepare_dimer, density_A):
    """Every voxel holds the value at its own grid point, as from per-point `compute_phi`."""

    coeffs = np.asarray(prepare_dimer.wfnA.Ca())
    basisset = prepare_dimer.get_psi4_basisset()

    # an orbital column (nbf, 1), as in examples/visualise_orbital.ipynb, and a vector
    cubes = prepare_dimer.make_cube(
        [density_A, coeffs[:, [0]], coeffs[:, 1]],
        obj_type=["density", "orbital", "orbital"],
        grid=ESP_GRID,
        chunk_size=7,
    )

    points = np.stack(
        np.meshgrid(ESP_GRID["x"], ESP_GRID["y"], ESP_GRID["z"], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    phi = np.array([np.asarray(basisset.compute_phi(*p)).reshape(-1) for p in points])
    expected = [
        np.einsum("pi,ij,pj->p", phi, density_A, phi),
        phi @ coeffs[:, 0],
        phi @ coeffs[:, 1],
    ]

    for cube, values in zip(cubes, expected):
        np.testing.assert_allclose(
            cube.volumetric_data, values.reshape(5, 4, 3), rtol=1e-12, atol=1e-14
        )


def test_orbital_cube_rejects_several_columns(prepare_dimer):
    """One cube per orbital, several columns in one entry are refused."""

    coeffs = np.asarray(prepare_dimer.wfnA.Ca())

    with pytest.raises(ValueError):
        prepare_dimer.make_cube(coeffs[:, :2], obj_type="orbital", grid=ESP_GRID)


def test_esp_cube_mixed_with_density(prepare_dimer, density_A):
    """Requesting ESP alongside a density leaves the density cube unchanged."""

    density_only = prepare_dimer.make_cube(density_A, obj_type="density", grid=ESP_GRID)
    mixed = prepare_dimer.make_cube(
        [density_A, density_A], obj_type=["esp", "density"], grid=ESP_GRID
    )

    np.testing.assert_array_equal(
        mixed[1].volumetric_data, density_only.volumetric_data
    )
    assert "ESP" in mixed[0].comment2


def test_dimer_cube_in_the_dimers_basis_set(prepare_dimer, density_A):
    """A dimer's cubes are evaluated in its own basis set, not in the one the `BASIS`
    option names now; `save_cube`, `calc_densities` and the runner make them so."""

    expected = prepare_dimer.make_cube(density_A, grid=ESP_GRID)

    saved_options = psi4.p4util.OptionsState(["BASIS"])
    try:
        psi4.set_options({"basis": "cc-pvdz"})
        cube = prepare_dimer.make_cube(density_A, grid=ESP_GRID)
    finally:
        saved_options.restore()

    np.testing.assert_array_equal(cube.volumetric_data, expected.volumetric_data)


# --- Cube objects without a dimer: construction, arithmetic, integrals, file I/O ---

# skewed cell, voxel volume = det([x, y, z]) = 0.2 * 0.3 * 0.25
X_VECTOR = np.array([0.2, 0.0, 0.0])
Y_VECTOR = np.array([0.05, 0.3, 0.0])
Z_VECTOR = np.array([0.0, 0.1, 0.25])
VOXEL_VOLUME = 0.015
TOY_SHAPE = (4, 3, 5)


def _toy_cube(seed=0, shape=TOY_SHAPE, origin=(-1.0, 0.5, 2.0), **kwargs):
    """A small cube with random data on the skewed cell, without counts given."""

    data = np.random.default_rng(seed).standard_normal(shape)
    fields = dict(
        comment1="toy cube",
        comment2=f"seed {seed}",
        origin=np.array(origin),
        atoms=[
            (2, 0.0, np.array([-0.5, 0.0, 0.0])),
            (1, 0.0, np.array([0.7, 0.0, 0.1])),
        ],
        x_vector=X_VECTOR.copy(),
        y_vector=Y_VECTOR.copy(),
        z_vector=Z_VECTOR.copy(),
        volumetric_data=data,
    )
    fields.update(kwargs)

    return Cube(**fields)


def test_cube_counts_from_data():
    """Counts not given are read off the atoms and the data."""

    cube = _toy_cube()

    assert (cube.n_x, cube.n_y, cube.n_z) == TOY_SHAPE
    assert cube.n_atoms == 2


def test_cube_all_counts_given():
    """The full constructor call, counts included, still works."""

    toy = _toy_cube()
    cube = Cube(
        comment1=toy.comment1,
        comment2=toy.comment2,
        origin=toy.origin,
        n_atoms=toy.n_atoms,
        atoms=toy.atoms,
        n_x=toy.n_x,
        n_y=toy.n_y,
        n_z=toy.n_z,
        x_vector=toy.x_vector,
        y_vector=toy.y_vector,
        z_vector=toy.z_vector,
        volumetric_data=toy.volumetric_data,
    )

    assert (cube.n_x, cube.n_y, cube.n_z, cube.n_atoms) == (*TOY_SHAPE, 2)
    # arrays are taken as they are, not copied
    assert cube.volumetric_data is toy.volumetric_data


@pytest.mark.parametrize(
    "counts",
    [{"n_x": 5}, {"n_y": 2}, {"n_z": 1}, {"n_atoms": 3}],
    ids=lambda c: next(iter(c)),
)
def test_cube_counts_must_match(counts):
    """A given count that disagrees with the data is refused."""

    with pytest.raises(ValueError):
        _toy_cube(**counts)


@pytest.mark.parametrize(
    "field, value",
    [
        ("origin", np.zeros(2)),
        ("x_vector", np.zeros((3, 1))),
        ("volumetric_data", np.zeros((4, 3))),
        ("volumetric_data", np.zeros((4, 0, 5))),
        ("atoms", [(1, 0.0, np.zeros(2))]),
    ],
)
def test_cube_rejects_bad_shapes(field, value):
    """Arrays of the wrong shape are refused at construction."""

    with pytest.raises(ValueError):
        _toy_cube(**{field: value})


def test_cube_missing_field():
    """Required fields have no defaults."""

    with pytest.raises(TypeError):
        Cube(origin=np.zeros(3), atoms=[])


def test_cube_converts_arrays():
    """Lists are converted to float arrays, so arithmetic on fields works."""

    cube = _toy_cube(origin=[0, 1, 2], atoms=[(1, 0, [0, 0, 0])])

    assert cube.origin.dtype == float
    assert isinstance(cube.atoms[0][2], np.ndarray)
    assert cube.atoms[0][:2] == (1, 0.0)


def test_cube_copy_is_deep():
    """Changing a copy leaves the original unchanged."""

    cube = _toy_cube()
    original = cube.volumetric_data.copy()

    copy = cube.copy()
    copy.volumetric_data[0, 0, 0] += 1.0
    copy.origin[0] += 1.0
    copy.atoms[0][2][0] += 1.0

    np.testing.assert_array_equal(cube.volumetric_data, original)
    assert cube.origin[0] == -1.0
    assert cube.atoms[0][2][0] == -0.5


def test_cube_save_checks_replaced_data(tmp_path):
    """Data replaced by another shape after construction is caught before writing."""

    cube = _toy_cube()
    cube.volumetric_data = np.zeros((2, 2, 2))

    with pytest.raises(ValueError):
        cube.save(tmp_path / "bad.cube")

    assert list(tmp_path.iterdir()) == []


def test_cube_from_file_on_the_class(tmp_path):
    """`from_file` is a classmethod, and still works on an instance."""

    cube = _toy_cube()
    cube.save(tmp_path / "toy.cube")

    for loaded in (
        Cube.from_file(tmp_path / "toy.cube"),
        cube.from_file(tmp_path / "toy.cube"),
    ):
        assert (loaded.n_x, loaded.n_y, loaded.n_z) == TOY_SHAPE
        assert loaded.comment1 == "toy cube"


def test_voxel_volume_skewed_cell():
    """The voxel volume is the triple product of the grid vectors."""

    assert _toy_cube().voxel_volume == pytest.approx(VOXEL_VOLUME, rel=1e-14)
    # orientation of the vectors does not matter
    assert _toy_cube(x_vector=-X_VECTOR).voxel_volume == pytest.approx(VOXEL_VOLUME)


def test_moved_charge_unchanged():
    """`moved_charge` gives the same number as its previous formula."""

    cube = _toy_cube()
    previous = (
        0.5
        * np.sum(np.abs(cube.volumetric_data))
        * np.linalg.norm(np.dot(np.cross(cube.x_vector, cube.y_vector), cube.z_vector))
    )

    assert cube.moved_charge() == previous


def test_integrate_without_weight():
    """The integral is the sum of the data times the voxel volume."""

    cube = _toy_cube()

    assert cube.integrate() == pytest.approx(
        np.sum(cube.volumetric_data) * VOXEL_VOLUME, rel=1e-13
    )


def test_integrate_with_weights():
    """Weights as a cube, as an array, and stacked along a leading axis."""

    cube = _toy_cube()
    w_a = _toy_cube(seed=1)
    w_b = _toy_cube(seed=2)
    expected = [
        np.sum(w.volumetric_data * cube.volumetric_data) * VOXEL_VOLUME
        for w in (w_a, w_b)
    ]

    assert cube.integrate(w_a) == pytest.approx(expected[0], rel=1e-13)
    assert cube.integrate(w_b.volumetric_data) == pytest.approx(expected[1], rel=1e-13)

    stacked = cube.integrate(np.stack([w_a.volumetric_data, w_b.volumetric_data]))
    assert stacked.shape == (2,)
    np.testing.assert_allclose(stacked, expected, rtol=1e-13)


def test_integrate_l2_norm():
    """The L2 norm of the data, as computed downstream, from `integrate`."""

    cube = _toy_cube()

    assert np.sqrt(cube.integrate(cube.volumetric_data)) == pytest.approx(
        np.linalg.norm(cube.volumetric_data) * np.sqrt(VOXEL_VOLUME), rel=1e-13
    )


@pytest.mark.parametrize(
    "shape", [(5,), (4, 3), (5, 3, 4), (4, 3, 5, 1)], ids=lambda s: str(s)
)
def test_integrate_rejects_weight_shape(shape):
    """A weight array must end with the grid shape, nothing is broadcast."""

    with pytest.raises(ValueError):
        _toy_cube().integrate(np.ones(shape))


@pytest.mark.parametrize(
    "other",
    [
        {"origin": (-1.0, 0.5, 2.1)},
        {"z_vector": np.array([0.0, 0.0, 0.3])},
        {"shape": (4, 3, 6)},
    ],
    ids=["origin", "vector", "shape"],
)
def test_integrate_rejects_weight_on_another_grid(other):
    """A weight cube must share the grid of the integrated cube."""

    with pytest.raises(ValueError):
        _toy_cube().integrate(_toy_cube(seed=1, **other))


def test_cube_add_and_subtract():
    """`+` and `-` give the data of `add_cubes` and `subtract_cubes`."""

    a, b = _toy_cube(seed=0), _toy_cube(seed=1)

    np.testing.assert_array_equal(
        (a + b).volumetric_data, a.volumetric_data + b.volumetric_data
    )
    np.testing.assert_array_equal(
        (a - b).volumetric_data, a.volumetric_data - b.volumetric_data
    )
    np.testing.assert_array_equal(
        (a + b).volumetric_data, add_cubes([a, b]).volumetric_data
    )
    np.testing.assert_array_equal(
        (a - b).volumetric_data, subtract_cubes(a, b).volumetric_data
    )


@pytest.mark.parametrize(
    "operation, factor",
    [
        (lambda c: 2.5 * c, 2.5),
        (lambda c: c * 2.5, 2.5),
        (lambda c: np.float64(2.5) * c, 2.5),
        (lambda c: c * np.int64(3), 3.0),
        (lambda c: c / 4, 0.25),
        (lambda c: -c, -1.0),
    ],
    ids=["left", "right", "numpy-float", "numpy-int", "divide", "negate"],
)
def test_cube_scalar_operations(operation, factor):
    """Scalars multiply and divide the data, the grid and atoms are kept."""

    cube = _toy_cube()
    result = operation(cube)

    assert isinstance(result, Cube)
    np.testing.assert_allclose(
        result.volumetric_data, factor * cube.volumetric_data, rtol=1e-15
    )
    np.testing.assert_array_equal(result.origin, cube.origin)
    assert result.voxel_volume == cube.voxel_volume
    assert result.n_atoms == cube.n_atoms


@pytest.mark.parametrize(
    "operation",
    [
        lambda c: c + 1.0,
        lambda c: 1.0 - c,
        lambda c: c * c,
        lambda c: c * np.ones(TOY_SHAPE),
        lambda c: np.ones(TOY_SHAPE) * c,
        lambda c: 2.0 / c,
    ],
    ids=[
        "cube+scalar",
        "scalar-cube",
        "cube*cube",
        "cube*array",
        "array*cube",
        "scalar/cube",
    ],
)
def test_cube_unsupported_operations(operation):
    """Only cube +/- cube and cube *, / scalar are defined."""

    with pytest.raises(TypeError):
        operation(_toy_cube())


OTHER_GRIDS = [
    {"origin": (-1.0, 0.5, 2.1)},
    {"y_vector": np.array([0.05, 0.31, 0.0])},
    {"shape": (1, 3, 5)},  # would broadcast against (4, 3, 5) without the check
]


@pytest.mark.parametrize(
    "operation",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: add_cubes([a, a, b]),
        cosine_similarity,
    ],
    ids=["add", "subtract", "add_cubes", "cosine_similarity"],
)
@pytest.mark.parametrize("other", OTHER_GRIDS, ids=["origin", "vector", "shape"])
def test_cube_operations_check_grid(operation, other):
    """Operations on two cubes refuse different grids."""

    with pytest.raises(ValueError):
        operation(_toy_cube(), _toy_cube(seed=1, **other))


def test_cube_grid_check_file_precision():
    """Grids that agree within the 6 decimals of a cube file are the same grid."""

    a = _toy_cube()
    b = _toy_cube(seed=1, origin=a.origin + 4e-7)

    np.testing.assert_allclose(
        (a - b).volumetric_data, a.volumetric_data - b.volumetric_data
    )


def test_cube_arithmetic_results_share_nothing():
    """Results own their arrays, take the left cube's atoms and have empty comments."""

    a = _toy_cube()
    b = _toy_cube(seed=1, atoms=[(1, 0.0, np.zeros(3))])
    a_origin, a_atom = a.origin.copy(), a.atoms[0][2].copy()

    for result in (a + b, a - b, 2.0 * a, -a, add_cubes([a, b]), subtract_cubes(a, b)):
        assert result.n_atoms == 2
        assert (result.comment1, result.comment2) == ("", "")
        assert not np.shares_memory(result.volumetric_data, a.volumetric_data)
        result.origin[0] += 1.0
        result.atoms[0][2][0] += 1.0

    np.testing.assert_array_equal(a.origin, a_origin)
    np.testing.assert_array_equal(a.atoms[0][2], a_atom)


def test_cosine_similarity_values():
    """Parallel and antiparallel data give +1 and -1."""

    cube = _toy_cube()

    assert cosine_similarity(cube, 3.0 * cube) == pytest.approx(1.0, abs=1e-15)
    assert cosine_similarity(cube, -cube) == pytest.approx(-1.0, abs=1e-15)


def _old_data_text(cube):
    """Volumetric data as the writer formatted it before NumPy's bulk writing."""

    flat = cube.volumetric_data.ravel()
    return "\n".join(
        "".join(f"{v: .5E} " for v in flat[i : i + 6]) for i in range(0, len(flat), 6)
    )


def _write_lines(path, lines):

    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.parametrize("shape", [TOY_SHAPE, (2, 2, 5), (1, 1, 1)], ids=str)
def test_cube_file_round_trip(tmp_path, shape):
    """A saved cube reads back within the precision of the file."""

    cube = _toy_cube(shape=shape)
    cube.save(tmp_path / "toy.cube")
    loaded = read_cube_file(tmp_path / "toy.cube")

    assert (loaded.comment1, loaded.comment2) == (cube.comment1, cube.comment2)
    assert (loaded.n_x, loaded.n_y, loaded.n_z, loaded.n_atoms) == (*shape, 2)
    np.testing.assert_allclose(loaded.volumetric_data, cube.volumetric_data, rtol=6e-6)
    np.testing.assert_allclose(loaded.origin, cube.origin, atol=1e-6)
    np.testing.assert_allclose(loaded.y_vector, cube.y_vector, atol=1e-6)
    for (num, charge, xyz), (num0, charge0, xyz0) in zip(loaded.atoms, cube.atoms):
        assert (num, charge) == (num0, charge0)
        np.testing.assert_allclose(xyz, xyz0, atol=1e-6)


@pytest.mark.parametrize("shape", [TOY_SHAPE, (2, 2, 5), (1, 1, 1)], ids=str)
def test_cube_file_data_unchanged(tmp_path, shape):
    """The data section is byte-identical to the previous writer's."""

    cube = _toy_cube(shape=shape)
    cube.save(tmp_path / "toy.cube")
    lines = (tmp_path / "toy.cube").read_text(encoding="utf-8").split("\n")

    assert "\n".join(lines[6 + cube.n_atoms :]) == _old_data_text(cube)


def test_read_cube_any_values_per_line(tmp_path):
    """The data may come in rows of n_z, as Gaussian writes them, or one per line."""

    cube = _toy_cube()
    cube.save(tmp_path / "toy.cube")
    lines = (tmp_path / "toy.cube").read_text(encoding="utf-8").split("\n")
    header = lines[: 6 + cube.n_atoms]
    values = [f"{v: .5E}" for v in cube.volumetric_data.ravel()]
    reference = read_cube_file(tmp_path / "toy.cube").volumetric_data

    rows = [" ".join(values[i : i + cube.n_z]) for i in range(0, len(values), cube.n_z)]
    for name, body in [("rows", rows), ("single", values)]:
        _write_lines(tmp_path / f"{name}.cube", header + body + [""])
        np.testing.assert_array_equal(
            read_cube_file(tmp_path / f"{name}.cube").volumetric_data, reference
        )


def _edited_toy_file(tmp_path, edit):
    """Path to a saved toy cube whose lines were changed by `edit`."""

    _toy_cube().save(tmp_path / "toy.cube")
    lines = (tmp_path / "toy.cube").read_text(encoding="utf-8").split("\n")
    _write_lines(tmp_path / "edited.cube", edit(lines))

    return tmp_path / "edited.cube"


def test_read_cube_values_per_voxel_field(tmp_path):
    """A fifth field of 1 (values per voxel) after the origin is accepted."""

    path = _edited_toy_file(
        tmp_path, lambda lines: lines[:2] + [lines[2] + "    1"] + lines[3:]
    )

    assert read_cube_file(path).volumetric_data.shape == TOY_SHAPE


@pytest.mark.parametrize(
    "edit",
    [
        lambda lines: lines[:-1],  # last data line lost
        lambda lines: lines + ["1.0"],  # one value too many
    ],
    ids=["truncated", "extra-value"],
)
def test_read_cube_wrong_number_of_values(tmp_path, edit):
    """Data that does not fill the grid exactly is refused."""

    with pytest.raises(ValueError):
        read_cube_file(_edited_toy_file(tmp_path, edit))


@pytest.mark.parametrize(
    "edit",
    [
        lambda lines: lines[:2]
        + ["    -2" + lines[2][6:]]
        + lines[3:],  # several orbitals
        lambda lines: lines[:2]
        + [lines[2] + "    3"]
        + lines[3:],  # three values per voxel
        lambda lines: lines[:3] + ["    -4" + lines[3][6:]] + lines[4:],  # angstrom
    ],
    ids=["orbitals", "values-per-voxel", "angstrom"],
)
def test_read_cube_unsupported_variants(tmp_path, edit):
    """Cube variants the reader does not handle are refused with a clear error."""

    with pytest.raises(NotImplementedError):
        read_cube_file(_edited_toy_file(tmp_path, edit))
