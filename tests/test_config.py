"""Tests of the run configuration (`prop_sapt.config`) and the `prop-sapt` command line.

None of these run a calculation.
"""

import json

import pytest

from prop_sapt.cli import main
from prop_sapt.config import ConfigError, load_config, resolve_config, TASK_DEFAULTS

GEOMETRY = """
symmetry c1
no_com
no_reorient
units bohr
0 1
He  -2.500000000   0.000000000   0.000000000
--
0 1
H    1.775500000   0.000000000   0.000000000
H    3.224500000   0.000000000   0.000000000
"""

YAML_CONFIG = """
reference: RHF
basis: aug-cc-pvdz
df_basis: aug-cc-pvtz
memory: 4 GB
threads: 2
psi4_options:
  scf_type: direct
  e_convergence: 1.0e-10
propsapt_options:
  r_convergence: 1.0e-9
output:
  dir: "results/{geometry}"
tasks:
  sapt_energy: {response: both}
  densities: true
"""

JSON_CONFIG = {
    "reference": "RHF",
    "basis": "aug-cc-pvdz",
    "df_basis": "aug-cc-pvtz",
    "memory": "4 GB",
    "threads": 2,
    "psi4_options": {"scf_type": "direct", "e_convergence": 1.0e-10},
    "propsapt_options": {"r_convergence": 1.0e-9},
    "output": {"dir": "results/{geometry}"},
    "tasks": {"sapt_energy": {"response": "both"}, "densities": True},
}


def test_yaml_and_json_resolve_identically(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(YAML_CONFIG, encoding="utf-8")
    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps(JSON_CONFIG), encoding="utf-8")

    from_yaml = resolve_config(load_config(yaml_path), base_dir=tmp_path)
    from_json = resolve_config(load_config(json_path), base_dir=tmp_path)

    assert from_yaml == from_json
    assert from_yaml.memory_gb == 4.0
    assert from_yaml.propsapt_options == {"r_convergence": 1.0e-9}
    assert from_yaml.tasks["sapt_energy"] == {
        "results": "sapt_results.csv",
        "response": "both",
    }
    assert from_yaml.tasks["densities"] == TASK_DEFAULTS["densities"]


def test_defaults():
    config = resolve_config({"tasks": ["sapt_energy"]})

    assert config.reference == "RHF"
    assert config.functional is None
    assert config.basis == "aug-cc-pvtz"
    assert config.df_basis_sapt == "aug-cc-pvtz-ri"
    assert config.df_basis_scf is None
    assert config.memory_gb == 2.0
    assert config.threads == 1
    assert config.psi4_output == "output_hf.dat"
    assert config.method_tag == "hf"
    assert config.output_dir("geo1") == "."
    assert config.effective_psi4_options() == {
        "basis": "aug-cc-pvtz",
        "DF_BASIS_SAPT": "aug-cc-pvtz-ri",
        "save_jk": True,
    }


def test_df_basis_stem_sets_both_fitting_bases():
    config = resolve_config(
        {"basis": "d-aug-cc-pv5z", "df_basis": "aug-cc-pv5z", "tasks": ["densities"]}
    )

    options = config.effective_psi4_options()
    assert options["basis"] == "d-aug-cc-pv5z"
    assert options["DF_BASIS_SCF"] == "aug-cc-pv5z-jkfit"
    assert options["DF_BASIS_SAPT"] == "aug-cc-pv5z-ri"


def test_explicit_fitting_bases_take_precedence_over_the_stem():
    raw = {
        "df_basis": "aug-cc-pvdz",
        "df_basis_sapt": "aug-cc-pvqz-ri",
        "tasks": ["sapt_energy"],
    }

    config = resolve_config(raw)
    assert config.df_basis_scf == "aug-cc-pvdz-jkfit"
    assert config.df_basis_sapt == "aug-cc-pvqz-ri"

    # a --df-basis flag replaces everything the file says about the fitting bases
    config = resolve_config(raw, overrides={"df_basis": "cc-pvtz"})
    assert config.df_basis_scf == "cc-pvtz-jkfit"
    assert config.df_basis_sapt == "cc-pvtz-ri"


def test_rks_defaults():
    config = resolve_config({"reference": "rks", "tasks": ["sapt_energy"]})

    assert config.reference == "RKS"
    assert config.functional == "pbe0"
    assert config.method_tag == "dft_pbe0"
    assert config.psi4_output == "output_propsapt_dft_pbe0.dat"

    config = resolve_config(
        {
            "reference": "RKS",
            "propsapt_options": {"default_functional": "b3lyp"},
            "grac": {"A": 0.1, "B": "2e-2"},
            "tasks": ["sapt_energy"],
        }
    )
    assert config.functional == "b3lyp"
    assert (config.grac_A, config.grac_B) == (0.1, 0.02)


def test_command_line_overrides_take_precedence():
    raw = {
        "basis": "aug-cc-pvdz",
        "memory": 4,
        "threads": 2,
        "output": {"dir": "from-file"},
        "tasks": ["sapt_energy"],
    }
    overrides = {
        "reference": "RKS",
        "functional": "b3lyp",
        "grac_A": 0.05,
        "grac_B": 0.07,
        "basis": "aug-cc-pvtz",
        "memory": "32 GB",
        "threads": 8,
        "output_dir": "from-flag",
        "df_basis": None,  # unset flags are ignored
    }

    config = resolve_config(raw, overrides=overrides)

    assert config.reference == "RKS"
    assert config.functional == "b3lyp"
    assert (config.grac_A, config.grac_B) == (0.05, 0.07)
    assert config.basis == "aug-cc-pvtz"
    assert config.df_basis_sapt == "aug-cc-pvtz-ri"
    assert config.memory_gb == 32.0
    assert config.threads == 8
    assert config.output_dir_template == "from-flag"


@pytest.mark.parametrize(
    "memory, expected",
    [(2, 2.0), (1.5, 1.5), ("2 GB", 2.0), ("2gb", 2.0), ("16G", 16.0), ("500 MB", 0.5)],
)
def test_memory_formats(memory, expected):
    config = resolve_config({"memory": memory, "tasks": ["sapt_energy"]})

    assert config.memory_gb == pytest.approx(expected)


def test_scientific_notation_strings_become_numbers(tmp_path):
    # PyYAML reads "1e-10" (no decimal point) as a string
    path = tmp_path / "config.yaml"
    path.write_text(
        "psi4_options: {e_convergence: 1e-10, scf_type: direct}\n"
        "propsapt_options: {r_convergence: 1e-8}\n"
        "tasks: [sapt_energy]\n",
        encoding="utf-8",
    )

    config = resolve_config(load_config(path))

    assert config.psi4_options == {"e_convergence": 1.0e-10, "scf_type": "direct"}
    assert config.propsapt_options == {"r_convergence": 1.0e-8}


def test_tasks_run_in_a_fixed_order():
    config = resolve_config(
        {
            "tasks": {
                "densities": True,
                "sapt_energy": True,
                "monomer_references": {"fchk": False},
            }
        }
    )

    assert list(config.tasks) == ["monomer_references", "sapt_energy", "densities"]
    assert config.tasks["monomer_references"] == {"cubes": True, "fchk": False}

    config = resolve_config({"tasks": {"property": True, "densities": False}})
    assert list(config.tasks) == ["property"]


def test_densities_matrix_format():
    config = resolve_config({"tasks": {"densities": {"matrix_format": "NPY"}}})

    assert config.tasks["densities"]["matrix_format"] == "npy"
    assert TASK_DEFAULTS["densities"]["matrix_format"] == "npz"


@pytest.mark.parametrize("compress, matrix_format", [(True, "npz"), (False, "npy")])
def test_deprecated_compress_matrices_sets_matrix_format(compress, matrix_format):
    with pytest.warns(FutureWarning, match="compress_matrices' is deprecated"):
        config = resolve_config(
            {"tasks": {"densities": {"compress_matrices": compress}}}
        )

    assert config.tasks["densities"] == {
        **TASK_DEFAULTS["densities"],
        "matrix_format": matrix_format,
    }


def test_applied_basis_overrides():
    config = resolve_config(
        {
            "basis_overrides": {
                "He": {"DF_BASIS_SAPT": "aug-cc-pvtz-ri", "DF_BASIS_MP2": "cc-pvdz-ri"}
            },
            "tasks": ["sapt_energy"],
        }
    )

    assert config.applied_basis_overrides() == {
        "DF_BASIS_SAPT": {"He": "aug-cc-pvtz-ri"}
    }


def test_output_dir_template():
    config = resolve_config(
        {"output": {"dir": "runs/{geometry}/hf"}, "tasks": ["sapt_energy"]}
    )

    assert config.output_dir("geo1") == "runs/geo1/hf"


def test_basis_overrides_file_is_resolved_against_the_config_directory(tmp_path):
    overrides = {
        "Zn": {"DF_BASIS_SCF": "def2-tzvpd-ri", "df_basis_sapt": "def2-tzvpd-ri"},
        "Li": {"DF_BASIS_MP2": "def2-tzvpp-ri"},
    }
    (tmp_path / "basis_overrides.json").write_text(
        json.dumps(overrides), encoding="utf-8"
    )
    raw = {"df_basis": "aug-cc-pvtz", "tasks": ["sapt_energy"]}

    from_file = resolve_config(
        {**raw, "basis_overrides": "basis_overrides.json"}, base_dir=tmp_path
    )
    inline = resolve_config({**raw, "basis_overrides": overrides})

    expected = {
        "DF_BASIS_SCF": {"Zn": "def2-tzvpd-ri"},
        "DF_BASIS_SAPT": {"Zn": "def2-tzvpd-ri"},
        "DF_BASIS_MP2": {"Li": "def2-tzvpp-ri"},
    }
    assert from_file.basis_overrides == expected
    assert inline.basis_overrides == expected


def test_basis_dirs_are_resolved_against_the_config_directory(tmp_path):
    (tmp_path / "basis").mkdir()
    (tmp_path / "more").mkdir()

    config = resolve_config(
        {"basis_dirs": "basis", "tasks": ["sapt_energy"]}, base_dir=tmp_path
    )
    assert config.basis_dirs == (str(tmp_path / "basis"),)

    config = resolve_config(
        {"basis_dirs": ["basis", str(tmp_path / "more")], "tasks": ["sapt_energy"]},
        base_dir=tmp_path,
    )
    assert config.basis_dirs == (str(tmp_path / "basis"), str(tmp_path / "more"))

    with pytest.raises(ConfigError, match="Basis set directory not found"):
        resolve_config(
            {"basis_dirs": ["missing"], "tasks": ["sapt_energy"]}, base_dir=tmp_path
        )


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"tasks": ["sapt_energy"], "bais": "cc-pvdz"}, "Unknown 'config' key"),
        ({}, "No tasks given"),
        ({"tasks": {"sapt_energy": False}}, "All tasks are disabled"),
        ({"tasks": ["sapt"]}, "Unknown task"),
        ({"tasks": {"sapt_energy": {"reponse": True}}}, "Unknown setting"),
        ({"tasks": {"sapt_energy": {"response": "yes"}}}, "response"),
        ({"tasks": {"property": {"response": "yes"}}}, "tasks.property.response"),
        ({"tasks": {"densities": {"save_cubes": "yes"}}}, "true or false"),
        ({"tasks": {"densities": {"orbital_basis": "NO"}}}, "orbital_basis"),
        ({"tasks": {"densities": {"matrix_format": "hdf5"}}}, "matrix_format"),
        (
            {
                "tasks": {
                    "densities": {"compress_matrices": True, "matrix_format": "npz"}
                }
            },
            "cannot be combined",
        ),
        ({"tasks": {"densities": {"compress_matrices": "yes"}}}, "true or false"),
        ({"tasks": {"property": {"prop": "quadrupole"}}}, "Unsupported property"),
        ({"tasks": {"sapt_energy": "yes"}}, "must be true, false or a mapping"),
        ({"reference": "UHF", "tasks": ["sapt_energy"]}, "Unsupported reference"),
        (
            {"functional": "pbe0", "tasks": ["sapt_energy"]},
            "only valid with reference RKS",
        ),
        (
            {"grac": {"A": 0.1}, "tasks": ["sapt_energy"]},
            "only valid with reference RKS",
        ),
        (
            {"reference": "RKS", "grac": {"C": 0.1}, "tasks": ["sapt_energy"]},
            "Unknown 'grac'",
        ),
        (
            {"reference": "RKS", "grac": {"A": "big"}, "tasks": ["sapt_energy"]},
            "must be a number",
        ),
        (
            {"psi4_options": {"BASIS": "cc-pvdz"}, "tasks": ["sapt_energy"]},
            "set by the runner",
        ),
        (
            {"psi4_options": {"df_basis_sapt": "x"}, "tasks": ["sapt_energy"]},
            "set by the runner",
        ),
        (
            {"psi4_options": {"save_jk": False}, "tasks": ["sapt_energy"]},
            "set by the runner",
        ),
        (
            {"propsapt_options": {"r_conv": 1e-8}, "tasks": ["sapt_energy"]},
            "propsapt_options",
        ),
        (
            {"propsapt_options": {"cpscf_type": "PK"}, "tasks": ["sapt_energy"]},
            "Invalid 'propsapt_options': Invalid `cpscf_type`",
        ),
        ({"memory": "lots", "tasks": ["sapt_energy"]}, "Invalid memory"),
        ({"memory": 0, "tasks": ["sapt_energy"]}, "Memory must be positive"),
        ({"threads": 0, "tasks": ["sapt_energy"]}, "threads"),
        ({"output": {"directory": "x"}, "tasks": ["sapt_energy"]}, "Unknown 'output'"),
        ({"basis_overrides": ["Zn"], "tasks": ["sapt_energy"]}, "basis_overrides"),
        (
            {
                "basis_overrides": {"Zn": {"DF_BASIS_SCF": "def2-tzvpd-ri"}},
                "tasks": ["sapt_energy"],
            },
            "set 'df_basis' or 'df_basis_scf'",
        ),
        (
            {
                "basis_overrides": {"He": {"DF_BASIS_SAPT": "bse:def2-qzvpp-ri"}},
                "tasks": ["sapt_energy"],
            },
            "'bse:' names cannot be assigned to single elements",
        ),
        (
            {
                "basis_overrides": {"He": {"DF_BASIS_SAPT": "my.basis"}},
                "tasks": ["sapt_energy"],
            },
            "can only contain",
        ),
        (
            {
                "df_basis_sapt": "bse:def2-qzvpp-ri",
                "basis_overrides": {"He": {"DF_BASIS_SAPT": "aug-cc-pvtz-ri"}},
                "tasks": ["sapt_energy"],
            },
            "cannot be combined with per-element overrides",
        ),
    ],
)
def test_invalid_configs(raw, message):
    with pytest.raises(ConfigError, match=message):
        resolve_config(raw)


def test_load_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(tmp_path / "missing.yaml")

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("tasks: [sapt_energy\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(bad_yaml)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{tasks: }", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid JSON"):
        load_config(bad_json)

    not_a_mapping = tmp_path / "list.yaml"
    not_a_mapping.write_text("- sapt_energy\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must hold a mapping"):
        load_config(not_a_mapping)


@pytest.fixture()
def config_and_geometry(tmp_path):
    """A valid config file and geometry file in a temporary directory."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(YAML_CONFIG, encoding="utf-8")
    geometry_path = tmp_path / "he-h2.p4m"
    geometry_path.write_text(GEOMETRY, encoding="utf-8")

    return config_path, geometry_path


def test_cli_dry_run(config_and_geometry, tmp_path, capsys, monkeypatch):
    config_path, geometry_path = config_and_geometry
    monkeypatch.chdir(tmp_path)

    status = main(
        ["run", str(config_path), str(geometry_path), "--memory", "32", "--dry-run"]
    )

    assert status == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["config"]["memory_gb"] == 32.0
    assert summary["psi4_options"]["DF_BASIS_SAPT"] == "aug-cc-pvtz-ri"
    assert summary["geometries"][0]["output_dir"] == str(tmp_path / "results" / "he-h2")
    # a dry run calculates nothing
    assert not (tmp_path / "results").exists()


def test_cli_reports_invalid_config(config_and_geometry, capsys):
    config_path, geometry_path = config_and_geometry

    status = main(["run", str(config_path), str(geometry_path), "--reference", "UHF"])

    assert status == 2
    assert "Unsupported reference" in capsys.readouterr().err


def test_cli_refuses_shared_output_directory(config_and_geometry, tmp_path, capsys):
    config_path, geometry_path = config_and_geometry
    other_geometry = tmp_path / "other.p4m"
    other_geometry.write_text(GEOMETRY, encoding="utf-8")
    geometries = [str(geometry_path), str(other_geometry)]

    # one directory per geometry: fine
    assert main(["run", str(config_path), *geometries, "--dry-run"]) == 0
    capsys.readouterr()

    # one directory for both: refused
    status = main(["run", str(config_path), *geometries, "--output-dir", str(tmp_path)])

    assert status == 2
    assert "same output directory" in capsys.readouterr().err


def test_cli_reports_missing_geometry(config_and_geometry, tmp_path, capsys):
    config_path, _ = config_and_geometry

    status = main(["run", str(config_path), str(tmp_path / "missing.p4m"), "--dry-run"])

    assert status == 2
    assert "Geometry file(s) not found" in capsys.readouterr().err
