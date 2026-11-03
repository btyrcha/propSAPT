"""
Run configuration for the propSAPT runner (`prop-sapt run`).

A run is described by a small YAML or JSON file. Every key is optional except `tasks`::

    reference: RHF                # "RHF" or "RKS"
    functional: pbe0              # RKS only; defaults to the `default_functional` option
    grac: {A: 0.0851, B: 0.0999}  # RKS only, GRAC shifts in Hartree

    basis: aug-cc-pvtz
    df_basis: aug-cc-pvtz         # stem: DF_BASIS_SCF = <stem>-jkfit, DF_BASIS_SAPT = <stem>-ri
    df_basis_scf: null            # full basis names, taking precedence over the stem
    df_basis_sapt: null
    basis_overrides: basis_overrides.json   # path, or inline {element: {basis_key: basis}}
    basis_dirs: [basis]           # extra directories searched for <basis name>.gbs files

    memory: 2                     # in GB, or a string such as "2 GB" or "500 MB"
    threads: 2

    psi4_options: {scf_type: direct, e_convergence: 1.0e-10}
    propsapt_options: {r_convergence: 1.0e-10}

    output:
      dir: "."                    # "{geometry}" expands to the geometry file stem
      psi4_output: null           # defaults to output_hf.dat / output_propsapt_dft_<functional>.dat

    tasks:                        # true for defaults, a mapping to change them
      monomer_references: {cubes: true, fchk: true}
      sapt_energy: {results: sapt_results.csv, response: true}
      property: {prop: dipole, results: property_results.csv, response: true}
      densities: {orbital_basis: AO, save_cubes: true, save_matrices: true,
                  matrix_format: npz}

Values are taken from the built-in defaults, then the file, then the command-line flags.
A `--df-basis` flag sets both fitting bases from its stem, replacing `df_basis_scf` and
`df_basis_sapt` given in the file.

DF_BASIS_SAPT is always set: from `df_basis_sapt`, else `<df_basis>-ri`, else `<basis>-ri`.
DF_BASIS_SCF is set from `df_basis_scf`, else `<df_basis>-jkfit`; with neither given it is
left to Psi4, which picks `<basis>-jkfit` where that exists and a universal JKFIT basis
otherwise (e.g. for He with aug-cc-pvtz).
The densities task saves its matrices into one self-describing bundle, ``densities.npz``
(`matrix_format: npz`, read with `prop_sapt.load_densities`), or as bare ``.npy`` files
(`matrix_format: npy`, deprecated). The older `compress_matrices: true/false` setting still
works, as `npz`/`npy`, with a deprecation warning.

Relative paths inside the file (`basis_overrides`, `basis_dirs`) are resolved against the
directory of the file; the output directory is resolved against the current working
directory.

Basis sets missing from Psi4's library can be added as Psi4-format files named
`<basis name>.gbs` in one of the `basis_dirs`, e.g. written from ccRepo or the Basis Set
Exchange with `prop-sapt basis fetch` (see `prop_sapt.basis_fetch`).
"""

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping

from .densities import MATRIX_FORMATS
from .options import _OPTION_DEFAULTS, check_if_valid_option, check_option_value
from .basis_overrides import ASSIGNABLE_BASIS_NAME, load_basis_overrides


class ConfigError(ValueError):
    """Raised when a run configuration cannot be read or is invalid."""


# References the propSAPT code can run (see `helper_SAPT.__init__`).
SUPPORTED_REFERENCES = ("RHF", "RKS")

# Psi4 options the runner sets itself, from `basis`, `df_basis` and the `save_jk` requirement.
MANAGED_PSI4_OPTIONS = ("basis", "df_basis_scf", "df_basis_sapt", "save_jk")

# Tasks in the order they run, with their default settings.
TASK_DEFAULTS = {
    "monomer_references": {"cubes": True, "fchk": True},
    "sapt_energy": {"results": "sapt_results.csv", "response": True},
    "property": {"prop": "dipole", "results": "property_results.csv", "response": True},
    "densities": {
        "orbital_basis": "AO",
        "save_cubes": True,
        "save_matrices": True,
        "matrix_format": "npz",
    },
}

TOP_LEVEL_KEYS = frozenset(
    {
        "reference",
        "functional",
        "grac",
        "basis",
        "df_basis",
        "df_basis_scf",
        "df_basis_sapt",
        "basis_overrides",
        "basis_dirs",
        "memory",
        "threads",
        "psi4_options",
        "propsapt_options",
        "output",
        "tasks",
    }
)

OUTPUT_KEYS = frozenset({"dir", "psi4_output"})

# Keys of `resolve_config`'s `overrides`, as set by the command-line flags.
OVERRIDE_KEYS = (
    "reference",
    "functional",
    "grac_A",
    "grac_B",
    "basis",
    "df_basis",
    "memory",
    "threads",
    "output_dir",
)

DEFAULT_BASIS = "aug-cc-pvtz"
DEFAULT_MEMORY_GB = 2.0
DEFAULT_THREADS = 1

_MEMORY_RE = re.compile(r"^\s*(\d+(?:\.\d*)?|\.\d+)\s*([a-z]*)\s*$", re.IGNORECASE)
_MEMORY_UNITS = {"": 1.0, "g": 1.0, "gb": 1.0, "m": 1.0e-3, "mb": 1.0e-3}

# PyYAML reads "1e-10" (no decimal point) as a string, so such values are converted back.
_SCIENTIFIC_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


@dataclass(frozen=True)
class RunConfig:
    """Resolved and validated run configuration. Build it with `resolve_config`."""

    reference: str = "RHF"
    functional: str | None = None
    grac_A: float | None = None
    grac_B: float | None = None
    basis: str = DEFAULT_BASIS
    df_basis_scf: str | None = None
    df_basis_sapt: str = DEFAULT_BASIS + "-ri"
    basis_overrides: dict = field(default_factory=dict)
    basis_dirs: tuple = ()
    memory_gb: float = DEFAULT_MEMORY_GB
    threads: int = DEFAULT_THREADS
    psi4_options: dict = field(default_factory=dict)
    propsapt_options: dict = field(default_factory=dict)
    output_dir_template: str = "."
    psi4_output: str = "output_hf.dat"
    tasks: dict = field(default_factory=dict)
    source: str | None = None

    @property
    def method_tag(self) -> str:
        """Short method label used in the default filenames, "hf" or "dft_<functional>"."""

        if self.reference == "RKS":
            return f"dft_{self.functional}"
        return "hf"

    def effective_psi4_options(self) -> dict:
        """Returns every Psi4 option the runner sets: the basis keys, `save_jk` and
        `psi4_options`."""

        options = {"basis": self.basis}
        if self.df_basis_scf is not None:
            options["DF_BASIS_SCF"] = self.df_basis_scf
        options["DF_BASIS_SAPT"] = self.df_basis_sapt
        options["save_jk"] = True  # required by propSAPT
        options.update(self.psi4_options)

        return options

    def applied_basis_overrides(self) -> dict:
        """Returns the per-element basis overrides that apply to this run, those of the
        basis options it sets (see `apply_basis_overrides`)."""

        options = {key.upper() for key in self.effective_psi4_options()}

        return {
            option: dict(elements)
            for option, elements in self.basis_overrides.items()
            if option.upper() in options
        }

    def output_dir(self, geometry_stem: str) -> str:
        """Returns the output directory for a geometry, with "{geometry}" expanded."""

        return self.output_dir_template.replace("{geometry}", geometry_stem)

    def to_dict(self) -> dict:
        """Returns the configuration as a plain, JSON-serialisable dictionary."""

        return {
            "reference": self.reference,
            "functional": self.functional,
            "grac": {"A": self.grac_A, "B": self.grac_B},
            "basis": self.basis,
            "df_basis_scf": self.df_basis_scf,
            "df_basis_sapt": self.df_basis_sapt,
            "basis_overrides": self.basis_overrides,
            "basis_dirs": list(self.basis_dirs),
            "memory_gb": self.memory_gb,
            "threads": self.threads,
            "psi4_options": self.psi4_options,
            "propsapt_options": self.propsapt_options,
            "output": {
                "dir": self.output_dir_template,
                "psi4_output": self.psi4_output,
            },
            "tasks": self.tasks,
            "source": self.source,
        }


def load_config(path: str | os.PathLike) -> dict:
    """Reads a run configuration file.

    Files ending in ".json" are read as JSON, anything else as YAML.

    Args:
        path (str | os.PathLike): Path to the configuration file.

    Raises:
        ConfigError: If the file cannot be read or parsed, or does not hold a mapping.

    Returns:
        dict: The raw configuration, to be passed to `resolve_config`.
    """

    path = Path(path)

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise ConfigError(f"Cannot read config file '{path}': {err}") from err

    if path.suffix.lower() == ".json":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as err:
            raise ConfigError(f"Invalid JSON in '{path}': {err}") from err
    else:
        import yaml

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as err:
            raise ConfigError(f"Invalid YAML in '{path}': {err}") from err

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"Config file '{path}' must hold a mapping, not {type(raw).__name__}."
        )

    return dict(raw)


def resolve_config(
    raw: Mapping,
    base_dir: str | os.PathLike | None = None,
    overrides: Mapping | None = None,
    source: str | None = None,
) -> RunConfig:
    """Validates a raw configuration and fills in the defaults.

    Args:
        raw (Mapping): Raw configuration, e.g. from `load_config`.
        base_dir (str | os.PathLike | None, optional): Directory that relative paths in
            `raw` are resolved against, normally the directory of the config file.
            Defaults to the current working directory.
        overrides (Mapping | None, optional): Values that take precedence over `raw`, keyed
            by the names in `OVERRIDE_KEYS`. None values are ignored. Defaults to None.
        source (str | None, optional): Path of the config file, kept for provenance.

    Raises:
        ConfigError: If the configuration is invalid.

    Returns:
        RunConfig: The resolved configuration.
    """

    raw = _section(raw, "config", TOP_LEVEL_KEYS)
    base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    overrides = {
        key: value for key, value in (overrides or {}).items() if value is not None
    }
    unknown = sorted(set(overrides) - set(OVERRIDE_KEYS))
    if unknown:
        raise ConfigError(f"Unknown override(s): {', '.join(unknown)}.")

    reference = overrides.get("reference", raw.get("reference", "RHF"))
    if not isinstance(reference, str) or reference.upper() not in SUPPORTED_REFERENCES:
        raise ConfigError(
            f"Unsupported reference '{reference}'. "
            f"Supported references: {', '.join(SUPPORTED_REFERENCES)}."
        )
    reference = reference.upper()

    propsapt_options = _resolve_propsapt_options(raw.get("propsapt_options"))
    functional, grac_A, grac_B = _resolve_dft(
        reference, raw, overrides, propsapt_options
    )

    basis = overrides.get("basis", raw.get("basis", DEFAULT_BASIS))
    if not isinstance(basis, str) or not basis:
        raise ConfigError("'basis' must be a non-empty string.")
    df_basis_scf, df_basis_sapt = _resolve_fitting_bases(raw, overrides, basis)

    try:
        basis_overrides = load_basis_overrides(
            raw.get("basis_overrides"), base_dir=base_dir
        )
    except (OSError, ValueError, TypeError) as err:
        raise ConfigError(f"Invalid 'basis_overrides': {err}") from err

    _check_basis_overrides(
        basis_overrides,
        {"BASIS": basis, "DF_BASIS_SCF": df_basis_scf, "DF_BASIS_SAPT": df_basis_sapt},
    )

    basis_dirs = _resolve_basis_dirs(raw.get("basis_dirs"), base_dir)

    memory_gb = _parse_memory(
        overrides.get("memory", raw.get("memory", DEFAULT_MEMORY_GB))
    )
    threads = overrides.get("threads", raw.get("threads", DEFAULT_THREADS))
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ConfigError(f"'threads' must be a positive integer, not {threads!r}.")

    output_dir, psi4_output = _resolve_output(raw, overrides, reference, functional)

    return RunConfig(
        reference=reference,
        functional=functional,
        grac_A=grac_A,
        grac_B=grac_B,
        basis=basis,
        df_basis_scf=df_basis_scf,
        df_basis_sapt=df_basis_sapt,
        basis_overrides=basis_overrides,
        basis_dirs=basis_dirs,
        memory_gb=memory_gb,
        threads=threads,
        psi4_options=_resolve_psi4_options(raw.get("psi4_options")),
        propsapt_options=propsapt_options,
        output_dir_template=output_dir,
        psi4_output=psi4_output,
        tasks=_resolve_tasks(raw.get("tasks")),
        source=source,
    )


def _section(value, name: str, valid_keys=None) -> dict:
    """Checks that a config section is a mapping (None counts as empty) with only
    `valid_keys`, if given."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping, not {type(value).__name__}.")

    if valid_keys is not None:
        unknown = sorted(str(key) for key in set(value) - set(valid_keys))
        if unknown:
            raise ConfigError(
                f"Unknown '{name}' key(s): {', '.join(unknown)}. "
                f"Valid keys: {', '.join(sorted(valid_keys))}."
            )

    return dict(value)


def _resolve_dft(
    reference: str, raw: Mapping, overrides: Mapping, propsapt_options: Mapping
):
    """Returns the functional and the GRAC shifts of monomers A and B."""

    functional = overrides.get("functional", raw.get("functional"))
    grac = _section(raw.get("grac"), "grac", ("A", "B"))
    grac_A = _as_float(overrides.get("grac_A", grac.get("A")), "grac.A")
    grac_B = _as_float(overrides.get("grac_B", grac.get("B")), "grac.B")

    if reference != "RKS":
        if functional is not None:
            raise ConfigError("'functional' is only valid with reference RKS.")
        if grac_A is not None or grac_B is not None:
            raise ConfigError("'grac' is only valid with reference RKS.")
        return None, None, None

    if functional is None:
        functional = propsapt_options.get(
            "default_functional", _OPTION_DEFAULTS["default_functional"]
        )
    if not isinstance(functional, str) or not functional:
        raise ConfigError("'functional' must be a non-empty string.")

    return functional, grac_A, grac_B


def _resolve_fitting_bases(raw: Mapping, overrides: Mapping, basis: str):
    """Returns the DF_BASIS_SCF (None to leave it to Psi4) and DF_BASIS_SAPT basis names."""

    # a --df-basis flag replaces every fitting basis given in the file
    if "df_basis" in overrides:
        df_stem = overrides["df_basis"]
        df_basis_scf = df_basis_sapt = None
    else:
        df_stem = raw.get("df_basis")
        df_basis_scf = raw.get("df_basis_scf")
        df_basis_sapt = raw.get("df_basis_sapt")

    for name, value in (
        ("df_basis", df_stem),
        ("df_basis_scf", df_basis_scf),
        ("df_basis_sapt", df_basis_sapt),
    ):
        if value is not None and (not isinstance(value, str) or not value):
            raise ConfigError(f"'{name}' must be a non-empty string.")

    if df_basis_scf is None and df_stem is not None:
        df_basis_scf = df_stem + "-jkfit"
    if df_basis_sapt is None:
        df_basis_sapt = (df_stem or basis) + "-ri"

    return df_basis_scf, df_basis_sapt


def _resolve_basis_dirs(basis_dirs, base_dir: Path) -> tuple:
    """Returns the extra basis set directories as absolute paths, checking they exist."""

    if basis_dirs is None:
        return ()
    if isinstance(basis_dirs, (str, os.PathLike)):
        basis_dirs = [basis_dirs]
    if not isinstance(basis_dirs, (list, tuple)):
        raise ConfigError("'basis_dirs' must be a directory path or a list of them.")

    resolved = []
    for directory in basis_dirs:
        if not isinstance(directory, (str, os.PathLike)):
            raise ConfigError(f"'basis_dirs' entries must be paths, not {directory!r}.")
        path = (base_dir / directory).resolve()
        if not path.is_dir():
            raise ConfigError(f"Basis set directory not found: '{path}'.")
        resolved.append(str(path))

    return tuple(resolved)


def _check_basis_overrides(basis_overrides: Mapping, defaults: Mapping) -> None:
    """Checks that every overridden basis option has a default Psi4 can assign to the
    remaining elements."""

    for option in basis_overrides:
        if option not in defaults:
            continue

        default = defaults[option]
        if default is None:
            raise ConfigError(
                f"A {option} basis override needs a default basis to apply it to: "
                "set 'df_basis' or 'df_basis_scf'."
            )
        if not ASSIGNABLE_BASIS_NAME.match(default):
            raise ConfigError(
                f"{option} '{default}' cannot be combined with per-element overrides: Psi4 "
                "cannot assign that name to single elements. Write the basis set into a file "
                "with `prop-sapt basis fetch` and use it with 'basis_dirs'."
            )


def _resolve_psi4_options(options) -> dict:
    """Validates the `psi4_options` mapping."""

    options = _section(options, "psi4_options")

    for key in options:
        if not isinstance(key, str):
            raise ConfigError(f"Psi4 option keys must be strings, not {key!r}.")
        if key.lower() in MANAGED_PSI4_OPTIONS:
            raise ConfigError(
                f"Psi4 option '{key}' is set by the runner: use 'basis', 'df_basis', "
                "'df_basis_scf' and 'df_basis_sapt' for the basis sets "
                "('save_jk' is always on)."
            )

    return {key: _coerce_scientific(value) for key, value in options.items()}


def _resolve_propsapt_options(options) -> dict:
    """Validates the `propsapt_options` mapping, returning it with lowercase keys."""

    options = _section(options, "propsapt_options")

    for key in options:
        try:
            check_if_valid_option(key)
        except (KeyError, TypeError) as err:
            raise ConfigError(f"Invalid 'propsapt_options': {err.args[0]}") from err

    resolved = {key.lower(): _coerce_scientific(value) for key, value in options.items()}

    for key, value in resolved.items():
        try:
            check_option_value(key, value)
        except ValueError as err:
            raise ConfigError(f"Invalid 'propsapt_options': {err.args[0]}") from err

    return resolved


def _resolve_output(
    raw: Mapping, overrides: Mapping, reference: str, functional: str | None
):
    """Returns the output directory template and the Psi4 output filename."""

    output = _section(raw.get("output"), "output", OUTPUT_KEYS)

    output_dir = os.fspath(overrides.get("output_dir", output.get("dir")) or ".")

    if reference == "RKS":
        default_psi4_output = f"output_propsapt_dft_{functional}.dat"
    else:
        default_psi4_output = "output_hf.dat"
    psi4_output = output.get("psi4_output") or default_psi4_output
    if not isinstance(psi4_output, str):
        raise ConfigError("'output.psi4_output' must be a string.")

    return output_dir, psi4_output


def _resolve_tasks(tasks) -> dict:
    """Validates the `tasks` entry and merges each enabled task with its defaults.

    Returns the enabled tasks in run order, whatever order they are given in.
    """

    if tasks is None:
        raise ConfigError(
            f"No tasks given. Add a 'tasks' entry with any of: {', '.join(TASK_DEFAULTS)}."
        )

    # a plain list of task names enables them with default settings
    if isinstance(tasks, (list, tuple)):
        if not all(isinstance(name, str) for name in tasks):
            raise ConfigError("A 'tasks' list must hold task names.")
        tasks = {name: True for name in tasks}

    if not isinstance(tasks, Mapping):
        raise ConfigError("'tasks' must be a mapping or a list of task names.")

    unknown = sorted(str(name) for name in set(tasks) - set(TASK_DEFAULTS))
    if unknown:
        raise ConfigError(
            f"Unknown task(s): {', '.join(unknown)}. "
            f"Valid tasks: {', '.join(TASK_DEFAULTS)}."
        )

    resolved = {}
    for name in TASK_DEFAULTS:
        settings = tasks.get(name)
        if settings is not None and settings is not False:
            resolved[name] = _resolve_task(name, settings)

    if not resolved:
        raise ConfigError("All tasks are disabled; enable at least one.")

    return resolved


def _resolve_task(name: str, settings) -> dict:
    """Merges one enabled task's settings (true or a mapping) with its defaults."""

    if settings is True:
        settings = {}
    if not isinstance(settings, Mapping):
        raise ConfigError(
            f"Task '{name}' must be true, false or a mapping of settings."
        )

    if name == "densities" and "compress_matrices" in settings:
        settings = _translate_compress_matrices(dict(settings))

    defaults = TASK_DEFAULTS[name]
    unknown = sorted(str(key) for key in set(settings) - set(defaults))
    if unknown:
        raise ConfigError(
            f"Unknown setting(s) for task '{name}': {', '.join(unknown)}. "
            f"Valid settings: {', '.join(defaults)}."
        )

    merged = {**defaults, **settings}
    _check_task_settings(name, merged)

    return merged


def _check_task_settings(name: str, settings: dict) -> None:
    """Checks the types and values of one task's settings."""

    for key, default in TASK_DEFAULTS[name].items():
        value = settings[key]

        if key == "response":
            if value not in (True, False, "both"):
                raise ConfigError(
                    f"'tasks.{name}.response' must be true, false or \"both\", not {value!r}."
                )

        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(
                    f"'tasks.{name}.{key}' must be true or false, not {value!r}."
                )

        elif not isinstance(value, str) or not value:
            raise ConfigError(f"'tasks.{name}.{key}' must be a non-empty string.")

    if name == "property" and settings["prop"] != "dipole":
        raise ConfigError(
            f"Unsupported property '{settings['prop']}' in 'tasks.property'. "
            "Supported: dipole."
        )

    if name == "densities":
        settings["orbital_basis"] = settings["orbital_basis"].upper()
        if settings["orbital_basis"] not in ("AO", "MO"):
            raise ConfigError('\'tasks.densities.orbital_basis\' must be "AO" or "MO".')

        settings["matrix_format"] = settings["matrix_format"].lower()
        if settings["matrix_format"] not in MATRIX_FORMATS:
            raise ConfigError(
                f"'tasks.densities.matrix_format' must be one of {', '.join(MATRIX_FORMATS)}."
            )


def _translate_compress_matrices(settings: dict) -> dict:
    """Replaces the deprecated `compress_matrices` setting of the densities task with
    `matrix_format`: true is "npz", false is "npy"."""

    compress = settings.pop("compress_matrices")

    if "matrix_format" in settings:
        raise ConfigError(
            "'tasks.densities.compress_matrices' is deprecated and cannot be combined "
            "with 'matrix_format'; use 'matrix_format' alone."
        )
    if not isinstance(compress, bool):
        raise ConfigError(
            f"'tasks.densities.compress_matrices' must be true or false, not {compress!r}."
        )

    warnings.warn(
        "'tasks.densities.compress_matrices' is deprecated, use 'matrix_format: npz' "
        "(the default) or 'matrix_format: npy'.",
        FutureWarning,
        stacklevel=2,
    )
    settings["matrix_format"] = "npz" if compress else "npy"

    return settings


def _parse_memory(value) -> float:
    """Parses a memory amount into GB: a number (GB) or a string such as "2 GB" or "500 MB"."""

    value = _coerce_scientific(value)

    if isinstance(value, bool):
        raise ConfigError(f"Invalid memory {value!r}.")

    if isinstance(value, (int, float)):
        memory_gb = float(value)

    elif isinstance(value, str):
        match = _MEMORY_RE.match(value)
        if match is None or match.group(2).lower() not in _MEMORY_UNITS:
            raise ConfigError(
                f"Invalid memory '{value}'. Use a number of GB or a string such as "
                "'2 GB' or '500 MB'."
            )
        memory_gb = float(match.group(1)) * _MEMORY_UNITS[match.group(2).lower()]

    else:
        raise ConfigError(f"Invalid memory {value!r}.")

    if memory_gb <= 0:
        raise ConfigError(f"Memory must be positive, not {value!r}.")

    return memory_gb


def _as_float(value, name: str) -> float | None:
    """Converts an optional numeric value to float."""

    value = _coerce_scientific(value)

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be a number, not {value!r}.")

    return float(value)


def _coerce_scientific(value):
    """Converts strings in scientific notation, e.g. "1e-10", to float; other values pass
    through unchanged."""

    if isinstance(value, str) and _SCIENTIFIC_RE.match(value.strip()):
        return float(value)

    return value
