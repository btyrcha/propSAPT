"""
Configurable runner: one configuration file, one geometry, the selected propSAPT tasks.

Used by the `prop-sapt run` command (see `prop_sapt.cli`), and from Python::

    >>> from prop_sapt import run
    >>> results = run("config.yaml", "geos/geo1.p4m")

See `prop_sapt.config` for the configuration format.
"""

import datetime
import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import psi4

from ._version import __version__
from .basis_overrides import apply_basis_overrides
from .config import RunConfig, load_config, resolve_config
from .densities import calc_densities
from .fragments import parse_fragments
from .molecule import Dimer
from .options import reset_propsapt_options, set_propsapt_options
from .properties import calc_property
from .sapt_driver import calc_sapt_energy
from .utils import CalcTimer, prepare_path

PROVENANCE_FILENAME = "propsapt_run.json"


def setup_psi4(config: RunConfig, output_file: str) -> list[str]:
    """Sets up Psi4 for a calculation described by `config`.

    Resets the Psi4 and propSAPT options first, so the calculation depends on `config`
    alone. Then sets memory, threads and the output file, the basis set search path, the
    Psi4 options, the per-element basis overrides and the propSAPT options.

    The directories in `config.basis_dirs` are prepended to the PSIPATH environment
    variable, which Psi4 searches for `<basis name>.gbs` files.

    Args:
        config (RunConfig): Run configuration.
        output_file (str): Path of the Psi4 output file.

    Returns:
        list[str]: The basis options that were overridden per element.
    """

    psi4.core.clean_options()
    reset_propsapt_options()

    psi4.core.set_output_file(prepare_path(output_file), False)
    psi4.set_memory(int(config.memory_gb * 1e9))
    psi4.set_num_threads(config.threads)

    add_basis_dirs(config.basis_dirs)

    options = config.effective_psi4_options()
    psi4.set_options(options)
    applied = apply_basis_overrides(options, config.basis_overrides)

    if config.propsapt_options:
        set_propsapt_options(config.propsapt_options)

    return applied


def add_basis_dirs(basis_dirs) -> None:
    """Prepends directories to the PSIPATH environment variable, where Psi4 looks for
    `<basis name>.gbs` files, skipping those already on it.

    Args:
        basis_dirs (Iterable[str]): Directories holding basis set files.
    """

    current = [path for path in os.environ.get("PSIPATH", "").split(os.pathsep) if path]
    new = [str(path) for path in basis_dirs if str(path) not in current]

    if new:
        os.environ["PSIPATH"] = os.pathsep.join(new + current)


def build_dimer(config: RunConfig, geometry: str) -> Dimer:
    """Creates the `Dimer` for `geometry` as described by `config`.

    Psi4 has to be set up first, see `setup_psi4`.

    Args:
        config (RunConfig): Run configuration.
        geometry (str): Dimer geometry in Psi4 format.

    Returns:
        Dimer: The dimer, with the monomer SCF calculations done.
    """

    # memory comes from `psi4.set_memory` in `setup_psi4`
    kwargs = {"reference": config.reference}

    if config.reference == "RKS":
        kwargs["functional"] = config.functional
        if config.grac_A is not None:
            kwargs["grac_A"] = config.grac_A
        if config.grac_B is not None:
            kwargs["grac_B"] = config.grac_B

    return Dimer(geometry, **kwargs)


def _task_monomer_references(
    dimer: Dimer, config: RunConfig, out_dir: Path, settings: dict
):
    """Saves the monomer densities as cubes and the monomer wavefunctions as .fchk files."""

    tag = config.method_tag
    written = []

    if settings["cubes"]:
        cube_files = [str(out_dir / f"dm_{tag}_{monomer}.cube") for monomer in "AB"]
        dimer.save_cube(
            [
                2 * np.asarray(dimer.wfnA.Da()),
                2 * np.asarray(dimer.wfnB.Da()),
            ],  # RHF spin
            ["density", "density"],
            cube_files,
        )
        written += cube_files

    if settings["fchk"]:
        prefix = "" if config.reference == "RHF" else f"propsapt_{tag}_"
        fchk_files = [
            str(out_dir / f"{prefix}monomer_{monomer}.fchk") for monomer in "ab"
        ]
        psi4.fchk(dimer.wfnA, fchk_files[0])
        psi4.fchk(dimer.wfnB, fchk_files[1])
        written += fchk_files

    return written


def _task_sapt_energy(dimer: Dimer, config: RunConfig, out_dir: Path, settings: dict):
    """Calculates the SAPT interaction energy components."""

    return calc_sapt_energy(
        dimer,
        results=str(out_dir / settings["results"]),
        response=settings["response"],
    )


def _task_property(dimer: Dimer, config: RunConfig, out_dir: Path, settings: dict):
    """Calculates an interaction-induced property."""

    return calc_property(
        dimer,
        settings["prop"],
        results=str(out_dir / settings["results"]),
        response=settings["response"],
    )


def _task_densities(dimer: Dimer, config: RunConfig, out_dir: Path, settings: dict):
    """Calculates the interaction-induced density matrices; the bundle records the run
    configuration and the basis overrides applied."""

    return calc_densities(
        dimer,
        orbital_basis=settings["orbital_basis"],
        save_cubes=settings["save_cubes"],
        cubes_dir=str(out_dir),
        save_matrices=settings["save_matrices"],
        matrices_dir=str(out_dir),
        matrix_format=settings["matrix_format"],
        metadata={
            "run": {
                "config_file": config.source,
                "config": config.to_dict(),
                "basis_overrides_applied": config.applied_basis_overrides(),
            }
        },
    )


# Task functions, called in this order for the tasks enabled in the configuration.
TASKS = {
    "monomer_references": _task_monomer_references,
    "sapt_energy": _task_sapt_energy,
    "property": _task_property,
    "densities": _task_densities,
}


def as_run_config(config: RunConfig | str | os.PathLike | Mapping) -> RunConfig:
    """Returns `config` as a `RunConfig`.

    Args:
        config (RunConfig | str | os.PathLike | Mapping): A resolved configuration, the path
            of a configuration file, or a raw configuration mapping (relative paths in it are
            resolved against the current working directory).

    Returns:
        RunConfig: The resolved configuration.
    """

    if isinstance(config, RunConfig):
        return config

    if isinstance(config, (str, os.PathLike)):
        path = Path(config)
        return resolve_config(
            load_config(path), base_dir=path.resolve().parent, source=str(path)
        )

    return resolve_config(config)


def run(
    config: RunConfig | str | os.PathLike | Mapping, geometry_file: str | os.PathLike
) -> dict:
    """Runs the tasks of a configuration for one dimer geometry.

    Every file is written into the configured output directory, and a record of the run
    (configuration, Psi4 options, versions, timings, the geometry's fragments) into
    `propsapt_run.json` there.

    Args:
        config (RunConfig | str | os.PathLike | Mapping): Run configuration, see
            `as_run_config`.
        geometry_file (str | os.PathLike): Path to the dimer geometry in Psi4 format
            (e.g. a .p4m file).

    Returns:
        dict: The result of each task that ran, keyed by task name.
    """

    config = as_run_config(config)

    geometry_path = Path(geometry_file).resolve()
    geometry = geometry_path.read_text(encoding="utf-8")

    out_dir = Path(config.output_dir(geometry_path.stem)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.datetime.now().astimezone()

    try:
        applied_overrides = setup_psi4(config, str(out_dir / config.psi4_output))

        with CalcTimer("propSAPT runner"):
            dimer = build_dimer(config, geometry)

            results = {}
            for name, settings in config.tasks.items():
                with CalcTimer(f"runner task '{name}'"):
                    results[name] = TASKS[name](dimer, config, out_dir, settings)

        record = {
            "geometry_file": str(geometry_path),
            "config_file": config.source,
            "config": config.to_dict(),
            "psi4_options": config.effective_psi4_options(),
            "basis_overrides_applied": applied_overrides,
            "fragments": parse_fragments(geometry).to_dict(),
            "tasks": list(results),
            "prop_sapt_version": __version__,
            "psi4_version": psi4.__version__,
            "started": started.isoformat(timespec="seconds"),
            "finished": datetime.datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
        }
        with open(out_dir / PROVENANCE_FILENAME, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, default=str)
            handle.write("\n")

    finally:
        psi4.core.clean()

    return results
