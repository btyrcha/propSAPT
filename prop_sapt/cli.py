"""
Command-line interface of propSAPT::

    prop-sapt run CONFIG GEOMETRY [GEOMETRY ...] [--memory 32 --threads 8 ...]
    prop-sapt basis fetch SOURCE NAME --elements He,Li [--dir basis]

`run` runs the tasks of a configuration file (see `prop_sapt.config`) for each geometry in
turn; flags override the values in the file. `basis fetch` writes a basis set missing from
Psi4's library into a file for `basis_dirs` (see `prop_sapt.basis_fetch`).
Exit status: 0 on success, 1 if a calculation failed, 2 for an invalid configuration,
command line or basis set request.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

from .config import ConfigError, OVERRIDE_KEYS, load_config, resolve_config


def build_parser() -> argparse.ArgumentParser:
    """Builds the `prop-sapt` argument parser."""

    parser = argparse.ArgumentParser(
        prog="prop-sapt",
        description="Run propSAPT calculations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the tasks of a configuration file for one or more dimer geometries.",
        description=(
            "Run the tasks of a YAML or JSON configuration file for each dimer geometry "
            "in turn, stopping at the first failure. Flags override the file."
        ),
    )
    run_parser.add_argument(
        "config", help="Path to the YAML or JSON configuration file."
    )
    run_parser.add_argument(
        "geometries",
        nargs="+",
        metavar="GEOMETRY",
        help="Path to a dimer geometry file in Psi4 format (e.g. .p4m).",
    )
    run_parser.add_argument(
        "--memory", help="Memory in GB, or a string such as '32 GB' or '500 MB'."
    )
    run_parser.add_argument("--threads", type=int, help="Number of threads.")
    run_parser.add_argument("--reference", help="Reference: RHF or RKS.")
    run_parser.add_argument("--functional", help="DFT functional (RKS only).")
    run_parser.add_argument(
        "--grac-a",
        dest="grac_A",
        type=float,
        help="GRAC shift of monomer A in Hartree (RKS only).",
    )
    run_parser.add_argument(
        "--grac-b",
        dest="grac_B",
        type=float,
        help="GRAC shift of monomer B in Hartree (RKS only).",
    )
    run_parser.add_argument("--basis", help="Orbital basis set.")
    run_parser.add_argument(
        "--df-basis",
        dest="df_basis",
        help="Density-fitting basis stem, extended with -jkfit and -ri (default: --basis).",
    )
    run_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Output directory; '{geometry}' expands to the geometry file stem.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and the Psi4 options, then exit.",
    )

    _add_basis_parser(subparsers)

    return parser


def _add_basis_parser(subparsers) -> None:
    """Adds the `prop-sapt basis` commands."""

    from .basis_fetch import SOURCES

    basis_parser = subparsers.add_parser(
        "basis",
        help="Manage basis set files for the 'basis_dirs' of a run configuration.",
    )
    basis_subparsers = basis_parser.add_subparsers(dest="basis_command", required=True)

    fetch_parser = basis_subparsers.add_parser(
        "fetch",
        help="Write a basis set missing from Psi4's library into a Psi4-format file.",
        description=(
            "Fetch a basis set from ccRepo (needs ccrepo_utility) or the Basis Set Exchange "
            "(needs basis_set_exchange) and write it into DIR/<name>.gbs, for use through "
            "'basis_dirs'. Run it once where those packages are installed; calculations "
            "then read the file."
        ),
    )
    fetch_parser.add_argument(
        "source", choices=sorted(SOURCES), help="Where to fetch from."
    )
    fetch_parser.add_argument(
        "name",
        help="Basis set name at the source, e.g. aug-cc-pVTZ_MP2FIT or def2-universal-jkfit.",
    )
    fetch_parser.add_argument(
        "--elements", required=True, help="Comma-separated element symbols, e.g. He,Li."
    )
    fetch_parser.add_argument(
        "--dir",
        dest="directory",
        default="basis",
        help="Directory to write into (default: basis).",
    )
    fetch_parser.add_argument(
        "--as",
        dest="as_name",
        help="Basis set name to use in propSAPT, which also names the file (default: SOURCE-NAME).",
    )
    fetch_parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing file."
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point of the `prop-sapt` command.

    Args:
        argv (list[str] | None, optional): Command-line arguments without the program name.
            Defaults to `sys.argv[1:]`.

    Returns:
        int: Exit status.
    """

    args = build_parser().parse_args(argv)

    if args.command == "run":
        return _run_command(args)

    if args.command == "basis":
        return _basis_fetch_command(args)

    return 2  # unreachable: argparse requires a known command


def _run_command(args: argparse.Namespace) -> int:
    """Runs `prop-sapt run`."""

    overrides = {key: getattr(args, key) for key in OVERRIDE_KEYS}

    try:
        config_path = Path(args.config)
        config = resolve_config(
            load_config(config_path),
            base_dir=config_path.resolve().parent,
            overrides=overrides,
            source=str(config_path),
        )

        missing = [
            geometry for geometry in args.geometries if not Path(geometry).is_file()
        ]
        if missing:
            raise ConfigError(f"Geometry file(s) not found: {', '.join(missing)}.")

        # every geometry writes the same filenames, so each needs its own directory
        output_dirs = [
            Path(config.output_dir(Path(geometry).stem)).resolve()
            for geometry in args.geometries
        ]
        if len(set(output_dirs)) < len(output_dirs):
            raise ConfigError(
                "Several geometries would write into the same output directory; "
                "put '{geometry}' in 'output.dir' (or --output-dir) and use distinct "
                "geometry file names."
            )

    except ConfigError as err:
        print(f"prop-sapt: error: {err}", file=sys.stderr)
        return 2

    if args.dry_run:
        summary = {
            "config": config.to_dict(),
            "psi4_options": config.effective_psi4_options(),
            "geometries": [
                {
                    "geometry_file": str(Path(geometry).resolve()),
                    "output_dir": str(
                        Path(config.output_dir(Path(geometry).stem)).resolve()
                    ),
                }
                for geometry in args.geometries
            ],
        }
        print(json.dumps(summary, indent=2))
        return 0

    from .runner import run

    for geometry in args.geometries:
        try:
            run(config, geometry)
        except Exception:
            traceback.print_exc()
            print(f"prop-sapt: calculation failed for '{geometry}'.", file=sys.stderr)
            return 1

    return 0


def _basis_fetch_command(args: argparse.Namespace) -> int:
    """Runs `prop-sapt basis fetch`."""

    from .basis_fetch import BasisFetchError, fetch_basis_file

    try:
        path, basis_name = fetch_basis_file(
            args.source,
            args.name,
            args.elements.split(","),
            directory=args.directory,
            as_name=args.as_name,
            force=args.force,
        )
    except BasisFetchError as err:
        print(f"prop-sapt: error: {err}", file=sys.stderr)
        return 2

    lower = basis_name.lower()
    if "jkfit" in lower:
        option = "DF_BASIS_SCF"
    elif "mp2fit" in lower or "rifit" in lower or lower.endswith("-ri"):
        option = "DF_BASIS_SAPT"
    else:
        option = "BASIS"
    element = args.elements.split(",")[0].strip().capitalize()

    print(f"Wrote {path}")
    print(
        f"Use it as basis set '{basis_name}' with the directory listed in the configuration"
    )
    print(f"(relative to the configuration file), e.g. for {element}:")
    print(f"  basis_dirs: [{path.parent}]")
    print(f"  basis_overrides: {{{element}: {{{option}: {basis_name}}}}}")

    return 0
