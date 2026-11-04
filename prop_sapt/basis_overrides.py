"""
Per-element basis set overrides.

Some elements lack the auxiliary basis sets that pair with the orbital basis in Psi4 (e.g.
aug-cc-pvtz has no -jkfit / -ri set for Zn). For those elements another basis is substituted,
leaving every other element on the default. Overrides are given element-first, as a mapping
or as a JSON file holding one::

    {
        "Zn": {"DF_BASIS_SCF": "def2-tzvpd-ri", "DF_BASIS_SAPT": "def2-tzvpd-ri"}
    }

where each key under an element is a Psi4 basis option ("BASIS", "DF_BASIS_SCF",
"DF_BASIS_SAPT", ...). An override is applied only for a basis option that the calculation
sets, so one file can serve several kinds of calculation: entries for options the
calculation does not use (e.g. "DF_BASIS_MP2" in a propSAPT run) are skipped.
"""

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

# Basis set names that `psi4.basis_helper` can assign to an element.
ASSIGNABLE_BASIS_NAME = re.compile(r"^[-+*()\w]+$")


def load_basis_overrides(
    spec: str | os.PathLike | Mapping | None,
    base_dir: str | os.PathLike | None = None,
) -> dict[str, dict[str, str]]:
    """Reads per-element basis overrides and groups them by basis option.

    Args:
        spec (str | os.PathLike | Mapping | None): Path to a JSON file, or a mapping
            `{element: {basis_option: basis_name}}`. None means no overrides.
        base_dir (str | os.PathLike | None, optional): Directory that a relative path is
            resolved against. Defaults to the current working directory.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is not valid JSON, or a basis set name cannot be assigned to
            an element by Psi4 (e.g. a "bse:" name).
        TypeError: If the overrides are not a mapping of mappings of strings.

    Returns:
        dict[str, dict[str, str]]: Overrides as `{BASIS_OPTION: {element: basis_name}}`, with
            basis options in upper case.
    """

    if spec is None:
        return {}

    if isinstance(spec, (str, os.PathLike)):
        path = Path(spec)
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        with open(path, "r", encoding="utf-8") as handle:
            spec = json.load(handle)

    if not isinstance(spec, Mapping):
        raise TypeError(
            "Basis overrides must map element symbols to {basis_option: basis_name}."
        )

    by_option = {}
    for element, option_map in spec.items():
        if not isinstance(element, str) or not isinstance(option_map, Mapping):
            raise TypeError(
                f"Basis overrides for '{element}' must be a mapping of "
                "{basis_option: basis_name}."
            )
        for basis_option, basis_name in option_map.items():
            if not isinstance(basis_option, str) or not isinstance(basis_name, str):
                raise TypeError(
                    f"Basis override '{element}: {basis_option}' must map a basis option "
                    "name to a basis set name."
                )
            _check_assignable(element, basis_name)
            by_option.setdefault(basis_option.upper(), {})[element] = basis_name

    return by_option


def _check_assignable(element: str, basis_name: str) -> None:
    """Checks that Psi4 can assign `basis_name` to an element in a basis block."""

    if ASSIGNABLE_BASIS_NAME.match(basis_name):
        return

    if basis_name.lower().startswith("bse:"):
        raise ValueError(
            f"'{basis_name}' for {element}: 'bse:' names cannot be assigned to single "
            "elements. Use the name for the whole molecule ('basis', 'df_basis_scf', "
            "'df_basis_sapt'), or write the basis set into a file with "
            "`prop-sapt basis fetch bse ...` and use that file's name with 'basis_dirs'."
        )

    raise ValueError(
        f"'{basis_name}' for {element}: basis set names in overrides can only contain "
        "letters, digits and the characters - + * ( ) _."
    )


def apply_basis_overrides(
    options: Mapping, overrides: Mapping[str, Mapping[str, str]]
) -> list[str]:
    """Registers per-element basis overrides with Psi4.

    Call after `psi4.set_options(options)`. For each basis option that is both overridden
    and set in `options`, a custom basis is registered with `psi4.basis_helper`: every
    element gets `options[basis_option]` and the overridden elements get their substitutes.
    `psi4.basis_helper` then points the basis option at the custom basis.

    Args:
        options (Mapping): Psi4 options already passed to `psi4.set_options`.
        overrides (Mapping[str, Mapping[str, str]]): Overrides as returned by
            `load_basis_overrides`.

    Returns:
        list[str]: The basis options that were overridden.
    """

    import psi4

    # basis options set in `options`, case-insensitively
    option_keys = {key.upper(): key for key in options}

    applied = []
    for basis_option, element_map in overrides.items():
        basis_option = basis_option.upper()
        option_key = option_keys.get(basis_option)

        if option_key is None:
            psi4.core.print_out(
                f"\nBasis overrides: skipping {basis_option}, not used by this calculation.\n"
            )
            continue

        default_basis = options[option_key]
        assign_lines = [f"assign {default_basis}"]
        assign_lines += [
            f"assign {element} {name}" for element, name in element_map.items()
        ]

        psi4.basis_helper(
            "\n".join(assign_lines),
            name=f"propsapt_override_{basis_option.lower()}",
            key=basis_option,
        )
        applied.append(basis_option)

        substitutes = ", ".join(f"{el} -> {name}" for el, name in element_map.items())
        psi4.core.print_out(
            f"\nBasis overrides: {basis_option} defaults to {default_basis}; {substitutes}.\n"
        )

    return applied
