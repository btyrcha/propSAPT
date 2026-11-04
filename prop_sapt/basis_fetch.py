"""
Fetch basis sets missing from Psi4's library into Psi4-format files (`prop-sapt basis fetch`).

The files are written once, e.g. on a workstation or login node, into a directory that runs
then list in `basis_dirs` (see `prop_sapt.config`), so calculations read fixed inputs and
need no network or extra packages. Two sources are available, each through an optional
package that is installed separately and only where this command is used:

    ccrepo  ccRepo, http://www.grant-hill.group.shef.ac.uk/ccrepo/, through ccrepo_utility
            (its dependencies are not installed with it, so they are listed as well):
            pip install git+https://github.com/Sheffield-Theoretical-Chemistry/ccrepo_utility \
                colorlog tqdm requests
    bse     Basis Set Exchange, https://www.basissetexchange.org, through basis_set_exchange:
            pip install basis_set_exchange

Each file is named after its basis set, by default "<source>-<basis set>", e.g.
"ccrepo-aug-cc-pVTZ_MP2FIT" in `ccrepo-aug-cc-pvtz_mp2fit.gbs`. A file must not take the name
of a basis set in Psi4's library: Psi4 reads the first file of a name it finds, so a file
holding a few elements would hide the library basis set for every other element.
"""

import datetime
import re
import socket
from collections.abc import Sequence
from pathlib import Path

import psi4
import qcelemental as qcel
from psi4.driver.qcdb.libmintsbasisset import BasisSet

from .basis_overrides import ASSIGNABLE_BASIS_NAME
from .utils import package_version

CCREPO_URL = "http://www.grant-hill.group.shef.ac.uk/ccrepo/"
CCREPO_INSTALL = (
    "pip install git+https://github.com/Sheffield-Theoretical-Chemistry/ccrepo_utility "
    "colorlog tqdm requests"
)
BSE_URL = "https://www.basissetexchange.org"
BSE_INSTALL = "pip install basis_set_exchange (or: conda install -c conda-forge basis_set_exchange)"

# ccrepo_utility downloads its catalogue when imported, without a timeout of its own, and
# falls back to the copy it ships with; this bounds the download.
CCREPO_IMPORT_TIMEOUT = 30  # seconds

# an element entry of a basis set in Psi4 (Gaussian94) format, e.g. "He     0"
_ENTRY_RE = re.compile(r"^\s*([A-Za-z]{1,3})\s+0\s*$", re.MULTILINE)


class BasisFetchError(RuntimeError):
    """Raised when a basis set cannot be fetched or written."""


def _fetch_ccrepo(name: str, elements: Sequence[str]) -> tuple[str, str]:
    """Returns a ccRepo basis set in Psi4 format and a description of its source."""

    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(CCREPO_IMPORT_TIMEOUT)
    try:
        from ccrepo.fetch import fetch_basis
    except ImportError as err:
        # ccrepo_utility does not declare its dependencies, so they may be missing on their own
        if err.name is None or err.name.split(".")[0] == "ccrepo":
            missing = "ccrepo_utility"
        else:
            missing = f"its dependency {err.name}"
        raise BasisFetchError(
            f"The ccrepo source needs ccrepo_utility, but {missing} is not installed. "
            f"Install it with:\n  {CCREPO_INSTALL}"
        ) from err
    finally:
        socket.setdefaulttimeout(previous_timeout)

    try:
        text = fetch_basis(list(elements), name, format="psi4")
    except ValueError as err:
        raise BasisFetchError(f"ccRepo: {err}") from err

    source = f"ccRepo ({CCREPO_URL}) via ccrepo_utility {package_version('ccrepo')}"

    # ccRepo basis sets use spherical functions, which Psi4 also assumes without this line
    return "spherical\n\n" + text.strip() + "\n", source


def _fetch_bse(name: str, elements: Sequence[str]) -> tuple[str, str]:
    """Returns a Basis Set Exchange basis set in Psi4 format and a description of its source."""

    try:
        import basis_set_exchange as bse
    except ImportError as err:
        raise BasisFetchError(
            f"The bse source needs basis_set_exchange ({err}). Install it with:\n  {BSE_INSTALL}"
        ) from err

    try:
        text = bse.get_basis(name, elements=list(elements), fmt="psi4")
    except KeyError as err:
        raise BasisFetchError(f"Basis Set Exchange: {err.args[0]}") from err

    return (
        text.strip() + "\n",
        f"Basis Set Exchange ({BSE_URL}) via basis_set_exchange {bse.version()}",
    )


# Basis set sources, by the name used on the command line. Each takes the basis set name and
# the element symbols, and returns the basis set in Psi4 format and a description of its source.
SOURCES = {
    "ccrepo": _fetch_ccrepo,
    "bse": _fetch_bse,
}


def fetch_basis_file(
    source: str,
    name: str,
    elements: Sequence[str],
    directory: str | Path = "basis",
    as_name: str | None = None,
    force: bool = False,
) -> tuple[Path, str]:
    """Fetches a basis set and writes it into a Psi4-format file.

    Args:
        source (str): Source of the basis set, a key of `SOURCES` ("ccrepo" or "bse").
        name (str): Name of the basis set at the source, e.g. "aug-cc-pVTZ_MP2FIT".
        elements (Sequence[str]): Element symbols to fetch the basis set for.
        directory (str | Path, optional): Directory to write the file into. Defaults to "basis".
        as_name (str | None, optional): Name of the basis set in propSAPT/Psi4, which also names
            the file. Defaults to "<source>-<name>".
        force (bool, optional): Whether to overwrite an existing file. Defaults to False.

    Raises:
        BasisFetchError: If the source or its package is unavailable, the basis set is missing
            for an element, the name would hide a Psi4 library basis set, or the file exists.

    Returns:
        tuple[Path, str]: Path of the written file and the basis set name to use in a run
            configuration.
    """

    if source not in SOURCES:
        raise BasisFetchError(
            f"Unknown basis set source '{source}'. Available: {', '.join(sorted(SOURCES))}."
        )

    elements = _normalize_elements(elements)

    basis_name = as_name or f"{source}-{name}"
    if not ASSIGNABLE_BASIS_NAME.match(basis_name):
        raise BasisFetchError(
            f"Basis set name '{basis_name}' can only contain letters, digits and the "
            "characters - + * ( ) _; choose another one with --as."
        )

    filename = BasisSet.make_filename(basis_name)
    library_file = Path(psi4.core.get_datadir()) / "basis" / filename
    if library_file.exists():
        raise BasisFetchError(
            f"Psi4's library already has a basis set named '{basis_name}' ({library_file}). "
            "A file of that name would be read instead, hiding the library basis set for "
            "every element not in the file; choose another name with --as."
        )

    path = Path(directory) / filename
    if path.exists() and not force:
        raise BasisFetchError(f"'{path}' already exists; use --force to overwrite it.")

    text, source_description = SOURCES[source](name, elements)

    found = {entry.capitalize() for entry in _ENTRY_RE.findall(text)}
    missing = [element for element in elements if element not in found]
    if missing:
        raise BasisFetchError(
            f"{source}: basis set '{name}' is not available for {', '.join(missing)}; "
            "nothing was written."
        )

    command = f"prop-sapt basis fetch {source} {name} --elements {','.join(elements)}"
    if as_name:
        command += f" --as {as_name}"
    written = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    header = [
        f"! Basis set {basis_name}, written by `{command}`",
        f"! on {written}.",
        f"! Source: {source_description}",
        f"! Basis set at the source: {name}; elements: {', '.join(elements)}",
        "! Please cite the source and the original publications of this basis set.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header) + "\n\n" + text, encoding="utf-8")

    return path, basis_name


def _normalize_elements(elements: Sequence[str]) -> list[str]:
    """Returns the element symbols in standard capitalisation, without blanks or repeats."""

    symbols = []
    for element in elements:
        element = element.strip()
        if not element:
            continue
        try:
            symbol = qcel.periodictable.to_E(qcel.periodictable.to_Z(element))
        except qcel.exceptions.NotAnElementError as err:
            raise BasisFetchError(f"'{element}' is not an element symbol.") from err
        if symbol not in symbols:
            symbols.append(symbol)

    if not symbols:
        raise BasisFetchError("No elements given.")

    return symbols
