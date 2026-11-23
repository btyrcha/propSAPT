"""Static checks of how the package imports itself."""

import ast
import re
from pathlib import Path

import prop_sapt

ABSOLUTE_IMPORT = re.compile(r"^\s*(?:from|import)\s+prop_sapt\b", re.MULTILINE)

#: The modules that may import Psi4: its backend, and those not yet working through the
#: backend contract (the analysis modules, bundles, runner, configuration and finite
#: field). The SAPT core is not among them.
PSI4_MODULES = {
    "artifacts.py",
    "backends/psi4_backend.py",
    "basis_fetch.py",
    "basis_overrides.py",
    "checks.py",
    "cli.py",
    "cubes.py",
    "evaluate.py",
    "finite_field/core.py",
    "finite_field/utils.py",
    "forces.py",
    "fragments.py",
    "molecule.py",
    "partition.py",
    "runner.py",
    "trexio_export.py",
}


def test_package_imports_itself_relatively():
    """Modules import the package relatively (`from ..molecule import Dimer`). An
    absolute import resolves to whichever `prop_sapt` is on `sys.path`, and
    `from prop_sapt import Dimer` inside the package works only while `__init__`
    happens to import `molecule` first."""

    root = Path(prop_sapt.__file__).parent
    absolute = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        for match in ABSOLUTE_IMPORT.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            absolute.append(f"{path.relative_to(root)}:{line}")

    assert absolute == []


def test_term_packages_export_their_functions():
    """Every `*_terms` package re-exports the functions of its modules."""

    root = Path(prop_sapt.__file__).parent
    packages = sorted(root.glob("*_components/*_terms"))
    assert len(packages) == 3

    for package in packages:
        init = (package / "__init__.py").read_text()
        for module in sorted(package.glob("*.py")):
            if module.name == "__init__.py":
                continue
            for name in re.findall(r"^def (\w+)\(", module.read_text(), re.MULTILINE):
                assert f'"{name}"' in init, f"{package.name} does not export {name}"


def _imports_psi4(path: Path) -> bool:
    """Whether a module imports Psi4 anywhere, functions included."""

    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        if any(name.split(".")[0] == "psi4" for name in names):
            return True

    return False


def test_only_psi4_facing_modules_import_psi4():
    """The SAPT core (`helper_SAPT`, the contractions, the energy, property and density
    drivers, CPSCF) reaches Psi4 only through the backend, so another backend can take
    its place. The list of Psi4 modules is exact: it shrinks as modules move onto the
    backend contract."""

    root = Path(prop_sapt.__file__).parent
    importing = {
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*.py"))
        if _imports_psi4(path)
    }

    assert sorted(importing - PSI4_MODULES) == []
    assert sorted(PSI4_MODULES - importing) == []
