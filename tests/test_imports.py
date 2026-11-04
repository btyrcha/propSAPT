"""Static checks of how the package imports itself."""

import re
from pathlib import Path

import prop_sapt

ABSOLUTE_IMPORT = re.compile(r"^\s*(?:from|import)\s+prop_sapt\b", re.MULTILINE)


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
