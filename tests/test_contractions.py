"""Static checks of the contraction strings in the package source."""

import re
from collections import Counter
from pathlib import Path

import prop_sapt

SUBSCRIPTS = re.compile(r'(?:oe\.contract|np\.einsum)\(\s*"([^"]+)"')


def explicit_contractions() -> list[tuple[str, str]]:
    """Location and subscripts of every contraction with an explicit output ("->")."""

    root = Path(prop_sapt.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        for match in SUBSCRIPTS.finditer(text):
            subscripts = match.group(1).replace(" ", "")
            if "->" in subscripts:
                line = text.count("\n", 0, match.start()) + 1
                found.append((f"{path.relative_to(root)}:{line}", subscripts))

    return found


def test_summed_indices_are_contracted():
    """Every index summed over appears in at least two operands. An index in one operand
    only sums that tensor over its orbitals, which is not invariant under rotations of
    the orbitals: it is a mistyped index."""

    contractions = explicit_contractions()
    # the pattern still finds the contractions of every component tree
    scanned = {where.split("/")[0] for where, _ in contractions}
    assert {
        "sapt_components",
        "properties_components",
        "densities_components",
    } <= scanned

    summed_alone = []
    for where, subscripts in contractions:
        inputs, output = subscripts.split("->")
        counts = Counter(inputs.replace(",", ""))
        lonely = sorted(c for c, n in counts.items() if n == 1 and c not in output)
        if lonely:
            summed_alone.append(f"{where} {subscripts}: {''.join(lonely)}")

    assert not summed_alone, "\n".join(summed_alone)
