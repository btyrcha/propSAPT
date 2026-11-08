# Contributing to propSAPT

Thanks for your interest in contributing! Bug reports and questions are welcome as
[GitHub issues](https://github.com/btyrcha/propSAPT/issues). For setup, see the
[installation guide](https://btyrcha.github.io/propSAPT/installation.html) and
`README.md`.

## Making a change

1. Create a development install: the conda environment from `prop-sapt.yaml` and
   `pip install -e . --config-settings editable_mode=compat`.
2. Work on a branch off `main`, in commits that each make one logical change.
3. Add or update tests in `tests/` for what you change. The tests run real Psi4
   calculations; reference values are checked against Psi4's own SAPT or finite field,
   to `1e-8` Eh for energies.
4. Format the Python files you add or change with [black](https://black.readthedocs.io)
   (`black <files>`; it comes with `prop-sapt.yaml`). Before opening a pull request, run
   from the repository root:

   ```bash
   black --check .
   pytest
   flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
   ```

   and, if you changed the docs or docstrings, `sphinx-build -W -b html docs docs/_build/html`.
5. Add an entry under `[Unreleased]` in `CHANGELOG.md` for any change users will notice.
6. Open a pull request against `main`. CI runs the tests, the black check, the
   packaging check and the docs build.

## Releasing

For the maintainer. The version is set in `prop_sapt/_version.py`, and pull requests
don't change it. Between releases `main` carries the next minor version as a
development version (e.g. `0.5.0.dev0`), so results saved from `main` don't claim to
come from a release.

Before 1.0, a patch release (`0.4.1`) only fixes bugs or speeds things up and leaves
every result unchanged. A new option, an API change or a change in any numerical
result, even from a bug fix, makes a minor release (`0.5.0`).

To release version X.Y.Z:

1. On a branch, in one commit: set the version in `prop_sapt/_version.py` and
   `CITATION.cff`, point the install commands in `README.md` and
   `docs/installation.rst` at `vX.Y.Z`, and turn `[Unreleased]` in `CHANGELOG.md` into
   `[X.Y.Z] - YYYY-MM-DD` under a new, empty `[Unreleased]`. A dry run of the Release
   workflow on the branch (`gh workflow run release.yml --ref <branch>`) checks these
   and prints the release notes.
2. Merge the pull request, then tag the merge commit on `main` and push the tag
   (`git tag vX.Y.Z && git push origin vX.Y.Z`). The Release workflow creates the
   GitHub release, and Zenodo archives it under a new version DOI.
3. Set `main` to the next development version, e.g. `0.6.0.dev0`.

## Logging AI-assisted work

This project keeps a running log of AI coding assistance in `AI-USAGE.md`, so that
accurate AI-involvement disclosures can be produced later.

If you use an AI coding assistant (e.g. Claude Code, Copilot, or similar) to write or modify
code, tests, or documentation in this repository, please add a short entry to `AI-USAGE.md`
as part of your change:

- Append under the `## Log` section, newest at the bottom.
- Group entries by date using a `### YYYY-MM-DD` heading.
- Add one line per pull request or task, saying what it was and what was produced or
  changed.
- Keep it factual and simple — this is a disclosure aid, not a changelog.
- If the use changes what the summary at the top of `AI-USAGE.md` says, update it too.

Trivial actions (reading files, running tests, non-code Q&A) do not need to be logged.
