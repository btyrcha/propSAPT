"""Allows `python -m prop_sapt run ...`, the same as the `prop-sapt` command."""

import sys

from .cli import main

sys.exit(main())
