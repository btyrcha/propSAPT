# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "prop-sapt"
copyright = "2026, Bartosz Tyrcha"
author = "Bartosz Tyrcha"

# The version is defined once, in prop_sapt/_version.py.
_version = {}
with open(
    os.path.join(os.path.dirname(__file__), "..", "prop_sapt", "_version.py")
) as f:
    exec(f.read(), _version)
release = _version["__version__"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# No html_static_path or templates_path: the docs have no static files or templates of
# their own, and a missing html_static_path folder fails sphinx-build -W.

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
}

# Source - https://stackoverflow.com/a/68931907
# Posted by anirudh sharma
# Retrieved 2026-02-16, License - CC BY-SA 4.0

rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight
"""
