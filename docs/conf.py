# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
from importlib.metadata import version

# Define path to the code to be documented **relative to where conf.py (this file) is kept**
sys.path.insert(0, os.path.abspath("../src/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "lsdb"
copyright = "2025, LINCC Frameworks"
author = "LINCC Frameworks"
release = version("lsdb")
# for example take major/minor
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinx_design",
    "sphinx.ext.autosummary",
]

extensions.append("nbsphinx")

# -- sphinx-copybutton configuration ----------------------------------------
extensions.append("sphinx_copybutton")
## sets up the expected prompt text from console blocks, and excludes it from
## the text that goes into the clipboard.
copybutton_exclude = ".linenos, .gp"
copybutton_prompt_text = ">> "

## lets us suppress the copy button on select code blocks.
copybutton_selector = "div:not(.no-copybutton) > div.highlight > pre"

graphviz_output_format = "svg"

templates_path = []
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# This assumes that sphinx-build is called from the root directory
master_doc = "index"
# Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = False
# Remove namespaces from class/method signatures
add_module_names = False


templates_path = ["_templates"]
# Hide full module path in navigation
modindex_common_prefix = ["lsdb.", "lsdb.catalog", "lsdb.catalog.dataset."]
# Customize display of autosummary entries
autosummary_imported_members = True

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_logo = "_static/lincc_logo.png"
html_title = "LSDB"
html_context = {"default_mode": "light"}

pygments_style = "sphinx"

# Cross-link hats documentation from the API reference:
# https://docs.readthedocs.io/en/stable/guides/intersphinx.html
intersphinx_mapping = {
    "hats": ("http://hats.readthedocs.io/en/stable/", None),
}
