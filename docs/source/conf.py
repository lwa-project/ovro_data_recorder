# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../ovro_data_recorder'))


# -- Project information -----------------------------------------------------

project = 'OVRO-LWA Data Recorders'
copyright = '2021, Jayce Dowell'
author = 'Jayce Dowell'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.coverage', 'sphinx.ext.imgmath', 'sphinx.ext.intersphinx'
]

# Add mappings
intersphinx_mapping = {
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Auto-extract some help --------------------------------------------------

import sys
import glob
import subprocess
for script in glob.glob('../../scripts/dr_*.py'):
    outname = os.path.basename(script)
    outname = os.path.splitext(outname)[0]
    outname += '.help'
    with open(outname, 'wb') as fh:
        try:
            output = subprocess.check_output([sys.executable, script, '--help'])
            fh.write(output)
        except (OSError, subprocess.CalledProcessError):
            fh.write(b"Failed to extract help message")
for script in glob.glob('../../services/*.py'):
    outname = os.path.basename(script)
    outname = os.path.splitext(outname)[0]
    outname += '.help'
    with open(outname, 'wb') as fh:
        try:
            output = subprocess.check_output([sys.executable, script, '--help'])
            fh.write(output)
        except (OSError, subprocess.CalledProcessError):
            fh.write(b"Failed to extract help message")
