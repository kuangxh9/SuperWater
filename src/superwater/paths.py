"""Resolved paths to data shipped inside the installed package.

Using package-relative paths (rather than CWD-relative ones) keeps the CLI tools
working regardless of where they are invoked from.
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Placeholder water files (_water.pdb / _water.mol2) used as a template when
# organizing protein-only datasets for inference. Water positions are predicted,
# so the content is irrelevant -- only the file's presence and naming matter.
DUMMY_WATER_DIR = os.path.join(PACKAGE_DIR, "data", "dummy_water")
