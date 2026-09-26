"""Sphinx configuration shared by local builds, CI and Read the Docs."""
import ast
from pathlib import Path

project = "fABBA"
author = "Stefan Güttel, Xinye Chen"
copyright = "2026, Stefan Güttel, Xinye Chen"
# Read the literal version without importing optional scientific backends.
_tree = ast.parse((Path(__file__).resolve().parents[2] / "fABBA/__init__.py").read_text())
release = next(ast.literal_eval(node.value) for node in _tree.body
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets))
extensions = ["sphinx.ext.autodoc", "sphinx.ext.doctest", "sphinx.ext.mathjax", "sphinx.ext.viewcode"]
html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 3}
exclude_patterns = []
html_static_path = ["_static"]
