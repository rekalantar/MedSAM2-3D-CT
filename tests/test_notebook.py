"""Static checks on the tutorial notebook.

A broken notebook is only discovered when someone spends ten minutes installing
MedSAM2 on a GPU runtime, so these run in CI instead.
"""
from __future__ import annotations

import ast
import builtins
import json
import pathlib

import pytest

NOTEBOOK = pathlib.Path(__file__).parent.parent / "tutorial" / "medsam2_3d_ct.ipynb"


def _code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text())
    return [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]


def _strip_magics(source: str) -> str:
    """Comment out shell (!) and magic (%) lines so the cell is parseable Python."""
    source = source.replace("%%capture", "#")
    return "\n".join(
        ("# " + line) if line.lstrip().startswith(("!", "%")) else line
        for line in source.splitlines()
    )


def test_notebook_exists_and_is_valid_json():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert len(_code_cells()) > 5


def test_every_cell_parses():
    for i, source in enumerate(_code_cells(), 1):
        try:
            ast.parse(_strip_magics(source))
        except SyntaxError as exc:                       # pragma: no cover
            pytest.fail(f"cell {i} line {exc.lineno}: {exc.msg}")


def test_is_standalone():
    """The tutorial must not depend on this package being installed.

    Readers should be able to run it having installed only MedSAM2. It also removes a
    whole class of failure where pip serves a stale build of our own wrapper.
    """
    for i, source in enumerate(_code_cells(), 1):
        tree = ast.parse(_strip_magics(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("medsam2_ct"), \
                    f"cell {i} imports medsam2_ct; the notebook must be self-contained"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("medsam2_ct"), \
                        f"cell {i} imports medsam2_ct; the notebook must be self-contained"


def test_defines_the_functions_it_uses():
    """Self-contained means the helpers are defined in the notebook itself."""
    defined = {
        node.name
        for source in _code_cells()
        for node in ast.parse(_strip_magics(source)).body
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("load_nifti", "dice", "window_hu", "to_frames", "largest_component"):
        assert name in defined, f"notebook should define {name}() itself"


def test_no_undefined_names():
    """Names are bound before use, reading cells top to bottom."""
    defined = set(dir(builtins)) | {"__name__", "__file__", "_"}
    problems = []

    class Binder(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                defined.add(node.id)

        def visit_FunctionDef(self, node):
            defined.add(node.name)          # body is a separate scope

        def visit_ClassDef(self, node):
            defined.add(node.name)

        def visit_Import(self, node):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split(".")[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                defined.add(alias.asname or alias.name)

    class Loader(ast.NodeVisitor):
        def __init__(self, index):
            self.index = index

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id not in defined:
                problems.append(f"cell {self.index}: undefined {node.id!r} (line {node.lineno})")

        # these introduce their own scopes
        def visit_FunctionDef(self, node): pass
        def visit_Lambda(self, node): pass
        def visit_ListComp(self, node): pass
        def visit_SetComp(self, node): pass
        def visit_DictComp(self, node): pass
        def visit_GeneratorExp(self, node): pass

    for i, source in enumerate(_code_cells(), 1):
        tree = ast.parse(_strip_magics(source))
        Binder().visit(tree)
        Loader(i).visit(tree)

    assert not problems, "\n".join(dict.fromkeys(problems))


def test_does_not_shadow_pil_image():
    """`from IPython.display import Image` breaks any later PIL use."""
    joined = "\n".join(_code_cells())
    assert "from IPython.display import Image" not in joined


def test_only_external_install_is_medsam2():
    """The install cell should clone MedSAM2 and nothing of ours."""
    joined = "\n".join(_code_cells())
    assert "git clone -q https://github.com/bowang-lab/MedSAM2.git" in joined
    assert "MedSAM2-3D-CT.git" not in joined, \
        "the tutorial should not need this repo installed"


def test_has_a_colab_badge():
    """The badge is how anyone browsing the repo actually runs this."""
    nb = json.loads(NOTEBOOK.read_text())
    markdown = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "colab-badge.svg" in markdown
    assert "colab.research.google.com/github/rekalantar/MedSAM2-3D-CT" in markdown
