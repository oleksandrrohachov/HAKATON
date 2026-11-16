import ast
from pathlib import Path

import glossary

UI_DIR = Path(__file__).parent.parent / "ui"


def _help_keys_used_in_ui() -> set[str]:
    """Second argument of every ``chart(fig, "key", ...)`` call in ui/*.py."""
    keys = set()
    for path in UI_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "chart" and len(node.args) > 1:
                keys.add(node.args[1].value)
    return keys


def test_all_texts_are_filled():
    assert all(text.strip() for text in [*glossary.TERMS.values(), *glossary.CHART_HELP.values()])


def test_every_chart_in_ui_has_an_explanation():
    used = _help_keys_used_in_ui()
    assert len(used) >= 10
    assert used <= set(glossary.CHART_HELP)
