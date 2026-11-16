"""Smoke test of the whole Streamlit page on the real data (skipped when data/ is empty)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import loader

APP = Path(__file__).parent.parent / "app.py"
pytestmark = pytest.mark.skipif(not loader.list_cities(), reason="no CSV files in data/")


def test_app_renders_without_errors():
    app = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not app.exception
    assert not app.error
    assert len(app.tabs) == 4
