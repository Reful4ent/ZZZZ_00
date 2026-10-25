"""README mentions every subcommand of the real parser (no TF)."""
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_readme_mentions_every_command():
    from metric.cli import _parser
    sub = next(a for a in _parser()._actions if a.dest == "command")
    text = (Path(__file__).resolve().parents[3] / "src/metric/README.md").read_text(encoding="utf-8")
    missing = [c for c in sub.choices if not re.search(rf"python -m metric {c}\b", text)]
    assert not missing, missing
