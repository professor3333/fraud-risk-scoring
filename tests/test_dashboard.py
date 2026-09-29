"""The browser pages treat uploaded and API values as text, and export them safely.

A CSV handed to an analyst is untrusted: any value in it can come back in a response
and be drawn by the dashboard. These tests pin the two properties that keep that safe:
no page parses data as HTML, and the export cannot turn a cell into a formula.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "fraud" / "serve" / "static"
PAGES = sorted(STATIC.glob("*.html"))

# Every DOM API that parses a string as markup. Clearing with `replaceChildren()` covers
# the one legitimate use, so none of these needs to appear at all.
HTML_SINKS = re.compile(r"\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML|document\.write")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_pages_never_parse_data_as_html(page: Path) -> None:
    assert not HTML_SINKS.findall(page.read_text()), f"{page.name} parses a string as HTML"


def test_both_pages_are_checked() -> None:
    assert {p.name for p in PAGES} >= {"dashboard.html", "index.html"}


def _csv_cell() -> str:
    source = (STATIC / "dashboard.html").read_text()
    match = re.search(r"^function csvCell\(v\) \{.*?^\}", source, re.S | re.M)
    assert match, "csvCell not found in dashboard.html"
    return match.group(0)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_export_quotes_cells_and_neutralises_formulas() -> None:
    cases = [
        (123, "123"),
        (-4.5, "-4.5"),  # a number stays a number, sign and all
        (None, ""),
        ("W", "W"),
        ("a,b", '"a,b"'),
        ('say "hi"', '"say ""hi"""'),
        ("two\nlines", '"two\nlines"'),
        ("=HYPERLINK(1)", "'=HYPERLINK(1)"),
        ("+1", "'+1"),
        ("-1", "'-1"),  # text that looks numeric is still text from the upload
        ("@SUM(A1)", "'@SUM(A1)"),
        ("=1,2", '"\'=1,2"'),
        ("<img src=x onerror=alert(1)>", "<img src=x onerror=alert(1)>"),
    ]
    inputs = json.dumps([value for value, _ in cases])
    program = f"{_csv_cell()}\nconsole.log(JSON.stringify({inputs}.map(csvCell)));"
    out = subprocess.run(["node", "-e", program], capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == [expected for _, expected in cases]
