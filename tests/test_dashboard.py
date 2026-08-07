"""Dashboard smoke test.

Streamlit's own AppTest harness executes the script exactly as the server would,
so this catches import errors, bad column references and rendering exceptions
without a browser. Skipped when no run artifacts are present, which is the case
in a fresh clone before the pipeline has been executed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="install with: pip install -e '.[app]'")
pytest.importorskip("plotly")


def _has_artifacts() -> bool:
    return any(
        (Path(root) / d.name / "model_metrics.csv").exists()
        for root in ("artifacts", "demo_artifacts")
        if Path(root).exists()
        for d in Path(root).iterdir()
        if d.is_dir()
    )


@pytest.mark.skipif(not _has_artifacts(), reason="no completed run to render")
@pytest.mark.slow
def test_dashboard_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("dashboard/app.py", default_timeout=240)
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.tabs) >= 5, "expected the full set of console tabs"
    assert at.dataframe, "expected at least one results table"

    text = " ".join(m.value for m in at.markdown if isinstance(m.value, str))
    assert "StockRank" in text
    assert "Verdict" in text
