from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_v4_4_evaluation_framework.py")
SPEC = importlib.util.spec_from_file_location("evaluation_framework", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
framework = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(framework)


def test_interval_slug_uses_exact_evaluation_dates() -> None:
    assert (
        framework.interval_slug("2026-07-08 23:52:15", "2026-08-07 03:21:45")
        == "20260708T235215__20260807T032145"
    )


def test_redirect_preserves_query_and_hash() -> None:
    html = framework.redirect_html("../target/index.html", "结果")
    assert "location.search+location.hash" in html
    assert "../target/index.html" in html


def test_role_projection_and_reconstruction_preserve_rows() -> None:
    rows = [
        {
            "combo_id": "c1",
            "e": 320,
            "source_trade_count": 10,
            "source_cost_total_return": 0.1,
            "target_trade_count": 8,
            "target_cost_total_return": 0.2,
            "rank_percentile_change": 0.3,
        }
    ]
    source = framework.role_projection(
        rows,
        evaluation_id="K200_A",
        prefix="source",
        instrument_id="K200",
        display_name="K200",
        start="2026-01-01 00:00:00",
        end="2026-01-02 00:00:00",
    )
    target = framework.role_projection(
        rows,
        evaluation_id="SI_B",
        prefix="target",
        instrument_id="SImain",
        display_name="SI",
        start="2026-02-01 00:00:00",
        end="2026-02-02 00:00:00",
    )
    base = framework.strip_role_metrics(rows[0], ("source", "target"))
    roles = [
        {"roleKey": "source", "outputPrefix": "source", "evaluationId": "K200_A"},
        {"roleKey": "target", "outputPrefix": "target", "evaluationId": "SI_B"},
    ]
    rebuilt = framework.reconstruct_rows(
        [base], roles, {"K200_A": source, "SI_B": target}
    )
    assert rebuilt == rows
