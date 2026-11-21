"""Phase 7 smoke tests: end-to-end benchmark runner (`--small`) and report/figure generation."""

from __future__ import annotations

from pathlib import Path

from nhc.bench.report import generate_figures, write_summary_markdown
from nhc.bench.run import run_phase7_benchmark


def test_phase7_small_benchmark_and_report(tmp_path: Path) -> None:
    out_json = tmp_path / "bench_small.json"
    rep = run_phase7_benchmark(output_path=out_json, small=True)

    assert out_json.exists()
    for hid in ("H1", "H2", "H3", "H4", "H5"):
        assert hid in rep["verdicts"]
        assert rep["verdicts"][hid]["verdict"] in {
            "CONFIRMED",
            "REFUTED",
            "INCONCLUSIVE",
        }

    for eng in (
        "E0_cos_cold",
        "E1_cos_warm",
        "E2_sobolev_64x4",
        "E3_pointwise_64x4",
        "E4_inverse_ens5",
        "E5_hybrid_5cos",
    ):
        assert eng in rep["summary_table"]
        assert rep["summary_table"][eng]["latency_p50_ms"] > 0.0

    md_path = tmp_path / "summary.md"
    fig_dir = tmp_path / "figs"
    write_summary_markdown(rep, md_path)
    figs = generate_figures(rep, None, fig_dir)
    assert md_path.exists()
    assert len(figs) >= 3
    for f in figs:
        assert f.exists() and f.stat().st_size > 5_000
