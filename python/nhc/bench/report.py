"""Reproducible Phase 7 Report & Figure Generator (T7.3, Gate G7).

Reads `results/benchmark_full.json` (plus optional `results/phase4_gate_g4.json` and
`results/phase6_gate_g6.json`) and generates:
  - `results/summary_tables.md` (Section 7.5 main table, sliced accuracy, error budget,
    Wilcoxon tests, H1–H5 verdicts, hardware/compiler stamp)
  - `results/figures/fig1_latency_vs_accuracy.png`
  - `results/figures/fig2_learning_curve.png`
  - `results/figures/fig3_error_budget.png`
  - `results/figures/fig4_daily_stability.png`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fmt(val: float | None, fmt: str = ".3f") -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "—"
    return format(val, fmt)


def generate_figures(
    bench: dict,
    phase4: dict | None,
    fig_dir: Path,
) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    st = bench["summary_table"]

    # ------------------------------------------------------------------
    # Fig 1: Latency vs Accuracy (Log-scale latency p50 & p99 across engines)
    # ------------------------------------------------------------------
    order = [
        ("Ref_naive", "Ref\n(QL+SciPy)"),
        ("E3_pointwise_64x4", "E3\n(Pointwise)"),
        ("E0_cos_cold", "E0\n(COS cold)"),
        ("E1_cos_warm", "E1\n(COS warm)"),
        ("E5_hybrid_5cos", "E5\n(Hybrid)"),
        ("E2_sobolev_128x4", "E2\n(Sob 128×4)"),
        ("E2_sobolev_64x4", "E2\n(Sob 64×4)"),
        ("E4_inverse_ens5", "E4\n(Inv only)"),
    ]
    labels, p50s, p99s, rmses = [], [], [], []
    for key, lbl in order:
        if key in st:
            labels.append(lbl)
            p50s.append(st[key]["latency_p50_ms"])
            p99s.append(st[key]["latency_p99_ms"])
            rmses.append(st[key]["rmse_iv_vp_mean"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.7))
    x = np.arange(len(labels))
    w = 0.36
    ax1.bar(x - w / 2, p50s, width=w, label="p50 (ms)", color="#2563eb")
    ax1.bar(x + w / 2, p99s, width=w, label="p99 (ms)", color="#93c5fd")
    ax1.axhline(10.0, color="#dc2626", linestyle="--", linewidth=1.2, label="H1 limit (10 ms)")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.0)
    ax1.set_ylabel("Calibration latency (ms, log scale)")
    ax1.set_title(f"Single-Core C++ Latency ({bench['hardware']['cpu']})")
    ax1.legend(loc="upper right", fontsize=8.5)

    colors = [
        "#64748b" if "Ref" in lbl else "#0f766e" if "E2" in lbl else "#2563eb" for lbl in labels
    ]
    ax2.bar(x, rmses, color=colors, width=0.55)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.0)
    ax2.set_ylabel("Mean Exact COS-Audited RMSE (vol pts)")
    ax2.set_title(f"Calibration Fit Quality ({bench['data_source']})")

    fig.tight_layout()
    p1 = fig_dir / "fig1_latency_vs_accuracy.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    generated.append(p1)

    # ------------------------------------------------------------------
    # Fig 2: Phase 4 5-Seed Learning Curve (Baseline vs Sobolev)
    # ------------------------------------------------------------------
    if phase4 and "t43_learning_curve" in phase4:
        lc = phase4["t43_learning_curve"]
        ns = np.array([r["n_train"] for r in lc["baseline"]], dtype=np.float64) / 1000.0
        b_rmse = np.array([r["in_domain_normal_emp"]["rmse_iv_vp"]["mean"] for r in lc["baseline"]])
        b_rmse_sd = np.array(
            [r["in_domain_normal_emp"]["rmse_iv_vp"]["std"] for r in lc["baseline"]]
        )
        s_rmse = np.array([r["in_domain_normal_emp"]["rmse_iv_vp"]["mean"] for r in lc["sobolev"]])
        s_rmse_sd = np.array(
            [r["in_domain_normal_emp"]["rmse_iv_vp"]["std"] for r in lc["sobolev"]]
        )

        b_jac = (
            np.array(
                [r["in_domain_normal_emp"]["jac_rel_err_median"]["mean"] for r in lc["baseline"]]
            )
            * 100.0
        )
        b_jac_sd = (
            np.array(
                [r["in_domain_normal_emp"]["jac_rel_err_median"]["std"] for r in lc["baseline"]]
            )
            * 100.0
        )
        s_jac = (
            np.array(
                [r["in_domain_normal_emp"]["jac_rel_err_median"]["mean"] for r in lc["sobolev"]]
            )
            * 100.0
        )
        s_jac_sd = (
            np.array(
                [r["in_domain_normal_emp"]["jac_rel_err_median"]["std"] for r in lc["sobolev"]]
            )
            * 100.0
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        ax1.errorbar(
            ns,
            b_rmse,
            yerr=b_rmse_sd,
            marker="o",
            capsize=4,
            label="Baseline MLP (64×4)",
            color="#64748b",
        )
        ax1.errorbar(
            ns,
            s_rmse,
            yerr=s_rmse_sd,
            marker="s",
            capsize=4,
            label="Sobolev MLP (64×4, λ=1.0)",
            color="#0f766e",
        )
        ax1.axhline(
            0.05, color="#dc2626", linestyle="--", linewidth=1.1, label="Gate G4 target (0.05 vp)"
        )
        ax1.set_xscale("log")
        ax1.set_xlabel("Training Surfaces (thousands, log scale)")
        ax1.set_ylabel("In-Domain Test RMSE (vol pts, 5 seeds ± 1σ)")
        ax1.set_title("Surrogate IV Accuracy Learning Curve")
        ax1.legend(fontsize=8.5)

        ax2.errorbar(
            ns,
            b_jac,
            yerr=b_jac_sd,
            marker="o",
            capsize=4,
            label="Baseline MLP (64×4)",
            color="#64748b",
        )
        ax2.errorbar(
            ns,
            s_jac,
            yerr=s_jac_sd,
            marker="s",
            capsize=4,
            label="Sobolev MLP (64×4, λ=1.0)",
            color="#0f766e",
        )
        ax2.set_xscale("log")
        ax2.set_xlabel("Training Surfaces (thousands, log scale)")
        ax2.set_ylabel("Median Relative Jacobian Error (%, 5 seeds ± 1σ)")
        ax2.set_title("Sobolev Jacobian Accuracy Learning Curve")
        ax2.legend(fontsize=8.5)

        fig.tight_layout()
        p2 = fig_dir / "fig2_learning_curve.png"
        fig.savefig(p2, dpi=160)
        plt.close(fig)
        generated.append(p2)

    # ------------------------------------------------------------------
    # Fig 3: H4 Error Budget Decomposition
    # ------------------------------------------------------------------
    eb = bench["error_budget"]
    eb_names = [
        "(a) Surrogate approx.\n(Sobolev 64×4)",
        "(b) Interp. operator M\n(spline + total var)",
        "(c) Optimizer multi-start\n(10 cold starts σ)",
        "(e) Bid-ask / quote\nmicrostructure (est.)",
        "(d) Structural Heston\nmisspecification",
    ]
    eb_vals = [
        eb["a_surrogate_rmse_vp"],
        eb["b_interp_M_rmse_vp"],
        eb["c_optimizer_multistart_std_vp"],
        eb["e_bidask_microstructure_est_vp"],
        eb["d_structural_heston_vs_market_rmse_vp"],
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    y = np.arange(len(eb_names))
    bar_colors = ["#0f766e", "#0284c7", "#6366f1", "#f59e0b", "#dc2626"]
    bars = ax.barh(y, eb_vals, color=bar_colors, height=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels(eb_names, fontsize=9)
    ax.set_xlabel("IV Error Component (vol points)")
    ax.set_title(
        f"Error Budget Decomposition (H4: Structural / Surrogate = {eb['structural_over_surrogate_ratio']:.1f}×)"
    )
    for bar, v in zip(bars, eb_vals, strict=True):
        ax.text(
            v + max(eb_vals) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f} vp",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, max(eb_vals) * 1.25)
    fig.tight_layout()
    p3 = fig_dir / "fig3_error_budget.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    generated.append(p3)

    # ------------------------------------------------------------------
    # Fig 4: Daily Test-Period RMSE & Intertemporal Parameter Stability
    # ------------------------------------------------------------------
    recs_e0 = bench["daily_records"]["E0_cos_cold"]
    recs_e2 = bench["daily_records"]["E2_sobolev_64x4"]
    recs_e5 = bench["daily_records"]["E5_hybrid_5cos"]
    days = np.arange(len(recs_e0))
    stress_mask = np.array([r["regime"] == "stress" for r in recs_e0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)
    ax1.plot(
        days, [r["rmse_vp"] for r in recs_e0], marker="o", label="E0 (COS+LM cold)", color="#2563eb"
    )
    ax1.plot(
        days,
        [r["rmse_vp"] for r in recs_e2],
        marker="s",
        linestyle="--",
        label="E2 (Sobolev 64×4)",
        color="#0f766e",
    )
    ax1.plot(
        days,
        [r["rmse_vp"] for r in recs_e5],
        marker="^",
        linestyle=":",
        label="E5 (Hybrid 5 COS)",
        color="#7c3aed",
    )
    if stress_mask.any():
        idx_s = np.flatnonzero(stress_mask)
        ax1.axvspan(
            idx_s[0] - 0.5, idx_s[-1] + 0.5, color="#fecaca", alpha=0.35, label="Stress regime"
        )
    ax1.set_ylabel("Daily Exact COS RMSE (vol pts)")
    ax1.set_title("Chronological Held-Out Test Period: Daily Fit & Parameter Stability")
    ax1.legend(loc="upper left", fontsize=8.5)

    def _jumps(recs: list[dict]) -> np.ndarray:
        th = np.array([r["theta"] for r in recs], dtype=np.float64)
        return np.sum(np.abs(np.diff(th, axis=0)), axis=1)

    if len(days) > 1:
        ax2.plot(days[1:], _jumps(recs_e0), marker="o", label="E0 ||ΔΘ||₁", color="#2563eb")
        ax2.plot(
            days[1:],
            _jumps(recs_e2),
            marker="s",
            linestyle="--",
            label="E2 ||ΔΘ||₁",
            color="#0f766e",
        )
        ax2.plot(
            days[1:],
            _jumps(recs_e5),
            marker="^",
            linestyle=":",
            label="E5 ||ΔΘ||₁",
            color="#7c3aed",
        )
        if stress_mask.any():
            idx_s = np.flatnonzero(stress_mask)
            ax2.axvspan(idx_s[0] - 0.5, idx_s[-1] + 0.5, color="#fecaca", alpha=0.35)
    ax2.set_xlabel("Test Trading Day Index")
    ax2.set_ylabel("Daily Parameter Jump ||Θₜ − Θₜ₋₁||₁")
    ax2.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout()
    p4 = fig_dir / "fig4_daily_stability.png"
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    generated.append(p4)

    return generated


def write_summary_markdown(bench: dict, md_path: Path) -> None:
    hw = bench["hardware"]
    st = bench["summary_table"]
    noarb = bench["no_arbitrage_audit"]
    eb = bench["error_budget"]
    verdicts = bench["verdicts"]
    wil = bench["wilcoxon_vs_e0"]

    lines = [
        "# Benchmark Results Summary (Phase 7 — Gate G7)",
        "",
        "## Hardware, Compiler & Data Stamp",
        f"- **CPU / OS:** `{hw['cpu']}` ({hw['arch']}, {hw['os']}, single-threaded `OMP_NUM_THREADS={hw['omp_num_threads']}`)",
        f"- **Compiler Flags:** `{bench['compiler_flags']}` · Eigen `{hw['eigen']}` · Python `{hw['python']}`",
        f"- **Git Commit:** `{hw['git_commit']}` · Generated UTC: `{hw['timestamp_utc']}`",
        f"- **Market Series Source:** **`{bench['data_source']}`** "
        f"(train={bench['split']['n_train']}, val={bench['split']['n_val']}, test={bench['split']['n_test']} days, "
        f"mean {bench['split']['mean_quotes_per_test_day']:.1f} OTM quotes/day)",
        "",
        "## 1. Main Benchmark Table (Section 7.5 — All Final RMSEs Audited with Exact COS)",
        "",
        "| Engine | RMSE IV (vol pts, mean) | 95% CI | p50 (ms) | p99 (ms) | Conv. (no bounds) | Iter. LM (med) | Fallback | No-Arb (bfly/cal %) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]

    display_order = [
        ("Ref_naive", "Ref (QuantLib+SciPy)"),
        ("E0_cos_cold", "E0 COS+LM cold"),
        ("E1_cos_warm", "E1 COS+LM warm"),
        ("E2_sobolev_64x4", "E2 Surrogate 64×4 (Sobolev)"),
        ("E2_sobolev_64x4_safe", "E2 Surrogate 64×4 (+Circuit-Breaker)"),
        ("E2_baseline_64x4", "E2 Surrogate 64×4 (Baseline)"),
        ("E2_sobolev_128x4", "E2 Surrogate 128×4 (Sobolev)"),
        ("E3_pointwise_64x4", "E3 Pointwise 64×4 (Random Grid)"),
        ("E4_inverse_ens5", "E4 Inverse 5-Seed Ensemble only"),
        ("E5_hybrid_5cos", "E5 Hybrid (Inverse + 5 COS iters)"),
    ]

    for key, label in display_order:
        if key not in st:
            continue
        r = st[key]
        ci = r["rmse_iv_vp_ci95"]
        if key == "E2_sobolev_64x4":
            na_str = (
                f"raw {noarb['raw_surrogate_64x4']['butterfly_violation_pct']:.2f}% / "
                f"{noarb['raw_surrogate_64x4']['calendar_violation_pct']:.2f}% · "
                f"COS {noarb['cos_revalued_e2']['butterfly_violation_pct']:.2f}% / "
                f"{noarb['cos_revalued_e2']['calendar_violation_pct']:.2f}%"
            )
        elif key == "E0_cos_cold":
            na_str = (
                f"COS {noarb['cos_revalued_e0']['butterfly_violation_pct']:.2f}% / "
                f"{noarb['cos_revalued_e0']['calendar_violation_pct']:.2f}%"
            )
        else:
            na_str = "COS-audited"

        lines.append(
            f"| {label} | `{_fmt(r['rmse_iv_vp_mean'], '.4f')}` | `[{_fmt(ci[0], '.3f')}, {_fmt(ci[1], '.3f')}]` | "
            f"`{_fmt(r['latency_p50_ms'], '.3f')}` | `{_fmt(r['latency_p99_ms'], '.3f')}` | "
            f"`{r['converged_no_bounds_rate'] * 100:.1f}%` | `{r['iters_median']:.1f}` | "
            f"`{r['fallback_rate'] * 100:.1f}%` | {na_str} |"
        )

    lines.extend(
        [
            "",
            "## 2. Accuracy Sliced by Maturity, Moneyness & Regime (Mean Exact COS RMSE, vol pts)",
            "",
            "| Engine | < 1M | 1M–6M | > 6M | Put Wing | ATM | Call Wing | Calm Days | Stress Days | ||ΔΘ||₁ (med) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in display_order:
        if key not in st:
            continue
        r = st[key]
        b = r["buckets_rmse_vp"]
        lines.append(
            f"| {label} | `{_fmt(b['tau_lt_1m'])}` | `{_fmt(b['tau_1m_6m'])}` | `{_fmt(b['tau_gt_6m'])}` | "
            f"`{_fmt(b['put_wing'])}` | `{_fmt(b['atm'])}` | `{_fmt(b['call_wing'])}` | "
            f"`{_fmt(r['calm_rmse_vp_mean'])}` | `{_fmt(r['stress_rmse_vp_mean'])}` | `{_fmt(r['theta_l1_jump_median'])}` |"
        )

    lines.extend(
        [
            "",
            "## 3. Error Budget Decomposition (Section 7.3 — Hypothesis H4)",
            "",
            "| Component | Measured RMSE (vol pts) | Notes |",
            "|---|---:|---|",
            f"| (a) Surrogate approximation (Sobolev 64×4, in-box days) | `{eb['a_surrogate_rmse_vp']:.4f}` | Against exact COS on 180-node grid ({eb.get('a_surrogate_in_box_days', '?')} in-box days; all days incl. OOB extrapolation: `{eb.get('a_surrogate_all_including_oob_vp', eb['a_surrogate_rmse_vp']):.4f}` vp) |",
            f"| (b) Linear interpolation operator $M$ | `{eb['b_interp_M_rmse_vp']:.4f}` | Cubic spline in $k$ + linear in total variance |",
            f"| (c) Optimizer / multi-start dispersion (10 starts) | `{eb['c_optimizer_multistart_std_vp']:.4f}` | Std dev of RMSE across 10 dispersed cold starts |",
            f"| (d) Structural Heston vs. Market (`{bench['data_source']}`) | **`{eb['d_structural_heston_vs_market_rmse_vp']:.4f}`** | Dominant component (**{eb['structural_over_surrogate_ratio']:.1f}×** surrogate error) |",
            f"| (e) Quote bid-ask / microstructure noise (est.) | `{eb['e_bidask_microstructure_est_vp']:.4f}` | Estimated via high-frequency 2nd-difference MAD |",
            "",
            "## 4. Paired Daily Wilcoxon Signed-Rank Tests vs. E0",
            "",
            "| Comparison (Engine − E0) | Days ($n$) | Median ΔRMSE (vol pts) | Wilcoxon $p$-value |",
            "|---|---:|---:|---:|",
        ]
    )
    for eng, wres in wil.items():
        lines.append(
            f"| `{eng}` − `E0` | {wres['n']} | `{_fmt(wres['median_diff_vp'], '+.4f')}` | `{_fmt(wres['p_value'], '.3e')}` |"
        )

    lines.extend(
        [
            "",
            "## 5. Falsifiable Hypotheses Verdicts (H1–H5)",
            "",
            "| ID | Hypothesis | Measured Evidence | Verdict |",
            "|---|---|---|---|",
        ]
    )
    for hid in ("H1", "H2", "H3", "H4", "H5"):
        v = verdicts[hid]
        lines.append(f"| **{hid}** | {v['statement']} | {v['measured']} | **`{v['verdict']}`** |")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase 7 markdown tables and matplotlib figures"
    )
    parser.add_argument("--bench", type=Path, default=Path("results/benchmark_full.json"))
    parser.add_argument("--phase4", type=Path, default=Path("results/phase4_gate_g4.json"))
    parser.add_argument("--md-out", type=Path, default=Path("results/summary_tables.md"))
    parser.add_argument("--fig-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    bench = json.loads(args.bench.read_text())
    phase4 = json.loads(args.phase4.read_text()) if args.phase4.exists() else None
    write_summary_markdown(bench, args.md_out)
    figs = generate_figures(bench, phase4, args.fig_dir)
    print(f"Wrote {args.md_out} and {len(figs)} figures in {args.fig_dir}/")


if __name__ == "__main__":
    main()
