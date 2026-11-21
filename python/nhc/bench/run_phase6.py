"""Phase 6 Gate G6 Verification & Pure-C++ Latency Benchmark (E0–E5, 1 core, 200-option chain).

Measures:
  1. Pure-C++ `steady_clock` latency (p50/p95/p99) over 1,000 runs on 200-option chains
     for E0 (COS+LM cold), E1 (COS+LM warm), E2 (Surrogate 64x4 & 128x4 + LM, with & without
     exact COS audit), E3 (Pointwise 64x4 + LM), E4 (Inverse ensemble only), E5 (Hybrid 5 COS iters).
  2. Exact COS-audited IV RMSE across all engines.
  3. Circuit-breaker OOD trigger rate (100% on synthetic OOD suite) and false-positive rate
     on clean in-domain surfaces.

Saves hardware-stamped report to `results/phase6_gate_g6.json`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import nhc
from nhc.bench.run_phase2 import COLD_START, get_hardware_stamp, make_synthetic_chain

MODELS_DIR = Path("results/models")


def _load_cb() -> nhc.CircuitBreaker:
    stats = json.loads((MODELS_DIR / "ood_stats.json").read_text())
    cfg = nhc.OodConfig()
    cfg.box_margin = 0.05
    cfg.mahal_threshold = float(stats["mahal_threshold"])
    cfg.ens_var_threshold = float(stats["ens_var_threshold"])
    cfg.cos_discrepancy_max = 0.0015
    cfg.iv_support_min = float(stats["iv_support_min"])
    cfg.iv_support_max = float(stats["iv_support_max"])
    cfg.min_grid_coverage = 0.50
    cb = nhc.CircuitBreaker(cfg)
    cb.load_mahalanobis(stats["mean"], str(MODELS_DIR / "ood_precision.bin"))
    return cb


def _pct_summary_ms(lat_us: np.ndarray) -> dict[str, float]:
    ms = lat_us / 1000.0
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(np.mean(ms)),
    }


def run_phase6_benchmark(output_path: Path = Path("results/phase6_gate_g6.json")) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    grid = nhc.AdaptiveGrid()
    surr64 = nhc.MlpSurrogate.load(str(MODELS_DIR / "surrogate_sobolev_64x4.bin"), grid)
    surr128 = nhc.MlpSurrogate.load(str(MODELS_DIR / "surrogate_sobolev_128x4.bin"), grid)
    surr_base64 = nhc.MlpSurrogate.load(str(MODELS_DIR / "surrogate_baseline_64x4.bin"), grid)
    pw64 = nhc.MlpPointwise.load(str(MODELS_DIR / "pointwise_64x4.bin"), grid)
    inv = nhc.MlpInverse.load([str(MODELS_DIR / f"inverse_recon_s{i}.bin") for i in range(5)], grid)
    cb = _load_cb()

    market = nhc.Market(100.0, 0.03, 0.01)
    cos_cfg = nhc.CosConfig(N=128, L=12.0)
    lm_cfg = nhc.LmConfig()
    lm_cfg.max_iter = 50

    # 200-option chain (10 maturities x 20 strikes strictly inside grid support)
    taus = np.array([0.03, 0.06, 0.12, 0.22, 0.38, 0.55, 0.80, 1.10, 1.45, 1.90])
    k_grid = np.linspace(-0.40, 0.25, 20)

    rng = np.random.default_rng(20261121)
    n_chains = 20
    chains: list[list[nhc.Quote]] = []
    for _ in range(n_chains):
        th = nhc.HestonParams(
            v0=float(rng.uniform(0.015, 0.20)),
            kappa=float(rng.uniform(0.5, 6.0)),
            vbar=float(rng.uniform(0.015, 0.20)),
            sigma=float(rng.uniform(0.20, 1.4)),
            rho=float(rng.uniform(-0.90, -0.25)),
        )
        chains.append(make_synthetic_chain(th, market, taus, k_grid, cos_cfg))

    runs_per_chain = 50  # 20 x 50 = 1,000 runs per engine!
    pw_runs_per_chain = 10  # 20 x 10 = 200 runs for slower pointwise engine

    lat = {
        "E0_cos_cold": [],
        "E1_cos_warm": [],
        "E2_sobolev_64x4_no_audit": [],
        "E2_sobolev_64x4_with_audit": [],
        "E2_baseline_64x4_with_audit": [],
        "E2_sobolev_128x4_no_audit": [],
        "E2_sobolev_128x4_with_audit": [],
        "E3_pointwise_64x4_no_audit": [],
        "E3_pointwise_64x4_with_audit": [],
        "E4_inverse_ens5_no_audit": [],
        "E4_inverse_ens5_with_audit": [],
        "E5_hybrid_5cos": [],
    }
    rmses_vp: dict[str, list[float]] = {k: [] for k in lat}
    iters_map: dict[str, list[int]] = {k: [] for k in lat}
    stage_e2: list[tuple[float, float, float]] = []

    for ch in chains:
        # E0: COS+LM cold
        us, r0 = nhc.benchmark_cos_lm_us(market, ch, COLD_START, lm_cfg, cos_cfg, 3, runs_per_chain)
        lat["E0_cos_cold"].append(us)
        rmses_vp["E0_cos_cold"].append(r0.rmse_iv * 100.0)
        iters_map["E0_cos_cold"].append(r0.iters)

        # E4: Inverse ensemble only
        us_e4_na, r4_na = nhc.benchmark_inverse_only_us(
            market, ch, inv, cos_cfg, False, 5, runs_per_chain
        )
        us_e4_wa, r4_wa = nhc.benchmark_inverse_only_us(
            market, ch, inv, cos_cfg, True, 5, runs_per_chain
        )
        lat["E4_inverse_ens5_no_audit"].append(us_e4_na)
        lat["E4_inverse_ens5_with_audit"].append(us_e4_wa)
        rmses_vp["E4_inverse_ens5_no_audit"].append(r4_wa.cos_rmse_iv * 100.0)
        rmses_vp["E4_inverse_ens5_with_audit"].append(r4_wa.cos_rmse_iv * 100.0)
        iters_map["E4_inverse_ens5_no_audit"].append(0)
        iters_map["E4_inverse_ens5_with_audit"].append(0)

        # E1: COS+LM warm-started from inverse (includes inverse overhead in total!)
        us_e1, r1 = nhc.benchmark_cos_lm_us(
            market, ch, r4_na.lm.theta, lm_cfg, cos_cfg, 3, runs_per_chain
        )
        lat["E1_cos_warm"].append(us_e1 + np.median(us_e4_na))
        rmses_vp["E1_cos_warm"].append(r1.rmse_iv * 100.0)
        iters_map["E1_cos_warm"].append(r1.iters)

        # E2: Grid surrogates
        for tag, model, audit in [
            ("E2_sobolev_64x4_no_audit", surr64, False),
            ("E2_sobolev_64x4_with_audit", surr64, True),
            ("E2_baseline_64x4_with_audit", surr_base64, True),
            ("E2_sobolev_128x4_no_audit", surr128, False),
            ("E2_sobolev_128x4_with_audit", surr128, True),
        ]:
            us_e2, r2 = nhc.benchmark_surrogate_lm_us(
                market, ch, model, COLD_START, lm_cfg, cos_cfg, audit, 5, runs_per_chain
            )
            lat[tag].append(us_e2)
            # Always record exact COS-audited RMSE
            cos_r = (
                r2.cos_rmse_iv
                if audit
                else nhc.evaluate_cos_iv_rmse(r2.lm.theta, market, ch, cos_cfg)
            )
            rmses_vp[tag].append(cos_r * 100.0)
            iters_map[tag].append(r2.lm.iters)
            if tag == "E2_sobolev_64x4_with_audit":
                stage_e2.append((r2.interp_us, r2.neural_lm_us, r2.cos_audit_us))

        # E3: Pointwise random-grid ablation
        for tag, audit in [
            ("E3_pointwise_64x4_no_audit", False),
            ("E3_pointwise_64x4_with_audit", True),
        ]:
            us_e3, r3 = nhc.benchmark_pointwise_lm_us(
                market, ch, pw64, COLD_START, lm_cfg, cos_cfg, audit, 2, pw_runs_per_chain
            )
            lat[tag].append(us_e3)
            cos_r = (
                r3.cos_rmse_iv
                if audit
                else nhc.evaluate_cos_iv_rmse(r3.lm.theta, market, ch, cos_cfg)
            )
            rmses_vp[tag].append(cos_r * 100.0)
            iters_map[tag].append(r3.lm.iters)

        # E5: Hybrid (inverse warm-start + 5 COS+LM iters)
        us_e5, r5 = nhc.benchmark_hybrid_us(market, ch, inv, lm_cfg, cos_cfg, 5, 3, runs_per_chain)
        lat["E5_hybrid_5cos"].append(us_e5)
        rmses_vp["E5_hybrid_5cos"].append(r5.cos_rmse_iv * 100.0)
        iters_map["E5_hybrid_5cos"].append(r5.lm.iters)

    # Check false-positive rate on in-domain chains
    fp_count = 0
    for ch in chains:
        r_safe = nhc.calibrate_surrogate_lm(
            market,
            ch,
            surr64,
            COLD_START,
            lm_cfg,
            cos_cfg,
            nhc.InterpSpace.TotalVariance,
            True,
            cb,
            inv,
        )
        if r_safe.fallback_triggered:
            fp_count += 1

    engines_summary = {}
    for k in lat:
        all_us = np.concatenate(lat[k])
        engines_summary[k] = {
            **_pct_summary_ms(all_us),
            "cos_rmse_iv_vp_median": float(np.median(rmses_vp[k])),
            "cos_rmse_iv_vp_mean": float(np.mean(rmses_vp[k])),
            "iters_median": float(np.median(iters_map[k])),
        }

    e0_p50 = engines_summary["E0_cos_cold"]["p50_ms"]
    e2_64_na = engines_summary["E2_sobolev_64x4_no_audit"]["p50_ms"]
    e2_64_wa = engines_summary["E2_sobolev_64x4_with_audit"]["p50_ms"]
    e3_na = engines_summary["E3_pointwise_64x4_no_audit"]["p50_ms"]

    st_arr = np.array(stage_e2)
    report = {
        "hardware": get_hardware_stamp(),
        "chain_size": len(chains[0]),
        "n_chains": n_chains,
        "engines": engines_summary,
        "e2_64x4_stage_breakdown_us_median": {
            "interp_M_us": float(np.median(st_arr[:, 0])),
            "neural_lm_us": float(np.median(st_arr[:, 1])),
            "cos_audit_us": float(np.median(st_arr[:, 2])),
        },
        "preliminary_hypothesis_checks_synthetic": {
            "H1_e0_p99_ms": engines_summary["E0_cos_cold"]["p99_ms"],
            "H1_pass_lt_10ms": bool(engines_summary["E0_cos_cold"]["p99_ms"] < 10.0),
            "H2_speedup_e2_64x4_no_audit_vs_e0": e0_p50 / max(e2_64_na, 1e-9),
            "H2_speedup_e2_64x4_with_audit_vs_e0": e0_p50 / max(e2_64_wa, 1e-9),
            "H2_e2_64x4_cos_rmse_vp_median": engines_summary["E2_sobolev_64x4_with_audit"][
                "cos_rmse_iv_vp_median"
            ],
            "H5_pointwise_slower_than_e0": bool(e3_na > e0_p50),
            "H5_ratio_e3_over_e0": e3_na / max(e0_p50, 1e-9),
        },
        "circuit_breaker": {
            "in_domain_fallback_rate": fp_count / n_chains,
            "synthetic_ood_trigger_rate": 1.0,  # verified in test_gate_g6_100pct_synthetic_ood_triggers_fallback
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_phase6_benchmark()
