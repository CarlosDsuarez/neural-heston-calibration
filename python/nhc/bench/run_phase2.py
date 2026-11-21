"""Phase 2 Benchmark Harness v0 & Gate G2 Verification.

Measures:
  1. Gate G2: 100 noiseless synthetic surfaces calibrated from a fixed cold start;
     reports recovery fraction (RMSE IV <= 0.005 vol points == 5e-5 decimal IV).
  2. Hypothesis H1: pure C++ `steady_clock` latency (p50/p95/p99) over 1,000 runs
     on a 200-option chain (10 maturities x 20 strikes, N=128, L=12, 1 core).
  3. Ref (naive QuantLib + SciPy least_squares) latency & accuracy comparison.

Saves hardware-stamped artifact to `results/phase2_gate_g2.json`.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import nhc
from nhc.bench.naive_baseline import calibrate_naive_scipy_lsq

COLD_START = nhc.HestonParams(v0=0.04, kappa=1.5, vbar=0.04, sigma=0.5, rho=-0.65)


def make_synthetic_chain(
    theta: nhc.HestonParams,
    market: nhc.Market,
    taus: np.ndarray,
    k_grid: np.ndarray,
    cos_cfg: nhc.CosConfig,
) -> list[nhc.Quote]:
    """Builds a noiseless Heston option chain grouped by ascending tau (OTM quotes)."""
    chain: list[nhc.Quote] = []
    for tau in taus:
        F = market.S0 * math.exp((market.r - market.q) * float(tau))
        sqrt_tau = math.sqrt(float(tau))
        for k_scaled in k_grid:
            K = F * math.exp(float(k_scaled) * sqrt_tau)
            chain.append(
                nhc.Quote(
                    tau=float(tau),
                    strike=float(K),
                    mid=0.0,
                    iv=0.0,
                    vega=1.0,
                    weight=1.0,
                    is_call=bool(k_scaled >= 0.0),
                )
            )
    prices = nhc.price_chain(theta, market, chain, cos_cfg)
    populated: list[nhc.Quote] = []
    for q, px in zip(chain, prices, strict=True):
        iv = nhc.implied_volatility(
            float(px), market.S0, q.strike, market.r, market.q, q.tau, q.is_call
        )
        if not math.isfinite(iv) or iv <= 1e-4:
            continue
        vega = nhc.black_scholes_vega(market.S0, q.strike, market.r, market.q, q.tau, iv)
        populated.append(
            nhc.Quote(
                tau=q.tau,
                strike=q.strike,
                mid=float(px),
                iv=float(iv),
                vega=float(vega),
                weight=1.0,
                is_call=q.is_call,
            )
        )
    return populated


def get_hardware_stamp() -> dict[str, str]:
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        elif Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    git_hash = "unknown"
    with contextlib.suppress(Exception):
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_hash,
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu": cpu,
        "cpu_count": str(os.cpu_count()),
        "python": platform.python_version(),
        "eigen": nhc.eigen_version(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }


def run_phase2_benchmark(
    n_surfaces: int = 100,
    latency_runs: int = 1000,
    naive_runs: int = 5,
    output_path: Path | None = None,
) -> dict:
    # Enforce single-threaded execution for fair 1-core H1 measurement
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    rng = np.random.default_rng(20261121)
    market = nhc.Market(100.0, 0.03, 0.01)
    cos_cfg = nhc.CosConfig(N=128, L=12.0, use_c4=False)
    lm_cfg = nhc.LmConfig()
    lm_cfg.max_iter = 60
    lm_cfg.residual_mode = nhc.ResidualMode.VegaWeightedPrice

    # 200-option chain: 10 maturities x 20 log-moneyness strikes (exact day-multiples for QL parity)
    day_grid = np.array([15, 30, 60, 91, 135, 182, 273, 365, 547, 730], dtype=np.int64)
    taus = day_grid / 365.0
    k_grid = np.linspace(-0.45, 0.30, 20)

    rmses_vol_pts: list[float] = []
    iters_list: list[int] = []
    converged_list: list[bool] = []
    hit_bounds_list: list[bool] = []
    all_chains: list[tuple[nhc.HestonParams, list[nhc.Quote]]] = []

    for _i in range(n_surfaces):
        truth = nhc.HestonParams(
            v0=float(rng.uniform(0.01, 0.25)),
            kappa=float(rng.uniform(0.3, 8.0)),
            vbar=float(rng.uniform(0.01, 0.25)),
            sigma=float(rng.uniform(0.15, 2.0)),
            rho=float(rng.uniform(-0.95, -0.15)),
        )
        chain = make_synthetic_chain(truth, market, taus, k_grid, cos_cfg)
        res = nhc.calibrate_cos_lm(market, chain, COLD_START, lm_cfg, cos_cfg)
        rmses_vol_pts.append(res.rmse_iv * 100.0)  # convert decimal IV to vol points
        iters_list.append(res.iters)
        converged_list.append(res.converged)
        hit_bounds_list.append(res.hit_bounds)
        all_chains.append((truth, chain))

    rmses_arr = np.array(rmses_vol_pts, dtype=np.float64)
    iters_arr = np.array(iters_list, dtype=np.int64)
    recovered_005 = float(np.mean(rmses_arr <= 0.005))

    # H1 Latency benchmark: 1,000 runs across all 100 synthetic 200-option chains (10 runs each)
    # using pure C++ steady_clock inside nhc_core after warmup.
    all_lat_us: list[np.ndarray] = []
    runs_per_chain = max(1, latency_runs // len(all_chains))
    for _, chain in all_chains:
        lat_us, _ = nhc.benchmark_cos_lm_us(
            market,
            chain,
            COLD_START,
            lm_cfg,
            cos_cfg,
            warmup=3,
            runs=runs_per_chain,
            transform=nhc.ParamTransform.LogArtanh,
        )
        all_lat_us.append(lat_us)
    lat_ms = np.concatenate(all_lat_us) / 1000.0

    # Ref naive baseline (QuantLib Python + SciPy least_squares) on first `naive_runs` chains
    naive_ms: list[float] = []
    naive_rmses_vp: list[float] = []
    for _, chain in all_chains[:naive_runs]:
        nres = calibrate_naive_scipy_lsq(market, chain, COLD_START, cos_cfg_eval=cos_cfg)
        naive_ms.append(nres.elapsed_ms)
        naive_rmses_vp.append(nres.rmse_iv * 100.0)

    summary = {
        "hardware": get_hardware_stamp(),
        "config": {
            "n_surfaces": n_surfaces,
            "chain_size_target": int(len(taus) * len(k_grid)),
            "cos_N": cos_cfg.N,
            "cos_L": cos_cfg.L,
            "lm_max_iter": lm_cfg.max_iter,
            "residual_mode": "VegaWeightedPrice",
            "cold_start": {
                "v0": COLD_START.v0,
                "kappa": COLD_START.kappa,
                "vbar": COLD_START.vbar,
                "sigma": COLD_START.sigma,
                "rho": COLD_START.rho,
            },
        },
        "gate_g2_recovery": {
            "n_surfaces": n_surfaces,
            "recovery_rate_le_0p005_vp": recovered_005,
            "converged_rate": float(np.mean(converged_list)),
            "hit_bounds_rate": float(np.mean(hit_bounds_list)),
            "rmse_iv_vol_pts_median": float(np.median(rmses_arr)),
            "rmse_iv_vol_pts_p95": float(np.percentile(rmses_arr, 95)),
            "rmse_iv_vol_pts_p99": float(np.percentile(rmses_arr, 99)),
            "rmse_iv_vol_pts_max": float(np.max(rmses_arr)),
            "iters_median": float(np.median(iters_arr)),
            "iters_p95": float(np.percentile(iters_arr, 95)),
            "gate_g2_pass": bool(recovered_005 >= 0.99),
        },
        "h1_e0_latency_ms": {
            "total_runs": int(len(lat_ms)),
            "p50_ms": float(np.percentile(lat_ms, 50)),
            "p95_ms": float(np.percentile(lat_ms, 95)),
            "p99_ms": float(np.percentile(lat_ms, 99)),
            "mean_ms": float(np.mean(lat_ms)),
            "h1_pass_p99_lt_10ms": bool(np.percentile(lat_ms, 99) < 10.0),
        },
        "ref_naive_quantlib_scipy": {
            "n_runs": len(naive_ms),
            "p50_ms": float(np.median(naive_ms)) if naive_ms else None,
            "mean_ms": float(np.mean(naive_ms)) if naive_ms else None,
            "rmse_iv_vol_pts_median": float(np.median(naive_rmses_vp)) if naive_rmses_vp else None,
            "speedup_e0_over_ref": (
                float(np.median(naive_ms) / np.percentile(lat_ms, 50)) if naive_ms else None
            ),
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")

    return summary


if __name__ == "__main__":
    out = Path("results/phase2_gate_g2.json")
    report = run_phase2_benchmark(output_path=out)
    print(json.dumps(report, indent=2))
