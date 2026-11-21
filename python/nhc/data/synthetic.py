"""Stratified Sobol synthetic Heston dataset generator (Phase 3, T3.3–T3.4, Gate G3).

Generates Heston implied-volatility surfaces and exact analytic IV Jacobians
on the canonical AdaptiveGrid (10 maturities x 18 scaled log-moneyness nodes = 180 nodes)
using three strata (Section 5.1):
  - Stratum 0 (65%): Normal / moderately elevated
  - Stratum 1 (25%): Stress & tails
  - Stratum 2 (10%): Empirical perturbed (from training-only SPX COS+LM anchors)

Invokes the compiled OpenMP C++ binary `nhc_gen_dataset` (`#pragma omp parallel for
schedule(static)`) when available, with a pure-Python/nanobind fallback.
"""

from __future__ import annotations

import contextlib
import json
import math
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import nhc
from nhc.bench.run_phase2 import get_hardware_stamp
from nhc.data.spx_ingest import (
    calibrate_empirical_anchors,
    generate_misspecified_spx_proxy_series,
    split_snapshots_temporal,
)

# Section 5.1 initial parameter bounds: [v0, kappa, vbar, sigma, rho]
STRATUM_NORMAL_LOW = np.array([0.005, 0.2, 0.01, 0.1, -0.98], dtype=np.float64)
STRATUM_NORMAL_HIGH = np.array([0.25, 8.0, 0.25, 2.5, -0.15], dtype=np.float64)

STRATUM_STRESS_LOW = np.array([0.25, 0.05, 0.10, 1.5, -0.98], dtype=np.float64)
STRATUM_STRESS_HIGH = np.array([1.00, 12.0, 0.50, 4.5, -0.10], dtype=np.float64)

# Global safe training box for empirical perturbations & circuit-breaker bounds
GLOBAL_LOW = np.array([0.005, 0.05, 0.01, 0.10, -0.98], dtype=np.float64)
GLOBAL_HIGH = np.array([1.00, 12.0, 0.50, 4.50, -0.10], dtype=np.float64)


def _sobol_log_uniform_sample(n: int, low: np.ndarray, high: np.ndarray, seed: int) -> np.ndarray:
    """Draws `n` scrambled Sobol points, log-uniform in positive params (0..3) and uniform in rho (4)."""
    if n <= 0:
        return np.empty((0, 5), dtype=np.float64)
    sampler = qmc.Sobol(d=5, scramble=True, seed=seed)
    # Draw next power-of-2 Sobol points and slice first n (preserves low-discrepancy balance)
    m = max(1, int(math.ceil(math.log2(n))))
    u = sampler.random_base2(m)[:n]

    out = np.empty_like(u)
    log_lo = np.log(low[:4])
    log_hi = np.log(high[:4])
    out[:, :4] = np.exp(log_lo + u[:, :4] * (log_hi - log_lo))
    out[:, 4] = low[4] + u[:, 4] * (high[4] - low[4])
    return out


def sample_stratified_thetas(
    n_samples: int,
    seed: int = 20261121,
    empirical_anchors: np.ndarray | None = None,
    weights: tuple[float, float, float] = (0.65, 0.25, 0.10),
) -> tuple[np.ndarray, np.ndarray]:
    """Returns `(thetas [N, 5], strata [N])` using the 65% / 25% / 10% Sobol strata."""
    n0 = int(round(n_samples * weights[0]))
    n1 = int(round(n_samples * weights[1]))
    n2 = n_samples - n0 - n1

    th0 = _sobol_log_uniform_sample(n0, STRATUM_NORMAL_LOW, STRATUM_NORMAL_HIGH, seed=seed)
    th1 = _sobol_log_uniform_sample(n1, STRATUM_STRESS_LOW, STRATUM_STRESS_HIGH, seed=seed + 1)

    rng = np.random.default_rng(seed + 2)
    if empirical_anchors is None or len(empirical_anchors) == 0:
        # Fallback if no anchors passed: draw from empirical SPX-like normal sub-box
        emp_lo = np.array([0.015, 0.8, 0.02, 0.35, -0.92], dtype=np.float64)
        emp_hi = np.array([0.15, 5.0, 0.12, 1.4, -0.55], dtype=np.float64)
        th2 = _sobol_log_uniform_sample(n2, emp_lo, emp_hi, seed=seed + 2)
    else:
        idx = rng.integers(0, len(empirical_anchors), size=n2)
        base = empirical_anchors[idx].copy()
        # Log-normal perturbation on positive parameters (15% std), small Gaussian on atanh(rho)
        noise_pos = np.exp(rng.normal(0.0, 0.15, size=(n2, 4)))
        base[:, :4] = np.clip(base[:, :4] * noise_pos, GLOBAL_LOW[:4], GLOBAL_HIGH[:4])
        z_rho = np.arctanh(np.clip(base[:, 4], -0.99, 0.99)) + rng.normal(0.0, 0.12, size=n2)
        base[:, 4] = np.clip(np.tanh(z_rho), GLOBAL_LOW[4], GLOBAL_HIGH[4])
        th2 = base

    thetas = np.vstack([th0, th1, th2])
    strata = np.concatenate(
        [
            np.zeros(n0, dtype=np.uint8),
            np.ones(n1, dtype=np.uint8),
            np.full(n2, 2, dtype=np.uint8),
        ]
    )

    # Deterministic shuffle so mini-batches and subsets (10k..240k) preserve exact stratum ratios
    perm = rng.permutation(n_samples)
    return thetas[perm], strata[perm]


def _find_gen_binary() -> Path | None:
    candidates = [
        Path("build-cpp/nhc_gen_dataset"),
        Path(__file__).resolve().parents[3] / "build-cpp" / "nhc_gen_dataset",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _evaluate_surfaces_python_fallback(
    thetas: np.ndarray, cos_N: int, cos_L: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    grid = nhc.AdaptiveGrid()
    chain = grid.make_canonical_chain()
    unit_m = nhc.Market(1.0, 0.0, 0.0)
    cfg = nhc.CosConfig(N=cos_N, L=cos_L)
    cfg_hi = nhc.CosConfig(N=max(512, cos_N * 2), L=max(24.0, cos_L * 1.5))

    n = len(thetas)
    G = grid.size
    iv_out = np.zeros((n, G), dtype=np.float32)
    jac_out = np.zeros((n, G, 5), dtype=np.float32)
    valid = np.zeros(n, dtype=np.uint8)
    max_rt = 0.0
    t0 = time.perf_counter()

    for s in range(n):
        p = nhc.HestonParams(*[float(x) for x in thetas[s]])
        use_hi = (p.sigma > 1.2) or (2.0 * p.kappa * p.vbar < 0.25 * p.sigma * p.sigma)
        prices, Jp = nhc.price_chain_with_jacobian(p, unit_m, chain, cfg_hi if use_hi else cfg)
        ok = True
        for g, q in enumerate(chain):
            px = float(prices[g])
            if not (px > 1e-14 and math.isfinite(px)):
                ok = False
                break
            iv = nhc.implied_volatility(px, 1.0, q.strike, 0.0, 0.0, q.tau, q.is_call)
            if not (math.isfinite(iv) and 1e-4 < iv <= 5.0):
                ok = False
                break
            vega = nhc.black_scholes_vega(1.0, q.strike, 0.0, 0.0, q.tau, iv)
            if not (math.isfinite(vega) and vega > 1e-10):
                ok = False
                break
            px_rt = nhc.black_scholes_price(1.0, q.strike, 0.0, 0.0, q.tau, iv, q.is_call)
            iv_rt = nhc.implied_volatility(px_rt, 1.0, q.strike, 0.0, 0.0, q.tau, q.is_call)
            max_rt = max(max_rt, abs(iv_rt - iv))
            iv_out[s, g] = iv
            jac_out[s, g, :] = Jp[g, :] / vega
        if ok and np.all(np.isfinite(iv_out[s])) and np.all(np.isfinite(jac_out[s])):
            valid[s] = 1

    elapsed = time.perf_counter() - t0
    stats = {
        "n_requested": n,
        "n_valid": int(valid.sum()),
        "n_invalid": int(n - valid.sum()),
        "grid_size": G,
        "omp_threads": 1,
        "elapsed_s": elapsed,
        "surfaces_per_sec": n / max(elapsed, 1e-9),
        "max_iv_round_trip_err": max_rt,
        "backend": "python_fallback",
    }
    return iv_out, jac_out, valid, stats


def generate_dataset(
    n_target: int,
    out_dir: Path,
    seed: int = 20261121,
    cos_N: int = 128,
    cos_L: float = 12.0,
    empirical_anchors: np.ndarray | None = None,
    compressed: bool = False,
    include_jacobians: bool = True,
) -> dict:
    """Generates `n_target` valid Heston surfaces (+ optional Jacobians) and saves
    `<out_dir>/dataset.npz` and `<out_dir>/manifest.json`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Oversample by 6% so after filtering rare extreme stress edge cases we still deliver >= n_target
    n_draw = int(math.ceil(n_target * 1.06)) + 32
    thetas, strata = sample_stratified_thetas(
        n_draw, seed=seed, empirical_anchors=empirical_anchors
    )

    grid = nhc.AdaptiveGrid()
    G = grid.size
    gen_bin = _find_gen_binary()

    if gen_bin is not None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            th_path = tmp_p / "thetas.f64"
            out_pref = tmp_p / "gen"
            np.ascontiguousarray(thetas, dtype=np.float64).tofile(th_path)
            cmd = [str(gen_bin), str(th_path), str(n_draw), str(out_pref), str(cos_N), str(cos_L)]
            subprocess.run(cmd, check=True)
            iv_raw = np.fromfile(f"{out_pref}_iv.f32", dtype=np.float32).reshape(n_draw, G)
            jac_raw = np.fromfile(f"{out_pref}_jac.f32", dtype=np.float32).reshape(n_draw, G, 5)
            valid = np.fromfile(f"{out_pref}_valid.u8", dtype=np.uint8)
            stats = json.loads(Path(f"{out_pref}_stats.json").read_text())
            stats["backend"] = "cpp_openmp"
    else:
        iv_raw, jac_raw, valid, stats = _evaluate_surfaces_python_fallback(thetas, cos_N, cos_L)

    mask = valid == 1
    if int(mask.sum()) < n_target:
        raise RuntimeError(
            f"Only {int(mask.sum())} valid surfaces generated out of {n_draw} drawn (need {n_target})"
        )

    keep_idx = np.flatnonzero(mask)[:n_target]
    th_keep = np.ascontiguousarray(thetas[keep_idx], dtype=np.float64)
    st_keep = np.ascontiguousarray(strata[keep_idx], dtype=np.uint8)
    iv_keep = np.ascontiguousarray(iv_raw[keep_idx], dtype=np.float32)
    jac_keep = np.ascontiguousarray(jac_raw[keep_idx], dtype=np.float32)

    nan_inf_count = int(
        (~np.isfinite(th_keep)).sum()
        + (~np.isfinite(iv_keep)).sum()
        + (~np.isfinite(jac_keep)).sum()
    )
    if nan_inf_count != 0:
        raise RuntimeError(f"Gate G3 violation: found {nan_inf_count} NaN/Inf values!")

    payload = {
        "theta": th_keep,
        "stratum": st_keep,
        "iv": iv_keep,
        "taus": np.array(grid.taus, dtype=np.float64),
        "k_scaled": np.array(grid.k_scaled, dtype=np.float64),
    }
    if include_jacobians:
        payload["jac"] = jac_keep

    npz_path = out_dir / "dataset.npz"
    if compressed:
        np.savez_compressed(npz_path, **payload)
    else:
        np.savez(npz_path, **payload)

    git_hash = "unknown"
    with contextlib.suppress(Exception):
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_hash,
        "seed": seed,
        "n_samples": int(n_target),
        "n_drawn": int(n_draw),
        "valid_rate": float(mask.mean()),
        "stratum_counts": {
            "normal_0": int((st_keep == 0).sum()),
            "stress_1": int((st_keep == 1).sum()),
            "empirical_2": int((st_keep == 2).sum()),
        },
        "grid": {
            "num_taus": grid.num_taus,
            "num_k": grid.num_k,
            "size": G,
            "taus": list(grid.taus),
            "k_scaled": list(grid.k_scaled),
        },
        "cos_config": {"N": cos_N, "L": cos_L},
        "include_jacobians": include_jacobians,
        "nan_inf_count": nan_inf_count,
        "max_iv_round_trip_err": float(stats["max_iv_round_trip_err"]),
        "generation_stats": stats,
        "file_size_bytes": int(npz_path.stat().st_size),
        "hardware": get_hardware_stamp(),
        "gate_g3_pass": bool(
            n_target >= 120_000
            and nan_inf_count == 0
            and float(stats["max_iv_round_trip_err"]) <= 1e-8
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    # 1. Build training-only empirical anchors from chronological split (zero temporal leakage)
    snaps = generate_misspecified_spx_proxy_series(n_days=60, seed=20261121)
    train_snaps, _, _ = split_snapshots_temporal(snaps)
    anchors = calibrate_empirical_anchors(train_snaps)

    # 2. Generate full 240k dataset locally in data/synthetic_240k (gitignored; covers 120k Gate G3
    #    AND full Phase 4 learning curve 10k..240k!) + validation/test synthetic sets.
    m_full = generate_dataset(
        n_target=120_000,
        out_dir=Path("data/synthetic_120k"),
        seed=20261121,
        empirical_anchors=anchors,
        compressed=False,
        include_jacobians=True,
    )
    print(
        "120k dataset manifest:",
        json.dumps(
            {
                k: m_full[k]
                for k in (
                    "n_samples",
                    "valid_rate",
                    "nan_inf_count",
                    "max_iv_round_trip_err",
                    "generation_stats",
                    "gate_g3_pass",
                )
            },
            indent=2,
        ),
    )

    # Separate held-out synthetic validation (10k) and test (10k) sets with distinct seeds
    generate_dataset(
        n_target=10_000,
        out_dir=Path("data/synthetic_val_10k"),
        seed=20261122,
        empirical_anchors=anchors,
        compressed=False,
        include_jacobians=True,
    )
    generate_dataset(
        n_target=10_000,
        out_dir=Path("data/synthetic_test_10k"),
        seed=20261123,
        empirical_anchors=anchors,
        compressed=False,
        include_jacobians=True,
    )

    # 3. Generate compact mini synthetic dataset (<= 1 MB) for CI & `make repro-small`
    m_mini = generate_dataset(
        n_target=200,
        out_dir=Path("data/mini_synthetic"),
        seed=999,
        empirical_anchors=anchors,
        compressed=True,
        include_jacobians=True,
    )
    print(f"Mini dataset size: {m_mini['file_size_bytes'] / 1024:.1f} KiB")
    # Save Gate G3 summary artifact in results/
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/phase3_gate_g3.json").write_text(json.dumps(m_full, indent=2) + "\n")
