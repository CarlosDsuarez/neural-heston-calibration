"""Phase 2 test suite: IV solver round-trip, classical COS+LM calibration, naive baseline, Gate G2."""

from __future__ import annotations

import math

import numpy as np

import nhc
from nhc.bench.naive_baseline import calibrate_naive_scipy_lsq
from nhc.bench.run_phase2 import COLD_START, make_synthetic_chain


def test_t21_iv_round_trip_precision() -> None:
    """Section 3.5: IV -> price -> IV round-trip error <= 1e-10."""
    rng = np.random.default_rng(101)
    max_err = 0.0
    tested = 0
    for i in range(500):
        S0 = 100.0
        r = float(rng.uniform(0.0, 0.05))
        q = float(rng.uniform(0.0, 0.03))
        tau = float(rng.uniform(0.02, 3.0))
        vol = float(rng.uniform(0.03, 2.0))
        F = S0 * math.exp((r - q) * tau)
        K = F * math.exp(float(rng.uniform(-0.5, 0.4)) * math.sqrt(tau))
        is_call = (i % 2) == 0

        px = nhc.black_scholes_price(S0, K, r, q, tau, vol, is_call)
        df_r = math.exp(-r * tau)
        df_q = math.exp(-q * tau)
        intr = max(S0 * df_q - K * df_r, 0.0) if is_call else max(K * df_r - S0 * df_q, 0.0)
        if px - intr < 1e-11 * S0:
            continue
        iv = nhc.implied_volatility(px, S0, K, r, q, tau, is_call)
        assert math.isfinite(iv)
        max_err = max(max_err, abs(iv - vol))
        tested += 1

    assert tested >= 450
    assert max_err <= 1e-10


def test_t22_both_residual_modes_and_transforms() -> None:
    """Verify VegaWeightedPrice and ImpliedVol modes across LogArtanh and SqrtDrift transforms."""
    market = nhc.Market(100.0, 0.03, 0.01)
    truth = nhc.HestonParams(0.06, 2.2, 0.045, 0.7, -0.75)
    cos_cfg = nhc.CosConfig(N=128, L=12.0)
    taus = np.array([0.05, 0.15, 0.4, 0.8, 1.5])
    k_grid = np.linspace(-0.4, 0.25, 12)
    chain = make_synthetic_chain(truth, market, taus, k_grid, cos_cfg)

    for mode in (nhc.ResidualMode.VegaWeightedPrice, nhc.ResidualMode.ImpliedVol):
        for tr in (nhc.ParamTransform.LogArtanh, nhc.ParamTransform.SqrtDrift):
            lm_cfg = nhc.LmConfig()
            lm_cfg.residual_mode = mode
            res = nhc.calibrate_cos_lm(market, chain, COLD_START, lm_cfg, cos_cfg, tr)
            assert res.converged
            assert res.rmse_iv * 100.0 <= 0.005  # <= 0.005 vol points


def test_t24_naive_baseline_recovers_surface() -> None:
    """Verify naive QuantLib + SciPy least_squares baseline runs cleanly and recovers a surface."""
    market = nhc.Market(100.0, 0.03, 0.01)
    truth = nhc.HestonParams(0.05, 2.0, 0.04, 0.5, -0.7)
    cos_cfg = nhc.CosConfig(N=128, L=12.0)
    taus = np.array([30, 91, 182, 365], dtype=np.float64) / 365.0
    k_grid = np.linspace(-0.3, 0.2, 6)
    chain = make_synthetic_chain(truth, market, taus, k_grid, cos_cfg)

    res = calibrate_naive_scipy_lsq(market, chain, COLD_START, cos_cfg_eval=cos_cfg, max_nfev=60)
    assert res.elapsed_ms > 0.0
    assert res.rmse_iv * 100.0 <= 0.05


def test_gate_g2_100_synthetic_surfaces_recovery() -> None:
    """Gate G2: >= 99% recovery with RMSE IV <= 0.005 vol points across 100 noiseless surfaces."""
    rng = np.random.default_rng(20261121)
    market = nhc.Market(100.0, 0.03, 0.01)
    cos_cfg = nhc.CosConfig(N=128, L=12.0)
    lm_cfg = nhc.LmConfig()
    lm_cfg.max_iter = 60

    taus = np.array([15, 30, 60, 91, 135, 182, 273, 365, 547, 730], dtype=np.float64) / 365.0
    k_grid = np.linspace(-0.45, 0.30, 15)

    rmses_vp = []
    for _ in range(100):
        truth = nhc.HestonParams(
            v0=float(rng.uniform(0.01, 0.25)),
            kappa=float(rng.uniform(0.3, 8.0)),
            vbar=float(rng.uniform(0.01, 0.25)),
            sigma=float(rng.uniform(0.15, 2.0)),
            rho=float(rng.uniform(-0.95, -0.15)),
        )
        chain = make_synthetic_chain(truth, market, taus, k_grid, cos_cfg)
        res = nhc.calibrate_cos_lm(market, chain, COLD_START, lm_cfg, cos_cfg)
        rmses_vp.append(res.rmse_iv * 100.0)

    recovery_rate = float(np.mean(np.array(rmses_vp) <= 0.005))
    assert recovery_rate >= 0.99, (
        f"Gate G2 failed: recovery_rate={recovery_rate:.2%}, max_rmse={max(rmses_vp):.4f} vol pts"
    )
