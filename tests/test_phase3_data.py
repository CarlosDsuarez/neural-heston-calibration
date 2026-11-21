"""Phase 3 tests: AdaptiveGrid, InterpOperator M, SPX ingestion & PCP OLS, synthetic dataset."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import nhc
from nhc.data.spx_ingest import (
    calibrate_empirical_anchors,
    generate_misspecified_spx_proxy_series,
    split_snapshots_temporal,
)
from nhc.data.synthetic import generate_dataset, sample_stratified_thetas


def test_t31_grid_and_interp_operator_invariance() -> None:
    """Verify AdaptiveGrid size, forward-moneyness IV invariance, and operator M parity."""
    grid = nhc.AdaptiveGrid()
    assert grid.num_taus == 10
    assert grid.num_k == 18
    assert grid.size == 180

    # Verify IV invariance to (S0, r, q) in forward log-moneyness coordinates
    p = nhc.HestonParams(0.04, 2.0, 0.045, 0.55, -0.72)
    cfg = nhc.CosConfig(N=128, L=12.0)
    canon = grid.make_canonical_chain()
    px_unit = nhc.price_chain(p, nhc.Market(1.0, 0.0, 0.0), canon, cfg)
    iv_unit = np.array(
        [
            nhc.implied_volatility(float(px), 1.0, q.strike, 0.0, 0.0, q.tau, q.is_call)
            for px, q in zip(px_unit, canon, strict=True)
        ]
    )

    m_spx = nhc.Market(5800.0, 0.045, 0.013)
    spx_chain = []
    for q in canon:
        F = m_spx.S0 * math.exp((m_spx.r - m_spx.q) * q.tau)
        K = F * q.strike  # q.strike == exp(k) on canonical chain
        spx_chain.append(nhc.Quote(tau=q.tau, strike=K, is_call=q.is_call))
    px_spx = nhc.price_chain(p, m_spx, spx_chain, cfg)
    iv_spx = np.array(
        [
            nhc.implied_volatility(
                float(px), m_spx.S0, q.strike, m_spx.r, m_spx.q, q.tau, q.is_call
            )
            for px, q in zip(px_spx, spx_chain, strict=True)
        ]
    )
    assert float(np.max(np.abs(iv_unit - iv_spx))) < 1e-12

    # Verify InterpOperator dense matrix matches apply() in ImpliedVol mode
    op = nhc.InterpOperator(grid, m_spx, spx_chain)
    assert op.coverage == 1.0
    M = op.dense_matrix()
    iv_via_apply = op.apply(iv_unit, nhc.InterpSpace.ImpliedVol)
    iv_via_mat = M @ iv_unit
    assert float(np.max(np.abs(iv_via_apply - iv_via_mat))) < 1e-13
    assert float(np.max(np.abs(iv_via_apply - iv_unit))) < 1e-13


def test_t32_spx_ingestion_pcp_and_temporal_split() -> None:
    """Verify put-call parity OLS forward/discount extraction, filtering audit, and date split."""
    snaps = generate_misspecified_spx_proxy_series(n_days=10, seed=42)
    assert len(snaps) == 10
    for s in snaps:
        assert s.source_type == "synthetic_bates_spx_proxy"
        # Audit accounting must be exact
        a = s.audit
        assert (
            a.discarded_tau
            + a.discarded_bid_ask
            + a.discarded_spread
            + a.discarded_liquidity
            + a.discarded_itm
            + a.discarded_iv
            + a.kept_otm
            == a.total_raw
        )
        assert a.discarded_tau > 0  # 4-day expiry filtered out
        assert a.discarded_itm > 0  # ITM options filtered out
        assert a.kept_otm >= 50
        # Put-call parity OLS should recover r ~ 0.043 and q ~ 0.013 within ~30 bps
        assert abs(s.market.r - 0.043) < 0.005
        assert abs(s.market.q - 0.013) < 0.005

    train, val, test = split_snapshots_temporal(snaps)
    assert max(s.date for s in train) < min(s.date for s in val)
    assert max(s.date for s in val) < min(s.date for s in test)

    anchors = calibrate_empirical_anchors(train[:3])
    assert anchors.shape == (3, 5)
    assert np.all(np.isfinite(anchors))


def test_t33_stratified_sobol_and_dataset_generation(tmp_path: Path) -> None:
    """Verify stratified Sobol sampling and dataset generation (0 NaN/Inf, IV RT <= 1e-8)."""
    th, st = sample_stratified_thetas(1000, seed=7)
    assert th.shape == (1000, 5)
    assert abs((st == 0).mean() - 0.65) < 0.01
    assert abs((st == 1).mean() - 0.25) < 0.01
    assert abs((st == 2).mean() - 0.10) < 0.01

    manifest = generate_dataset(
        n_target=60,
        out_dir=tmp_path / "ds_test",
        seed=123,
        compressed=True,
        include_jacobians=True,
    )
    assert manifest["n_samples"] == 60
    assert manifest["nan_inf_count"] == 0
    assert manifest["max_iv_round_trip_err"] <= 1e-8

    data = np.load(tmp_path / "ds_test" / "dataset.npz")
    assert data["theta"].shape == (60, 5)
    assert data["iv"].shape == (60, 180)
    assert data["jac"].shape == (60, 180, 5)
    assert np.all(np.isfinite(data["iv"]))
    assert np.all(np.isfinite(data["jac"]))
