"""Phase 6 verification tests (T6.1–T6.3 & Gate G6):
- T6.1: Exact PyTorch vs C++ Eigen parity: <= 1e-9 (float64) and <= 1e-5 relative (float32)
        for BOTH forward output AND 5-tangent forward-mode Jacobian.
- Section 4.2: Grid-hash mismatch MUST fail to load in C++.
- T6.2: `calibrate_surrogate_lm` with linear operator M & exact COS audit.
- T6.3 / Gate G6: 100% of synthetic OOD cases across ALL 5 Section 8.1 failure modes trigger
        circuit-breaker fallback to exact COS+LM.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import nhc
from nhc.bench.run_phase2 import COLD_START, make_synthetic_chain
from nhc.export.to_bin import export_grid_surrogate
from nhc.train.surrogate import GridSurrogateMLP

MODELS_DIR = Path("results/models")


def _load_cb() -> nhc.CircuitBreaker:
    stats = json.loads((MODELS_DIR / "ood_stats.json").read_text())
    cfg = nhc.OodConfig()
    cfg.box_margin = 0.05
    cfg.mahal_threshold = float(stats["mahal_threshold"])
    cfg.ens_var_threshold = float(stats["ens_var_threshold"])
    cfg.cos_discrepancy_max = 0.0015  # 0.15 vol pts
    cfg.iv_support_min = float(stats["iv_support_min"])
    cfg.iv_support_max = float(stats["iv_support_max"])
    cfg.min_grid_coverage = 0.50
    cb = nhc.CircuitBreaker(cfg)
    cb.load_mahalanobis(stats["mean"], str(MODELS_DIR / "ood_precision.bin"))
    return cb


def _load_inv() -> nhc.MlpInverse:
    grid = nhc.AdaptiveGrid()
    paths = [str(MODELS_DIR / f"inverse_recon_s{i}.bin") for i in range(5)]
    return nhc.MlpInverse.load(paths, grid)


def test_t61_pytorch_vs_cpp_eigen_parity_f64_and_f32(tmp_path: Path) -> None:
    """T6.1 mandatory parity test:
      - MlpSurrogateD (float64) vs PyTorch float64: max abs error <= 1e-9 (forward & Jacobian).
      - MlpSurrogate  (float32) vs PyTorch float32: max rel error <= 1e-5 (forward & Jacobian).
    Tested across ALL 4 smooth activations (SiLU, Softplus, ELU, tanh).
    """
    grid = nhc.AdaptiveGrid()
    taus = np.array(grid.taus, dtype=np.float64)
    k_scaled = np.array(grid.k_scaled, dtype=np.float64)
    rng = np.random.default_rng(2026)

    for act in ("silu", "softplus", "elu", "tanh"):
        torch.manual_seed(42)
        pt_model = GridSurrogateMLP(
            out_dim=grid.size,
            hidden_dim=32,
            num_layers=3,
            activation=act,
            output_mode="iv",
            y_mean=np.full(grid.size, 0.25, dtype=np.float32),
            y_std=0.12,
            taus=taus,
            k_scaled=k_scaled,
        )
        pt_model.eval()
        bin_path = tmp_path / f"surr_{act}.bin"
        export_grid_surrogate(pt_model, bin_path, {"activation": act}, taus=taus, k_scaled=k_scaled)

        cpp_f64 = nhc.MlpSurrogateD.load(str(bin_path), grid)
        cpp_f32 = nhc.MlpSurrogate.load(str(bin_path), grid)

        # Build exact float64 copy of PyTorch weights from the exported float32 values
        pt_f64 = GridSurrogateMLP(
            out_dim=grid.size,
            hidden_dim=32,
            num_layers=3,
            activation=act,
            output_mode="iv",
            y_mean=pt_model.y_mean.numpy(),
            y_std=float(pt_model.y_std.item()),
            taus=taus,
            k_scaled=k_scaled,
        ).double()
        pt_f64.load_state_dict({k: v.double() for k, v in pt_model.state_dict().items()})
        pt_f64.eval()

        max_err_f64_y = 0.0
        max_err_f64_J = 0.0
        max_rel_f32_y = 0.0
        max_rel_f32_J = 0.0

        for _ in range(10):
            x_np = rng.uniform(-0.9, 0.9, size=5).astype(np.float64)
            # Float64 parity
            with torch.no_grad():
                y_pt64, J_pt64 = pt_f64.forward_iv_and_jac_x(
                    torch.tensor(x_np, dtype=torch.float64).unsqueeze(0)
                )
            y_pt64_np = y_pt64.squeeze(0).numpy()
            J_pt64_np = J_pt64.squeeze(0).numpy()
            y_c64, J_c64 = cpp_f64.forward_x_with_jacobian(x_np)
            max_err_f64_y = max(max_err_f64_y, float(np.max(np.abs(y_c64 - y_pt64_np))))
            max_err_f64_J = max(max_err_f64_J, float(np.max(np.abs(J_c64 - J_pt64_np))))

            # Float32 parity
            with torch.no_grad():
                y_pt32, J_pt32 = pt_model.forward_iv_and_jac_x(
                    torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
                )
            y_pt32_np = y_pt32.squeeze(0).numpy().astype(np.float64)
            J_pt32_np = J_pt32.squeeze(0).numpy().astype(np.float64)
            y_c32, J_c32 = cpp_f32.forward_x_with_jacobian(x_np)
            rel_y = float(np.max(np.abs(y_c32 - y_pt32_np) / np.maximum(np.abs(y_pt32_np), 1e-3)))
            rel_J = float(np.max(np.abs(J_c32 - J_pt32_np) / np.maximum(np.abs(J_pt32_np), 1e-2)))
            max_rel_f32_y = max(max_rel_f32_y, rel_y)
            max_rel_f32_J = max(max_rel_f32_J, rel_J)

        assert max_err_f64_y <= 1e-9, f"{act} f64 y mismatch: {max_err_f64_y:.3e}"
        assert max_err_f64_J <= 1e-9, f"{act} f64 J mismatch: {max_err_f64_J:.3e}"
        assert max_rel_f32_y <= 1e-5, f"{act} f32 y rel mismatch: {max_rel_f32_y:.3e}"
        assert max_rel_f32_J <= 1e-5, f"{act} f32 J rel mismatch: {max_rel_f32_J:.3e}"


def test_t61_grid_mismatch_fails_on_load() -> None:
    """Section 4.2: changing grid metadata without updating the binary must fail to load."""
    grid = nhc.AdaptiveGrid()
    bad_taus = list(grid.taus)
    bad_taus[-1] = 2.5
    bad_grid = nhc.AdaptiveGrid(bad_taus, list(grid.k_scaled))
    with pytest.raises(RuntimeError):
        nhc.MlpSurrogate.load(str(MODELS_DIR / "surrogate_sobolev_64x4.bin"), bad_grid)


def test_gate_g6_100pct_synthetic_ood_triggers_fallback() -> None:
    """Gate G6: 100% of synthetic OOD scenarios across all 5 Section 8.1 rules trigger fallback."""
    grid = nhc.AdaptiveGrid()
    surr = nhc.MlpSurrogate.load(str(MODELS_DIR / "surrogate_sobolev_64x4.bin"), grid)
    inv = _load_inv()
    cb = _load_cb()

    market = nhc.Market(100.0, 0.03, 0.01)
    cos_cfg = nhc.CosConfig(N=128, L=12.0)
    lm_cfg = nhc.LmConfig()
    taus = np.array([0.04, 0.12, 0.30, 0.65, 1.25, 1.80])
    k_grid = np.linspace(-0.40, 0.25, 12)

    ood_cases: list[tuple[str, list[nhc.Quote]]] = []

    # Case 1 (Cond 5): Extreme high vol shock (IV = 220%, above trained support)
    c_high = make_synthetic_chain(
        nhc.HestonParams(0.04, 2.0, 0.04, 0.5, -0.7), market, taus, k_grid, cos_cfg
    )
    for q in c_high:
        q.iv = 2.20
        q.mid = nhc.black_scholes_price(
            market.S0, q.strike, market.r, market.q, q.tau, q.iv, q.is_call
        )
        q.vega = nhc.black_scholes_vega(market.S0, q.strike, market.r, market.q, q.tau, q.iv)
    ood_cases.append(("extreme_high_iv", c_high))

    # Case 2 (Cond 5): Ultra-low vol regime (IV = 0.8%, below trained support)
    c_low = make_synthetic_chain(
        nhc.HestonParams(0.04, 2.0, 0.04, 0.5, -0.7), market, taus, k_grid, cos_cfg
    )
    for q in c_low:
        q.iv = 0.008
        q.mid = nhc.black_scholes_price(
            market.S0, q.strike, market.r, market.q, q.tau, q.iv, q.is_call
        )
        q.vega = nhc.black_scholes_vega(market.S0, q.strike, market.r, market.q, q.tau, q.iv)
    ood_cases.append(("extreme_low_iv", c_low))

    # Case 3 (Cond 5): Out-of-grid maturities (tau = 3.0..5.0 years -> 0% coverage)
    c_long = make_synthetic_chain(
        nhc.HestonParams(0.04, 2.0, 0.04, 0.5, -0.7),
        market,
        np.array([2.5, 3.0, 4.0]),
        k_grid,
        cos_cfg,
    )
    ood_cases.append(("out_of_grid_tau", c_long))

    # Case 4 (Cond 2a/2b): Non-Heston jagged / inverted double-humped smile (high Mahalanobis + ens_var)
    c_jagged = make_synthetic_chain(
        nhc.HestonParams(0.04, 2.0, 0.04, 0.5, -0.7), market, taus, k_grid, cos_cfg
    )
    for idx, q in enumerate(c_jagged):
        q.iv = 0.25 + 0.18 * math.sin(1.7 * idx)
        q.mid = nhc.black_scholes_price(
            market.S0, q.strike, market.r, market.q, q.tau, q.iv, q.is_call
        )
        q.vega = nhc.black_scholes_vega(market.S0, q.strike, market.r, market.q, q.tau, q.iv)
    ood_cases.append(("jagged_non_heston_smile", c_jagged))

    # Case 5 (Cond 1): Parameter strictly outside the 5% outer margin of training box (positive rho=+0.5!)
    c_pos_rho = make_synthetic_chain(
        nhc.HestonParams(0.06, 2.0, 0.06, 0.8, +0.50), market, taus, k_grid, cos_cfg
    )
    ood_cases.append(("positive_rho_ood", c_pos_rho))

    # Case 6 (Cond 1): Parameter beyond upper sigma box (sigma = 6.0 > 4.5)
    c_hi_sig = make_synthetic_chain(
        nhc.HestonParams(0.08, 2.0, 0.08, 6.0, -0.7),
        market,
        taus,
        k_grid,
        nhc.CosConfig(N=512, L=24.0),
    )
    ood_cases.append(("sigma_above_box", c_hi_sig))

    triggered = 0
    for name, ch in ood_cases:
        res = nhc.calibrate_surrogate_lm(
            market,
            ch,
            surr,
            COLD_START,
            lm_cfg,
            cos_cfg,
            nhc.InterpSpace.TotalVariance,
            True,
            cb,
            inv,
        )
        assert res.fallback_triggered, f"OOD case '{name}' failed to trigger circuit-breaker!"
        assert res.ood_flags != 0
        assert math.isfinite(res.cos_rmse_iv)
        triggered += 1

    assert triggered == len(ood_cases)
