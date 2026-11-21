"""Phase 4 unit tests: forward-mode MLP Jacobian parity vs torch.autograd, no-arb auditor,
and quick smoke training of Baseline vs Sobolev on the committed mini_synthetic dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nhc.eval.noarb import audit_surface_noarb
from nhc.train.surrogate import (
    GridSurrogateMLP,
    SurrogateConfig,
    evaluate_surrogate,
    train_surrogate,
)


def test_forward_mode_jacobian_matches_torch_autograd() -> None:
    """Section 3.8 mandatory parity test: analytic forward-mode Jacobian == torch.autograd.functional.jacobian."""
    torch.manual_seed(0)
    taus = np.linspace(0.05, 2.0, 10)
    k_scaled = np.linspace(-0.55, 0.35, 18)

    for act in ("silu", "softplus", "elu", "tanh"):
        model = GridSurrogateMLP(
            out_dim=180,
            hidden_dim=32,
            num_layers=3,
            activation=act,
            taus=taus,
            k_scaled=k_scaled,
        ).double()
        x = torch.randn(4, 5, dtype=torch.float64)
        y_fwd, J_fwd = model.forward_std_with_jac(x)

        for b in range(4):
            J_auto = torch.autograd.functional.jacobian(
                lambda z, m=model: m.forward_std(z.unsqueeze(0)).squeeze(0), x[b]
            )
            err = float((J_fwd[b] - J_auto).abs().max().item())
            assert err < 1e-12, f"Activation {act}: forward-mode Jacobian mismatch {err:.3e}"
            y_single = model.forward_std(x[b : b + 1]).squeeze(0)
            assert float((y_fwd[b] - y_single).abs().max().item()) < 1e-13


def test_noarb_auditor_flags_injected_violations_and_passes_cos_truth() -> None:
    """Verify static no-arbitrage auditor passes exact COS Heston surfaces and catches violations."""
    data = dict(np.load(Path("data/mini_synthetic/dataset.npz")))
    taus = data["taus"]
    k_scaled = data["k_scaled"]
    iv_true = data["iv"][:20]

    rep_true = audit_surface_noarb(iv_true, taus, k_scaled)
    assert rep_true.calendar_violation_pct < 1.0
    assert rep_true.butterfly_violation_pct < 1.0

    # Inject blatant calendar violation: make maturity 1 lower vol than maturity 0
    bad = iv_true[0].copy().reshape(len(taus), len(k_scaled))
    bad[1, :] = bad[0, :] * 0.2
    rep_bad = audit_surface_noarb(bad, taus, k_scaled)
    assert rep_bad.calendar_violation_pct > 5.0
    assert rep_bad.worst_calendar_violation < -1e-4


def test_sobol_mini_baseline_vs_sobolev_smoke() -> None:
    """Fast smoke test on mini_synthetic: Sobolev training lowers Jacobian error vs baseline."""
    raw = dict(np.load(Path("data/mini_synthetic/dataset.npz")))
    tr = {k: (v[:160] if k in ("theta", "stratum", "iv", "jac") else v) for k, v in raw.items()}
    va = {k: (v[160:] if k in ("theta", "stratum", "iv", "jac") else v) for k, v in raw.items()}
    dev = torch.device("cpu")

    cfg_base = SurrogateConfig(
        hidden_dim=32,
        num_layers=3,
        activation="silu",
        lambda_grad=0.0,
        epochs=25,
        batch_size=32,
        seed=1,
    )
    cfg_sob = SurrogateConfig(
        hidden_dim=32,
        num_layers=3,
        activation="silu",
        lambda_grad=0.5,
        epochs=25,
        batch_size=32,
        seed=1,
    )

    m_base, _ = train_surrogate(tr, va, cfg_base, device=dev)
    m_sob, _ = train_surrogate(tr, va, cfg_sob, device=dev)
    e_base = evaluate_surrogate(m_base, va, device=dev, audit_noarb_samples=10)
    e_sob = evaluate_surrogate(m_sob, va, device=dev, audit_noarb_samples=10)

    assert e_sob["all"]["jac_rel_err_median"] < e_base["all"]["jac_rel_err_median"]
