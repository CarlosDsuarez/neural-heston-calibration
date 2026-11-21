"""Phase 5 unit tests: InverseMLP (both losses), PointwiseSurrogateMLP, and NHC1 `.bin` exporter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nhc.export.to_bin import (
    compute_grid_hash,
    export_grid_surrogate,
    export_inverse_member,
    export_pointwise_surrogate,
    inspect_nhc1_header,
)
from nhc.train.inverse import InverseConfig, predict_ensemble_np, train_inverse_model
from nhc.train.pointwise import (
    PointwiseConfig,
    evaluate_pointwise_surrogate,
    train_pointwise_surrogate,
)
from nhc.train.surrogate import SurrogateConfig, train_surrogate


def test_t51_t52_t53_inverse_pointwise_and_nhc1_export(tmp_path: Path) -> None:
    """End-to-end smoke test of surrogate, inverse (both losses), pointwise, and NHC1 export."""
    raw = dict(np.load(Path("data/mini_synthetic/dataset.npz")))
    tr = {k: (v[:160] if k in ("theta", "stratum", "iv", "jac") else v) for k, v in raw.items()}
    va = {k: (v[160:] if k in ("theta", "stratum", "iv", "jac") else v) for k, v in raw.items()}
    dev = torch.device("cpu")

    # 1. Tiny surrogate
    s_cfg = SurrogateConfig(
        hidden_dim=16,
        num_layers=2,
        activation="silu",
        lambda_grad=0.5,
        epochs=10,
        batch_size=32,
        seed=1,
    )
    surr, _ = train_surrogate(tr, va, s_cfg, device=dev)

    # 2. Inverse with param_mse & surrogate_recon
    inv_mse, _ = train_inverse_model(
        tr,
        va,
        InverseConfig(
            hidden_dim=32, num_layers=2, loss_mode="param_mse", epochs=10, batch_size=32, seed=1
        ),
        device=dev,
    )
    inv_rec, _ = train_inverse_model(
        tr,
        va,
        InverseConfig(
            hidden_dim=32,
            num_layers=2,
            loss_mode="surrogate_recon",
            epochs=10,
            batch_size=32,
            seed=2,
        ),
        frozen_surrogate=surr,
        device=dev,
    )
    th_ens, z_ens, z_var = predict_ensemble_np([inv_mse, inv_rec], va["iv"])
    assert th_ens.shape == (len(va["iv"]), 5)
    assert z_ens.shape == (len(va["iv"]), 5)
    assert z_var.shape == (len(va["iv"]),)
    assert np.all(np.isfinite(th_ens))
    assert np.all(z_var >= 0.0)

    # 3. Pointwise 7D -> 1D
    pw_cfg = PointwiseConfig(
        hidden_dim=16, num_layers=2, points_per_surface=8, batch_size=128, epochs=5, seed=3
    )
    pw, _ = train_pointwise_surrogate(tr, va, pw_cfg, device=dev)
    pw_ev = evaluate_pointwise_surrogate(pw, va, device=dev)
    assert pw_ev["all"]["rmse_iv_vp"] > 0.0 and np.isfinite(pw_ev["all"]["rmse_iv_vp"])

    # 4. NHC1 binary export + header & grid-hash verification
    b_surr = tmp_path / "surr.bin"
    b_inv = tmp_path / "inv.bin"
    b_pw = tmp_path / "pw.bin"
    export_grid_surrogate(surr, b_surr, s_cfg.to_dict(), taus=raw["taus"], k_scaled=raw["k_scaled"])
    export_inverse_member(
        inv_rec, b_inv, {"loss_mode": "surrogate_recon"}, taus=raw["taus"], k_scaled=raw["k_scaled"]
    )
    export_pointwise_surrogate(pw, b_pw, raw["taus"], raw["k_scaled"], pw_cfg.to_dict())

    for b in (b_surr, b_inv, b_pw):
        assert b.exists() and b.with_suffix(".json").exists()
        hdr = inspect_nhc1_header(b)
        assert hdr["format_version"] == 1
        assert hdr["grid_hash_valid"]
        assert hdr["grid_hash"] == compute_grid_hash(raw["taus"], raw["k_scaled"])
        # Mutated grid must produce a different hash
        bad_taus = raw["taus"].copy()
        bad_taus[0] += 0.001
        assert hdr["grid_hash"] != compute_grid_hash(bad_taus, raw["k_scaled"])
