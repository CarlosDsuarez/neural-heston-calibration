"""Direct Inverse Network (surface -> Theta_0 warm-start) & 5-Seed Ensemble (Phase 5, Sec. 3.11).

Because surface -> Theta is ill-conditioned (kappa, vbar, sigma weakly identifiable),
the inverse network is NEVER used as a standalone final calibrator: it generates a warm-start
seed Theta_0 refined by LM and audited by COS, while ensemble disagreement serves as an OOD signal.

Compares both Section 3.11 losses:
  (a) `param_mse`: MSE in scaled unconstrained parameter space z_tilde in [-1, 1]^5.
  (b) `surrogate_recon`: Surface reconstruction loss through the frozen Sobolev surrogate
      ||S_frozen(z_hat) - sigma||_2^2 (+ tiny weak-identifiability anchor alpha_param).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nhc.train.surrogate import (
    GridSurrogateMLP,
    act_and_deriv,
    default_z_bounds,
    theta_to_z_np,
    z_to_theta_np,
)

InverseLossMode = Literal["param_mse", "surrogate_recon"]


@dataclass
class InverseConfig:
    hidden_dim: int = 128
    num_layers: int = 3
    activation: str = "silu"
    loss_mode: InverseLossMode = "surrogate_recon"
    alpha_param: float = 0.02  # tiny param anchor for surrogate_recon to break flat valleys
    batch_size: int = 1024
    epochs: int = 80
    lr: float = 3e-3
    min_lr: float = 1e-5
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


class InverseMLP(nn.Module):
    """Maps standardized 180-node IV surface -> scaled parameter z_tilde in [-1, 1]^5."""

    def __init__(
        self,
        in_dim: int = 180,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = "silu",
        x_mean: np.ndarray | None = None,
        x_std: np.ndarray | None = None,
        z_min: np.ndarray | None = None,
        z_max: np.ndarray | None = None,
        taus: np.ndarray | None = None,
        k_scaled: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 5
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

        dims = [in_dim] + [hidden_dim] * num_layers
        self.hidden = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.out = nn.Linear(hidden_dim, self.out_dim)

        for layer in [*self.hidden, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        if x_mean is None:
            x_mean = np.zeros(in_dim, dtype=np.float32)
        if x_std is None:
            x_std = np.ones(in_dim, dtype=np.float32)
        if z_min is None or z_max is None:
            z_min, z_max = default_z_bounds()

        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.tensor(x_std, dtype=torch.float32))
        self.register_buffer("z_min", torch.tensor(z_min, dtype=torch.float32))
        self.register_buffer("z_max", torch.tensor(z_max, dtype=torch.float32))
        if taus is not None and k_scaled is not None:
            self.register_buffer("taus", torch.tensor(taus, dtype=torch.float32))
            self.register_buffer("k_scaled", torch.tensor(k_scaled, dtype=torch.float32))
        else:
            self.register_buffer("taus", torch.empty(0, dtype=torch.float32))
            self.register_buffer("k_scaled", torch.empty(0, dtype=torch.float32))

    def forward_from_std(self, iv_std: torch.Tensor) -> torch.Tensor:
        """Takes standardized IV surface -> predicted scaled z_tilde in (-1, 1)^5."""
        h = iv_std
        for layer in self.hidden:
            a = layer(h)
            h, _ = act_and_deriv(a, self.activation)
        # Bound cleanly inside training box [-1, 1] with smooth tanh
        return torch.tanh(self.out(h))

    def forward(self, iv_raw: torch.Tensor) -> torch.Tensor:
        iv_std = (iv_raw - self.x_mean) / self.x_std
        return self.forward_from_std(iv_std)

    def predict_theta_np(self, iv_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns `(theta [N, 5], z_tilde [N, 5])` in float64."""
        self.eval()
        dev = next(self.parameters()).device
        with torch.no_grad():
            inp = torch.tensor(iv_raw, dtype=torch.float32, device=dev)
            if inp.ndim == 1:
                inp = inp.unsqueeze(0)
            z_tilde = self.forward(inp).cpu().numpy().astype(np.float64)
        z_min = self.z_min.cpu().numpy().astype(np.float64)
        z_max = self.z_max.cpu().numpy().astype(np.float64)
        z = z_min + 0.5 * (z_tilde + 1.0) * (z_max - z_min)
        theta = z_to_theta_np(z)
        return theta, z_tilde


def train_inverse_model(
    train_npz: dict[str, np.ndarray],
    val_npz: dict[str, np.ndarray],
    cfg: InverseConfig,
    frozen_surrogate: GridSurrogateMLP | None = None,
    max_train_samples: int | None = None,
    device: torch.device | None = None,
) -> tuple[InverseMLP, dict]:
    """Trains a single InverseMLP under either `param_mse` or `surrogate_recon` loss."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device is None:
        device = torch.device("cpu")

    if cfg.loss_mode == "surrogate_recon" and frozen_surrogate is None:
        raise ValueError("surrogate_recon loss requires a frozen_surrogate model")

    z_min, z_max = default_z_bounds()
    half_range = 0.5 * (z_max - z_min)

    th_tr = train_npz["theta"]
    iv_tr = train_npz["iv"]
    if max_train_samples is not None and max_train_samples < len(th_tr):
        th_tr = th_tr[:max_train_samples]
        iv_tr = iv_tr[:max_train_samples]

    th_va = val_npz["theta"]
    iv_va = val_npz["iv"]

    z_tr_tilde = ((theta_to_z_np(th_tr) - z_min) / half_range - 1.0).astype(np.float32)
    z_va_tilde = ((theta_to_z_np(th_va) - z_min) / half_range - 1.0).astype(np.float32)

    x_mean = iv_tr.mean(axis=0).astype(np.float32)
    x_std = np.maximum(iv_tr.std(axis=0).astype(np.float32), 1e-4)

    iv_tr_std = torch.tensor((iv_tr - x_mean) / x_std, dtype=torch.float32, device=device)
    iv_va_std = torch.tensor((iv_va - x_mean) / x_std, dtype=torch.float32, device=device)
    iv_tr_raw = torch.tensor(iv_tr, dtype=torch.float32, device=device)
    iv_va_raw = torch.tensor(iv_va, dtype=torch.float32, device=device)
    z_tr_t = torch.tensor(z_tr_tilde, dtype=torch.float32, device=device)
    z_va_t = torch.tensor(z_va_tilde, dtype=torch.float32, device=device)

    model = InverseMLP(
        in_dim=iv_tr.shape[1],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        activation=cfg.activation,
        x_mean=x_mean,
        x_std=x_std,
        z_min=z_min,
        z_max=z_max,
        taus=train_npz["taus"],
        k_scaled=train_npz["k_scaled"],
    ).to(device)

    if frozen_surrogate is not None:
        frozen_surrogate = frozen_surrogate.to(device)
        frozen_surrogate.eval()
        for p in frozen_surrogate.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.min_lr)

    N = len(iv_tr_std)
    bs = cfg.batch_size
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for _epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(N, generator=gen).to(device)
        for start in range(0, N, bs):
            idx = perm[start : start + bs]
            opt.zero_grad(set_to_none=True)
            z_hat = model.forward_from_std(iv_tr_std[idx])
            if cfg.loss_mode == "param_mse":
                loss = F.mse_loss(z_hat, z_tr_t[idx])
            else:
                assert frozen_surrogate is not None
                iv_hat = frozen_surrogate(z_hat)
                # Scale IV error to vol points^2 (~10000x) so gradient magnitudes match param MSE
                loss_surf = F.mse_loss(iv_hat * 100.0, iv_tr_raw[idx] * 100.0)
                loss = loss_surf + cfg.alpha_param * F.mse_loss(z_hat, z_tr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            zv = model.forward_from_std(iv_va_std)
            if cfg.loss_mode == "param_mse":
                vloss = float(F.mse_loss(zv, z_va_t).item())
            else:
                assert frozen_surrogate is not None
                vloss = float(F.mse_loss(frozen_surrogate(zv) * 100.0, iv_va_raw * 100.0).item())
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    return model, {"best_val_loss": best_val}


def predict_ensemble_np(
    members: list[InverseMLP], iv_raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates a 5-seed ensemble on `iv_raw [N, 180]`.
    Returns:
      - `theta_mean`: `[N, 5]` (mapped from mean z_tilde across ensemble)
      - `z_tilde_mean`: `[N, 5]`
      - `z_tilde_var`: `[N]` scalar disagreement metric (mean variance across the 5 z dims)
    """
    z_preds = np.stack([m.predict_theta_np(iv_raw)[1] for m in members], axis=0)  # (E, N, 5)
    z_mean = z_preds.mean(axis=0)
    z_var = z_preds.var(axis=0).mean(axis=-1)  # (N,)
    z_min = members[0].z_min.cpu().numpy().astype(np.float64)
    z_max = members[0].z_max.cpu().numpy().astype(np.float64)
    z_unscaled = z_min + 0.5 * (z_mean + 1.0) * (z_max - z_min)
    theta_mean = z_to_theta_np(z_unscaled)
    return theta_mean, z_mean, z_var


def fit_ood_mahalanobis_stats(
    train_iv: np.ndarray,
    val_iv: np.ndarray,
    members: list[InverseMLP],
    reg: float = 1e-4,
    quantile: float = 0.995,
) -> dict:
    """Calibrates Mahalanobis distance and ensemble-variance OOD thresholds on validation data
    for the Phase 6 C++ circuit-breaker (Section 8.1).
    """
    mu = train_iv.mean(axis=0).astype(np.float64)
    centered = train_iv.astype(np.float64) - mu
    cov = (centered.T @ centered) / max(len(train_iv) - 1, 1)
    # Tikhonov regularization for well-conditioned 180x180 precision matrix
    cov_reg = cov + reg * np.eye(cov.shape[0], dtype=np.float64)
    prec = np.linalg.inv(cov_reg)

    val_c = val_iv.astype(np.float64) - mu
    val_mahal = np.sqrt(np.maximum(np.sum((val_c @ prec) * val_c, axis=1), 0.0))
    _, _, val_ens_var = predict_ensemble_np(members, val_iv)

    # Also compute ATM IV training range (Section 8.1 #5)
    # Closest node to k = 0 across maturities
    atm_iv_min = float(np.percentile(train_iv.min(axis=1), 0.1))
    atm_iv_max = float(np.percentile(train_iv.max(axis=1), 99.9))

    return {
        "dim": int(len(mu)),
        "reg": reg,
        "quantile": quantile,
        "mahal_threshold": float(np.quantile(val_mahal, quantile)),
        "mahal_val_p50": float(np.median(val_mahal)),
        "mahal_val_p99": float(np.percentile(val_mahal, 99)),
        "ens_var_threshold": float(np.quantile(val_ens_var, quantile)),
        "ens_var_val_p50": float(np.median(val_ens_var)),
        "ens_var_val_p99": float(np.percentile(val_ens_var, 99)),
        "iv_support_min": max(0.01, atm_iv_min),
        "iv_support_max": atm_iv_max,
        "mean": mu.tolist(),
        "precision_diag_summary": {
            "min": float(np.min(np.diag(prec))),
            "max": float(np.max(np.diag(prec))),
        },
        # Store full precision matrix as separate numpy array when saved to .npz/.bin
        "_precision_matrix": prec,
    }


def cosine_iv_rmse_vol_pts(pred: np.ndarray, true: np.ndarray) -> float:
    diff = (pred - true) * 100.0
    return float(math.sqrt(np.mean(diff * diff)))
