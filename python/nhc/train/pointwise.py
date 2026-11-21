"""Pointwise Random-Grid Surrogate Ablation (Phase 5 T5.2, Baschetti et al. 2024, Hypothesis H5).

Architecture:
  Input:  7D vector (z_tilde_0..4, tau_tilde, k_scaled_tilde) in [-1, 1]^7
  Hidden: `num_layers` layers of width `hidden_dim` with SiLU activation
  Output: 1 scalar (sigma_IV)

Because evaluating a 200-option chain requires 200 separate MLP + 5-tangent Jacobian evaluations
per LM step (vs. 1 evaluation for the 180-output grid surrogate), Hypothesis H5 predicts that
this pointwise approach is slower than exact C++ COS in standard Heston.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nhc.train.surrogate import act_and_deriv, default_z_bounds, theta_to_z_np


@dataclass
class PointwiseConfig:
    hidden_dim: int = 64
    num_layers: int = 4
    activation: str = "silu"
    points_per_surface: int = 16  # random grid points sampled per surface per epoch
    batch_size: int = 2048
    epochs: int = 80
    lr: float = 3e-3
    min_lr: float = 1e-5
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


class PointwiseSurrogateMLP(nn.Module):
    """7-input (Theta_z_tilde[5], tau_tilde, k_scaled_tilde) -> 1-output (sigma_IV) MLP."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        activation: str = "silu",
        z_min: np.ndarray | None = None,
        z_max: np.ndarray | None = None,
        tau_min: float = 0.025,
        tau_max: float = 2.0,
        k_min: float = -0.55,
        k_max: float = 0.35,
        y_mean: float = 0.35,
        y_std: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_dim = 7
        self.out_dim = 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

        dims = [self.in_dim] + [hidden_dim] * num_layers
        self.hidden = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.out = nn.Linear(hidden_dim, 1)
        for layer in [*self.hidden, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        if z_min is None or z_max is None:
            z_min, z_max = default_z_bounds()

        in_min = np.concatenate([z_min, [math.log(tau_min), k_min]]).astype(np.float32)
        in_max = np.concatenate([z_max, [math.log(tau_max), k_max]]).astype(np.float32)
        self.register_buffer("in_min", torch.tensor(in_min, dtype=torch.float32))
        self.register_buffer("in_max", torch.tensor(in_max, dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor([float(y_mean)], dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor([float(y_std)], dtype=torch.float32))

    def forward_std(self, x7: torch.Tensor) -> torch.Tensor:
        h = x7
        for layer in self.hidden:
            a = layer(h)
            h, _ = act_and_deriv(a, self.activation)
        return self.out(h)

    def forward(self, x7: torch.Tensor) -> torch.Tensor:
        return self.forward_std(x7) * self.y_std + self.y_mean


def train_pointwise_surrogate(
    train_npz: dict[str, np.ndarray],
    val_npz: dict[str, np.ndarray],
    cfg: PointwiseConfig,
    max_train_surfaces: int | None = None,
    device: torch.device | None = None,
) -> tuple[PointwiseSurrogateMLP, dict]:
    """Trains the 7D -> 1D pointwise surrogate by sampling random (tau, k) nodes from each surface."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device is None:
        device = torch.device("cpu")

    taus = train_npz["taus"]
    k_scaled = train_npz["k_scaled"]
    n_tau, n_k = len(taus), len(k_scaled)
    G = n_tau * n_k

    # Precompute scaled (log_tau_tilde, k_scaled_tilde) for every grid node 0..G-1
    log_taus = np.log(taus)
    lt_min, lt_max = float(log_taus[0]), float(log_taus[-1])
    ks_min, ks_max = float(k_scaled[0]), float(k_scaled[-1])
    tau_grid_tilde = 2.0 * (np.repeat(log_taus, n_k) - lt_min) / (lt_max - lt_min) - 1.0
    k_grid_tilde = 2.0 * (np.tile(k_scaled, n_tau) - ks_min) / (ks_max - ks_min) - 1.0
    tk_grid = torch.tensor(
        np.column_stack([tau_grid_tilde, k_grid_tilde]), dtype=torch.float32, device=device
    )  # (G, 2)

    z_min, z_max = default_z_bounds()
    half_range = 0.5 * (z_max - z_min)

    th_tr = train_npz["theta"]
    iv_tr = train_npz["iv"]
    if max_train_surfaces is not None and max_train_surfaces < len(th_tr):
        th_tr = th_tr[:max_train_surfaces]
        iv_tr = iv_tr[:max_train_surfaces]

    x5_tr = torch.tensor(
        ((theta_to_z_np(th_tr) - z_min) / half_range - 1.0).astype(np.float32), device=device
    )
    iv_tr_t = torch.tensor(iv_tr, dtype=torch.float32, device=device)

    y_mean = float(iv_tr.mean())
    y_std = max(float(iv_tr.std()), 1e-6)
    iv_tr_std = (iv_tr_t - y_mean) / y_std

    # Validation subset (first 2,000 surfaces x all 180 nodes = 360k points)
    n_val_surf = min(2000, len(val_npz["theta"]))
    x5_va = torch.tensor(
        ((theta_to_z_np(val_npz["theta"][:n_val_surf]) - z_min) / half_range - 1.0).astype(
            np.float32
        ),
        device=device,
    )
    iv_va_np = val_npz["iv"][:n_val_surf].astype(np.float64)

    model = PointwiseSurrogateMLP(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        activation=cfg.activation,
        z_min=z_min,
        z_max=z_max,
        tau_min=float(taus[0]),
        tau_max=float(taus[-1]),
        k_min=ks_min,
        k_max=ks_max,
        y_mean=y_mean,
        y_std=y_std,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.min_lr)

    N_surf = len(x5_tr)
    pts_per_surf = cfg.points_per_surface
    surf_bs = max(1, cfg.batch_size // pts_per_surf)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    best_rmse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(N_surf, generator=gen).to(device)
        for start in range(0, N_surf, surf_bs):
            s_idx = perm[start : start + surf_bs]
            B = len(s_idx)
            # Sample `pts_per_surf` random grid indices per surface (random-grid training)
            g_idx = torch.randint(0, G, (B, pts_per_surf), generator=gen).to(device)
            x5_rep = x5_tr[s_idx].unsqueeze(1).expand(B, pts_per_surf, 5)
            tk_sel = tk_grid[g_idx]  # (B, P, 2)
            x7 = torch.cat([x5_rep, tk_sel], dim=-1).reshape(-1, 7)
            y_tgt = iv_tr_std[s_idx.unsqueeze(1), g_idx].reshape(-1, 1)

            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model.forward_std(x7), y_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.epochs:
            model.eval()
            with torch.no_grad():
                preds = []
                for s0 in range(0, n_val_surf, 256):
                    xb = x5_va[s0 : s0 + 256]
                    nb = len(xb)
                    x7_v = torch.cat(
                        [
                            xb.unsqueeze(1).expand(nb, G, 5),
                            tk_grid.unsqueeze(0).expand(nb, G, 2),
                        ],
                        dim=-1,
                    ).reshape(-1, 7)
                    preds.append(model(x7_v).reshape(nb, G).cpu().numpy())
                pred_va = np.concatenate(preds, axis=0)
                rmse_vp = float(math.sqrt(np.mean(((pred_va - iv_va_np) * 100.0) ** 2)))
            if rmse_vp < best_rmse:
                best_rmse = rmse_vp
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()
    return model, {"best_val_rmse_iv_vp": best_rmse}


def evaluate_pointwise_surrogate(
    model: PointwiseSurrogateMLP,
    test_npz: dict[str, np.ndarray],
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    taus = test_npz["taus"]
    k_scaled = test_npz["k_scaled"]
    n_tau, n_k = len(taus), len(k_scaled)
    G = n_tau * n_k

    in_min = model.in_min.cpu().numpy().astype(np.float64)
    in_max = model.in_max.cpu().numpy().astype(np.float64)
    z_min, z_max = in_min[:5], in_max[:5]
    lt_min, lt_max = float(in_min[5]), float(in_max[5])
    ks_min, ks_max = float(in_min[6]), float(in_max[6])

    log_taus = np.log(taus)
    tau_grid_tilde = 2.0 * (np.repeat(log_taus, n_k) - lt_min) / (lt_max - lt_min) - 1.0
    k_grid_tilde = 2.0 * (np.tile(k_scaled, n_tau) - ks_min) / (ks_max - ks_min) - 1.0
    tk_grid = torch.tensor(
        np.column_stack([tau_grid_tilde, k_grid_tilde]), dtype=torch.float32, device=device
    )

    x5 = torch.tensor(
        ((theta_to_z_np(test_npz["theta"]) - z_min) / (0.5 * (z_max - z_min)) - 1.0).astype(
            np.float32
        ),
        device=device,
    )
    iv_true = test_npz["iv"].astype(np.float64)
    stratum = test_npz["stratum"]

    preds = []
    with torch.no_grad():
        for s0 in range(0, len(x5), 256):
            xb = x5[s0 : s0 + 256]
            nb = len(xb)
            x7 = torch.cat(
                [xb.unsqueeze(1).expand(nb, G, 5), tk_grid.unsqueeze(0).expand(nb, G, 2)], dim=-1
            ).reshape(-1, 7)
            preds.append(model(x7).reshape(nb, G).cpu().numpy().astype(np.float64))
    iv_pred = np.concatenate(preds, axis=0)

    def _sub(mask: np.ndarray) -> dict[str, float]:
        err_vp = (iv_pred[mask] - iv_true[mask]) * 100.0
        abs_vp = np.abs(err_vp)
        return {
            "n_samples": int(mask.sum()),
            "rmse_iv_vp": float(math.sqrt(np.mean(err_vp * err_vp))),
            "mae_iv_vp": float(np.mean(abs_vp)),
            "p99_abs_iv_vp": float(np.percentile(abs_vp, 99)),
        }

    return {
        "all": _sub(np.ones(len(stratum), dtype=bool)),
        "in_domain_normal_emp": _sub(stratum != 1),
        "stress": _sub(stratum == 1),
    }
