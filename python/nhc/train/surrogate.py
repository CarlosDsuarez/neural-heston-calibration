"""Grid Surrogate MLP with Analytic Forward-Mode Sobolev Training (Phase 4, Section 3.7–3.8).

Architecture:
  Input:  z_tilde in [-1, 1]^5, where z = (ln v0, ln kappa, ln vbar, ln sigma, artanh rho)
          scaled linearly from training box [z_min, z_max].
  Hidden: `num_layers` linear layers of width `hidden_dim` with smooth activation
          (`silu`, `softplus`, `elu`, `tanh`). ReLU is intentionally avoided (discontinuous J).
  Output: 180 grid nodes (sigma_IV or total variance w = sigma^2 * tau).

Sobolev training uses analytic forward-mode tangent propagation with 5 tangents (Section 3.8),
computing exact (y, dy/dz_tilde) in a single batched pass with full autograd support for
weight updates.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nhc.data.synthetic import GLOBAL_HIGH, GLOBAL_LOW
from nhc.eval.noarb import audit_surface_noarb

ActivationName = Literal["silu", "softplus", "elu", "tanh"]
OutputMode = Literal["iv", "total_var"]

ACTIVATION_IDS: dict[str, int] = {
    "silu": 0,
    "softplus": 1,
    "elu": 2,
    "tanh": 3,
}


def theta_to_z_np(theta: np.ndarray) -> np.ndarray:
    """Maps raw Heston parameters [v0, kappa, vbar, sigma, rho] -> unconstrained z."""
    th = np.asarray(theta, dtype=np.float64)
    z = np.empty_like(th)
    z[..., :4] = np.log(np.clip(th[..., :4], 1e-8, None))
    z[..., 4] = np.arctanh(np.clip(th[..., 4], -0.9999, 0.9999))
    return z


def z_to_theta_np(z: np.ndarray) -> np.ndarray:
    """Inverse map unconstrained z -> raw Heston parameters."""
    z_arr = np.asarray(z, dtype=np.float64)
    th = np.empty_like(z_arr)
    th[..., :4] = np.exp(z_arr[..., :4])
    th[..., 4] = np.tanh(z_arr[..., 4])
    return th


def dtheta_dz_diag_np(theta: np.ndarray) -> np.ndarray:
    """Diagonal Jacobian dTheta_j / dz_j = (v0, kappa, vbar, sigma, 1 - rho^2)."""
    th = np.asarray(theta, dtype=np.float64)
    d = np.empty_like(th)
    d[..., :4] = th[..., :4]
    d[..., 4] = 1.0 - th[..., 4] ** 2
    return d


def default_z_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Fixed z-space training box matching Section 5.1 global strata bounds."""
    return theta_to_z_np(GLOBAL_LOW), theta_to_z_np(GLOBAL_HIGH)


def act_and_deriv(a: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth activation h = act(a) and exact elementwise derivative h' = act'(a)."""
    if name == "silu":
        sig = torch.sigmoid(a)
        h = a * sig
        da = sig * (1.0 + a * (1.0 - sig))
        return h, da
    if name == "softplus":
        h = F.softplus(a)
        da = torch.sigmoid(a)
        return h, da
    if name == "elu":
        h = F.elu(a)
        da = torch.where(a > 0.0, torch.ones_like(a), torch.exp(torch.clamp(a, max=0.0)))
        return h, da
    if name == "tanh":
        h = torch.tanh(a)
        da = 1.0 - h * h
        return h, da
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class SurrogateConfig:
    hidden_dim: int = 64
    num_layers: int = 4
    activation: ActivationName = "silu"
    output_mode: OutputMode = "iv"
    lambda_grad: float = 0.0
    lambda_arb: float = 0.0
    batch_size: int = 512
    epochs: int = 120
    lr: float = 3e-3
    min_lr: float = 1e-5
    weight_decay: float = 0.0
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


class GridSurrogateMLP(nn.Module):
    """Smooth MLP surrogate with fused analytic forward-mode Jacobian (5 tangents)."""

    def __init__(
        self,
        out_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        activation: ActivationName = "silu",
        output_mode: OutputMode = "iv",
        z_min: np.ndarray | None = None,
        z_max: np.ndarray | None = None,
        y_mean: np.ndarray | None = None,
        y_std: float = 1.0,
        taus: np.ndarray | None = None,
        k_scaled: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = 5
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.output_mode = output_mode

        dims = [self.in_dim] + [hidden_dim] * num_layers
        self.hidden = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.out = nn.Linear(hidden_dim, out_dim)

        for layer in [*self.hidden, self.out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        if z_min is None or z_max is None:
            z_min, z_max = default_z_bounds()
        if y_mean is None:
            y_mean = np.zeros(out_dim, dtype=np.float32)

        self.register_buffer("z_min", torch.tensor(z_min, dtype=torch.float32))
        self.register_buffer("z_max", torch.tensor(z_max, dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.tensor(float(y_std), dtype=torch.float32))
        self.register_buffer("eye5", torch.eye(5, dtype=torch.float32))

        if taus is not None and k_scaled is not None:
            self.register_buffer("taus", torch.tensor(taus, dtype=torch.float32))
            self.register_buffer("k_scaled", torch.tensor(k_scaled, dtype=torch.float32))
            self.register_buffer(
                "tau_grid",
                torch.tensor(np.repeat(taus, len(k_scaled)), dtype=torch.float32),
            )
        else:
            self.register_buffer("taus", torch.empty(0, dtype=torch.float32))
            self.register_buffer("k_scaled", torch.empty(0, dtype=torch.float32))
            self.register_buffer("tau_grid", torch.empty(0, dtype=torch.float32))

    def forward_std(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in standardized output units from scaled input x in [-1, 1]^5."""
        h = x
        for layer in self.hidden:
            a = layer(h)
            h, _ = act_and_deriv(a, self.activation)
        return self.out(h)

    def forward_std_with_jac(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Analytic forward-mode pass returning `(y_std [B, G], dy_std/dx [B, G, 5])`."""
        B = x.shape[0]
        h = x
        # Maintain J in shape (B, 5, d_l) so J @ W^T is a single batched GEMM
        J = self.eye5.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, 5, 5)
        for layer in self.hidden:
            a = layer(h)  # (B, D)
            A = torch.matmul(J, layer.weight.t())  # (B, 5, D)
            h, da = act_and_deriv(a, self.activation)  # (B, D), (B, D)
            J = A * da.unsqueeze(1)  # (B, 5, D)
        y = self.out(h)  # (B, G)
        J_y = torch.matmul(J, self.out.weight.t()).transpose(1, 2)  # (B, G, 5)
        return y, J_y

    def raw_output_from_std(self, y_std: torch.Tensor) -> torch.Tensor:
        return y_std * self.y_std + self.y_mean

    def iv_from_std(self, y_std: torch.Tensor) -> torch.Tensor:
        raw = self.raw_output_from_std(y_std)
        if self.output_mode == "iv":
            return raw
        # Total variance w -> sigma_IV = sqrt(max(w, 1e-10) / tau)
        return torch.sqrt(torch.clamp(raw, min=1e-10) / self.tau_grid.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.iv_from_std(self.forward_std(x))

    def forward_iv_and_jac_x(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(sigma_IV [B, G], d(sigma_IV)/dx [B, G, 5])`."""
        y_std, J_std = self.forward_std_with_jac(x)
        raw = self.raw_output_from_std(y_std)
        J_raw = J_std * self.y_std
        if self.output_mode == "iv":
            return raw, J_raw
        w = torch.clamp(raw, min=1e-10)
        tau_g = self.tau_grid.unsqueeze(0)
        iv = torch.sqrt(w / tau_g)
        # d(iv)/dw = 1 / (2 * tau * iv)
        factor = (1.0 / (2.0 * tau_g * iv)).unsqueeze(-1)
        return iv, J_raw * factor


def build_calendar_interp_weights(taus: np.ndarray, k_scaled: np.ndarray) -> torch.Tensor:
    """Precomputes linear interpolation weights mapping slice j+1 onto slice j's k coordinates
    so calendar monotonicity w_{j+1}(k_{j,i}) >= w_j(k_{j,i}) can be evaluated differentiably.
    Returns tensor of shape `(n_tau-1, n_k, n_k)`.
    """
    n_tau, n_k = len(taus), len(k_scaled)
    sqrt_taus = np.sqrt(taus)
    W = np.zeros((n_tau - 1, n_k, n_k), dtype=np.float32)
    for j in range(n_tau - 1):
        ratio = sqrt_taus[j] / sqrt_taus[j + 1]
        u_query = k_scaled * ratio  # strictly inside [k_scaled[0], k_scaled[-1]]
        for i, uq in enumerate(u_query):
            pos = int(np.searchsorted(k_scaled, uq, side="right")) - 1
            pos = max(0, min(n_k - 2, pos))
            t = (uq - k_scaled[pos]) / (k_scaled[pos + 1] - k_scaled[pos])
            W[j, i, pos] = 1.0 - t
            W[j, i, pos + 1] = t
    return torch.tensor(W, dtype=torch.float32)


def noarb_penalty(
    iv_pred: torch.Tensor,
    taus: torch.Tensor,
    k_scaled: torch.Tensor,
    cal_weights: torch.Tensor,
) -> torch.Tensor:
    """Differentiable calendar + Durrleman butterfly penalty on predicted IV grid `(B, J*I)`."""
    B = iv_pred.shape[0]
    n_tau, n_k = len(taus), len(k_scaled)
    sig = torch.clamp(iv_pred.view(B, n_tau, n_k), min=1e-4)
    w = (sig * sig) * taus.view(1, n_tau, 1)  # (B, n_tau, n_k)

    # 1. Calendar penalty: w_{j+1}(k_{j,i}) - w_j(k_{j,i}) >= 0
    # cal_weights: (n_tau-1, n_k, n_k), w[:, 1:, :]: (B, n_tau-1, n_k)
    w_next_at_k = torch.einsum("jik,bjk->bji", cal_weights, w[:, 1:, :])
    cal_viol = F.relu(w[:, :-1, :] - w_next_at_k)
    loss_cal = (cal_viol * cal_viol).mean()

    # 2. Butterfly Durrleman penalty via central finite differences along interior k nodes
    sqrt_t = torch.sqrt(taus).view(1, n_tau, 1)
    k = k_scaled.view(1, 1, n_k) * sqrt_t  # (B=1, n_tau, n_k)
    dk = k[:, :, 1:] - k[:, :, :-1]  # uniform per slice: dk[:, :, 0]
    h = dk[:, :, :1]  # (1, J, 1)

    w_c = w[:, :, 1:-1]
    k_c = k[:, :, 1:-1]
    wp = (w[:, :, 2:] - w[:, :, :-2]) / (2.0 * h)
    wpp = (w[:, :, 2:] - 2.0 * w_c + w[:, :, :-2]) / (h * h)

    g = (1.0 - (k_c * wp) / (2.0 * w_c)) ** 2 - 0.25 * (wp * wp) * (1.0 / w_c + 0.25) + 0.5 * wpp
    bfly_viol = F.relu(-g)
    loss_bfly = (bfly_viol * bfly_viol).mean()
    return loss_cal + 0.1 * loss_bfly


def prepare_tensors(
    npz_data: dict[str, np.ndarray],
    z_min: np.ndarray,
    z_max: np.ndarray,
    output_mode: OutputMode = "iv",
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Converts dataset arrays to `(x [-1,1], y_raw, J_raw_x, stratum)` float32 tensors."""
    theta = npz_data["theta"]
    iv = npz_data["iv"]
    jac = npz_data["jac"]
    stratum = npz_data["stratum"]
    taus = npz_data["taus"]
    k_scaled = npz_data["k_scaled"]

    if max_samples is not None and max_samples < len(theta):
        theta = theta[:max_samples]
        iv = iv[:max_samples]
        jac = jac[:max_samples]
        stratum = stratum[:max_samples]

    z = theta_to_z_np(theta)
    half_range = 0.5 * (z_max - z_min)
    x = (z - z_min) / half_range - 1.0  # in [-1, 1]

    # Chain rule: d(iv)/dx_j = d(iv)/dTheta_j * (dTheta_j / dz_j) * half_range_j
    scale_x = dtheta_dz_diag_np(theta) * half_range[None, :]  # (N, 5)
    J_iv_x = jac.astype(np.float64) * scale_x[:, None, :]  # (N, G, 5)

    if output_mode == "iv":
        y_raw = iv.astype(np.float32)
        J_raw_x = J_iv_x.astype(np.float32)
    else:
        tau_g = np.repeat(taus, len(k_scaled))[None, :]  # (1, G)
        y_raw = ((iv.astype(np.float64) ** 2) * tau_g).astype(np.float32)
        # dw/dx = 2 * tau * iv * d(iv)/dx
        J_raw_x = (2.0 * tau_g[:, :, None] * iv.astype(np.float64)[:, :, None] * J_iv_x).astype(
            np.float32
        )

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y_raw, dtype=torch.float32),
        torch.tensor(J_raw_x, dtype=torch.float32),
        stratum,
    )


def evaluate_surrogate(
    model: GridSurrogateMLP,
    test_npz: dict[str, np.ndarray],
    device: torch.device | None = None,
    audit_noarb_samples: int = 500,
) -> dict:
    """Evaluates surrogate IV accuracy (vol points), Jacobian relative error, and no-arb rates."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    z_min = model.z_min.cpu().numpy().astype(np.float64)
    z_max = model.z_max.cpu().numpy().astype(np.float64)
    taus = test_npz["taus"]
    k_scaled = test_npz["k_scaled"]

    x_all, _, _, stratum = prepare_tensors(test_npz, z_min, z_max, output_mode="iv")
    iv_true = test_npz["iv"].astype(np.float64)
    # True Jacobian in scaled x-space (same units across all 5 parameters!)
    theta = test_npz["theta"]
    half_range = 0.5 * (z_max - z_min)
    scale_x = dtheta_dz_diag_np(theta) * half_range[None, :]
    J_true_x = test_npz["jac"].astype(np.float64) * scale_x[:, None, :]

    preds_iv = []
    preds_Jx = []
    with torch.no_grad():
        bs = 1024
        for start in range(0, len(x_all), bs):
            xb = x_all[start : start + bs].to(device)
            iv_b, Jx_b = model.forward_iv_and_jac_x(xb)
            preds_iv.append(iv_b.cpu().numpy().astype(np.float64))
            preds_Jx.append(Jx_b.cpu().numpy().astype(np.float64))

    iv_pred = np.concatenate(preds_iv, axis=0)
    J_pred_x = np.concatenate(preds_Jx, axis=0)

    def _subset_metrics(mask: np.ndarray) -> dict[str, float]:
        err_vp = (iv_pred[mask] - iv_true[mask]) * 100.0  # vol points
        abs_vp = np.abs(err_vp)
        j_diff = np.linalg.norm(J_pred_x[mask] - J_true_x[mask], axis=(1, 2))
        j_norm = np.maximum(np.linalg.norm(J_true_x[mask], axis=(1, 2)), 1e-8)
        j_rel = j_diff / j_norm
        return {
            "n_samples": int(mask.sum()),
            "rmse_iv_vp": float(math.sqrt(np.mean(err_vp * err_vp))),
            "mae_iv_vp": float(np.mean(abs_vp)),
            "p99_abs_iv_vp": float(np.percentile(abs_vp, 99)),
            "jac_rel_err_median": float(np.median(j_rel)),
            "jac_rel_err_mean": float(np.mean(j_rel)),
            "jac_rel_err_p95": float(np.percentile(j_rel, 95)),
        }

    mask_all = np.ones(len(stratum), dtype=bool)
    mask_in_domain = stratum != 1  # Normal (0) + Empirical (2)
    mask_stress = stratum == 1

    out: dict = {
        "all": _subset_metrics(mask_all),
        "in_domain_normal_emp": _subset_metrics(mask_in_domain) if mask_in_domain.any() else {},
        "stress": _subset_metrics(mask_stress) if mask_stress.any() else {},
    }

    if audit_noarb_samples > 0:
        n_aud = min(audit_noarb_samples, len(iv_pred))
        out["noarb_raw_surrogate"] = audit_surface_noarb(iv_pred[:n_aud], taus, k_scaled).to_dict()
        out["noarb_cos_truth"] = audit_surface_noarb(iv_true[:n_aud], taus, k_scaled).to_dict()

    return out


def train_surrogate(
    train_npz: dict[str, np.ndarray],
    val_npz: dict[str, np.ndarray],
    cfg: SurrogateConfig,
    max_train_samples: int | None = None,
    device: torch.device | None = None,
) -> tuple[GridSurrogateMLP, dict]:
    """Trains a GridSurrogateMLP (baseline or Sobolev) with reproducible seeding."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    z_min, z_max = default_z_bounds()
    x_tr, y_tr, J_tr, _ = prepare_tensors(
        train_npz, z_min, z_max, output_mode=cfg.output_mode, max_samples=max_train_samples
    )
    x_va, y_va, J_va, _ = prepare_tensors(val_npz, z_min, z_max, output_mode=cfg.output_mode)

    # Center per node, scale by a single global scalar std so MSE stays strictly proportional to RMSE
    y_mean_np = y_tr.mean(dim=0).numpy()
    y_std_val = float(y_tr.std().item())
    y_std_val = max(y_std_val, 1e-6)

    y_mean_t = torch.tensor(y_mean_np, dtype=torch.float32)
    y_tr_std = (y_tr - y_mean_t) / y_std_val
    J_tr_std = J_tr / y_std_val
    y_va_std = (y_va - y_mean_t) / y_std_val
    J_va_std = J_va / y_std_val

    taus = train_npz["taus"]
    k_scaled = train_npz["k_scaled"]
    model = GridSurrogateMLP(
        out_dim=y_tr.shape[1],
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        activation=cfg.activation,
        output_mode=cfg.output_mode,
        z_min=z_min,
        z_max=z_max,
        y_mean=y_mean_np,
        y_std=y_std_val,
        taus=taus,
        k_scaled=k_scaled,
    ).to(device)

    cal_w = (
        build_calendar_interp_weights(taus, k_scaled).to(device) if cfg.lambda_arb > 0.0 else None
    )

    # Move dataset to device once (120k * 180 * 6 * 4B ~ 518 MB — fits comfortably in unified M5 RAM)
    x_tr_d = x_tr.to(device)
    y_tr_d = y_tr_std.to(device)
    J_tr_d = J_tr_std.to(device) if cfg.lambda_grad > 0.0 else None
    x_va_d = x_va.to(device)
    y_va_d = y_va_std.to(device)
    J_va_d = J_va_std.to(device) if cfg.lambda_grad > 0.0 else None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.min_lr)

    N = len(x_tr_d)
    bs = cfg.batch_size
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for _epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(N, generator=gen).to(device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, bs):
            idx = perm[start : start + bs]
            xb = x_tr_d[idx]
            yb = y_tr_d[idx]
            opt.zero_grad(set_to_none=True)

            if cfg.lambda_grad > 0.0 and J_tr_d is not None:
                pred_y, pred_J = model.forward_std_with_jac(xb)
                loss_val = F.mse_loss(pred_y, yb)
                loss_grad = F.mse_loss(pred_J, J_tr_d[idx])
                loss = loss_val + cfg.lambda_grad * loss_grad
            else:
                pred_y = model.forward_std(xb)
                loss = F.mse_loss(pred_y, yb)

            if cfg.lambda_arb > 0.0 and cal_w is not None:
                iv_p = model.iv_from_std(pred_y)
                loss = loss + cfg.lambda_arb * noarb_penalty(
                    iv_p, model.taus, model.k_scaled, cal_w
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        sched.step()

        # Validation check
        model.eval()
        with torch.no_grad():
            if cfg.lambda_grad > 0.0 and J_va_d is not None:
                vy, vJ = model.forward_std_with_jac(x_va_d)
                vloss = float(
                    (F.mse_loss(vy, y_va_d) + cfg.lambda_grad * F.mse_loss(vJ, J_va_d)).item()
                )
            else:
                vy = model.forward_std(x_va_d)
                vloss = float(F.mse_loss(vy, y_va_d).item())

        history["train_loss"].append(epoch_loss / max(n_batches, 1))
        history["val_loss"].append(vloss)
        if vloss < best_val_loss:
            best_val_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    metrics = evaluate_surrogate(model, val_npz, device=device, audit_noarb_samples=0)
    return model, {
        "best_val_loss": best_val_loss,
        "val_metrics": metrics,
        "history_last": history["val_loss"][-1],
    }
