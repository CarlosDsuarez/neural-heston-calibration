"""Static no-arbitrage auditor (Section 3.10 of NEURAL_HESTON_SPEC.md).

Evaluates on a dense mesh (default 40 log-moneyness points per maturity):
  1. Calendar spread condition: dw(k, tau) / dtau >= 0 at fixed log-moneyness k = ln(K/F).
  2. Durrleman / Gatheral-Jacquier butterfly condition:
       g(k) = (1 - k * w'(k) / (2 w(k)))^2 - (w'(k)^2 / 4) * (1/w(k) + 1/4) + w''(k) / 2 >= 0,
     where w(k, tau) = sigma_IV(k, tau)^2 * tau.

Flags violations exceeding `tol = 1e-6` and reports exact percentages for both raw surrogate
outputs and COS-revalued surfaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class NoArbReport:
    n_surfaces: int
    n_calendar_nodes: int
    n_butterfly_nodes: int
    calendar_violation_pct: float
    butterfly_violation_pct: float
    any_violation_surface_pct: float
    worst_calendar_violation: float
    worst_butterfly_violation: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def audit_surface_noarb(
    iv_grid: np.ndarray,
    taus: np.ndarray,
    k_scaled: np.ndarray,
    n_dense_k: int = 40,
    tol: float = 1e-6,
) -> NoArbReport:
    """Audits one surface `(J, I)` or a batch `(N, J, I)` (or `(N, J*I)`) of IV surfaces."""
    n_tau = len(taus)
    n_k = len(k_scaled)
    arr = np.asarray(iv_grid, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, n_tau, n_k)
    elif arr.ndim == 2:
        arr = (
            arr.reshape(arr.shape[0], n_tau, n_k)
            if arr.shape[1] == n_tau * n_k
            else arr.reshape(1, n_tau, n_k)
        )

    N = arr.shape[0]
    sqrt_taus = np.sqrt(taus)

    total_cal_nodes = 0
    total_cal_viol = 0
    total_bfly_nodes = 0
    total_bfly_viol = 0
    surfaces_with_viol = 0
    worst_cal = 0.0
    worst_bfly = 0.0

    # Precompute dense evaluation grids per slice (strictly interior to avoid boundary spline artefacts)
    u_dense = np.linspace(float(k_scaled[0]) * 0.98, float(k_scaled[-1]) * 0.98, n_dense_k)

    for s in range(N):
        surf_has_viol = False
        splines: list[CubicSpline] = []
        for j in range(n_tau):
            k_knots = k_scaled * sqrt_taus[j]
            w_knots = (arr[s, j, :] ** 2) * taus[j]
            cs = CubicSpline(k_knots, w_knots, bc_type="natural")
            splines.append(cs)

            # Butterfly Durrleman condition on dense k mesh for slice j
            k_eval = u_dense * sqrt_taus[j]
            w = np.maximum(cs(k_eval), 1e-12)
            wp = cs(k_eval, 1)
            wpp = cs(k_eval, 2)
            g = (
                (1.0 - (k_eval * wp) / (2.0 * w)) ** 2
                - 0.25 * (wp**2) * (1.0 / w + 0.25)
                + 0.5 * wpp
            )
            viol_mask = g < -tol
            n_v = int(viol_mask.sum())
            total_bfly_nodes += n_dense_k
            total_bfly_viol += n_v
            if n_v > 0:
                surf_has_viol = True
                worst_bfly = min(worst_bfly, float(np.min(g)))

        # Calendar monotonicity at fixed k between consecutive maturities j and j+1
        # Every k in slice j's range lies strictly inside slice j+1's range since sqrt(tau_j) < sqrt(tau_{j+1})
        for j in range(n_tau - 1):
            k_eval = u_dense * sqrt_taus[j]
            w_lo = splines[j](k_eval)
            w_hi = splines[j + 1](k_eval)
            dw = w_hi - w_lo
            viol_mask = dw < -tol
            n_v = int(viol_mask.sum())
            total_cal_nodes += n_dense_k
            total_cal_viol += n_v
            if n_v > 0:
                surf_has_viol = True
                worst_cal = min(worst_cal, float(np.min(dw)))

        if surf_has_viol:
            surfaces_with_viol += 1

    return NoArbReport(
        n_surfaces=N,
        n_calendar_nodes=total_cal_nodes,
        n_butterfly_nodes=total_bfly_nodes,
        calendar_violation_pct=100.0 * total_cal_viol / max(total_cal_nodes, 1),
        butterfly_violation_pct=100.0 * total_bfly_viol / max(total_bfly_nodes, 1),
        any_violation_surface_pct=100.0 * surfaces_with_viol / max(N, 1),
        worst_calendar_violation=worst_cal,
        worst_butterfly_violation=worst_bfly,
    )
