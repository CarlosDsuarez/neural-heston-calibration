# Viability Note: Transferring the Neural Calibration Stack to Rough Volatility (rBergomi & Rough Heston)

> **Scope:** Concise architectural and empirical note (Phase 8, T8.1) explaining why neural surrogates are an *optimization* in standard Markovian Heston, become an *enabling technology* in rough volatility models, how our C++20/Eigen stack transfers, and what empirical SPX evidence says about rough vs. Markovian specifications.

---

## 1. What Changes Between Standard Heston and Rough Volatility?

In standard Markovian Heston (1993), the log-price characteristic function is exponential-affine and reduces to an ordinary Riccati differential equation with a closed-form elementary solution. Combined with the Fourier–cosine expansion (Fang & Oosterlee, 2009) and single-pass analytic gradients (Cui, del Baño Rollin & Germano, 2017), our C++20 engine (`E0`) evaluates a full 200-option chain plus its exact 5-parameter Jacobian in **0.21 ms** and calibrates end-to-end in **2.35 ms p50 / 4.51 ms p99** on a single Apple M5 core (Gate G1 & Gate G7).

Rough volatility models replace Brownian motion driving variance with fractional Brownian motion of Hurst exponent $H \in (0, 1/2)$ to capture steep power-law short-dated ATM skews ($\propto \tau^{H - 1/2}$). That mathematical change destroys the classical pricing shortcut:

| Model | Pricing Mechanism | Typical Classical Single-Surface Eval | Analytic Parameter Jacobian |
|---|---|---|---|
| **Standard Heston** (5 params) | Closed-form affine CF + COS expansion | **~0.21 ms** (measured, C++20 1 core) | Exact single-pass ($\partial\ln\phi/\partial\Theta$) |
| **Rough Heston** (6 params) | Fractional Volterra Riccati integral equation ($\alpha = H + \tfrac{1}{2}$); no elementary closed form — requires Adams discretization or $N$-factor Markovian lift | **~5–50 ms** (order-of-magnitude slower depending on quadrature / factor count) | Expensive (differentiating Volterra solver or lifted ODEs) |
| **rBergomi** (4 params) | No tractable characteristic function — requires hybrid-scheme / Volterra Monte Carlo ($10^4$–$10^5$ paths) | **~50 ms – several seconds** (CPU; variance-reduced MC) | Noisy finite differences or pathwise / AAD through MC paths |

---

## 2. Why a Neural Surrogate *Is* Necessary in Rough Models

In standard Heston, our benchmark shows that:
1. A strong classical baseline (`E0`) already achieves **2.35 ms p50** (66.8× faster than a naive QuantLib+SciPy loop).
2. A grid surrogate (`E2`, Sobolev 64×4) delivers a useful **5.51×** further speedup (**0.426 ms** including exact COS audit), whereas a pointwise random-grid surrogate (`E3`, **3.59 ms**) is actually **1.53× slower** than exact COS (Hypothesis H5 confirmed).

In **rBergomi** and **Rough Heston**, the latency economics invert completely (Bayer & Stemper, 2018; Bayer et al., 2019; Horvath, Muguruza & Tomas, 2021):
- Running 8–15 Levenberg–Marquardt iterations with Monte Carlo pricing takes **seconds to minutes per snapshot** and suffers from Monte Carlo objective roughness that traps gradient-based optimizers.
- Replacing the inner pricer with a smooth grid surrogate turns calibration into a **sub-millisecond smooth deterministic optimization problem** (a $10^2\times$–$10^4\times$ speedup over classical MC calibration) while amortizing the expensive MC simulation strictly offline during dataset generation.
- Even **pointwise** random-grid surrogates (Baschetti, Bormetti & Rossi, 2024) become net-positive once exact pricing exceeds ~5–10 ms per surface.

---

## 3. How Our C++20/Eigen Stack Transfers Directly

Most of the engineering infrastructure built in this repository is **model-agnostic** and carries over to rough volatility with minimal modification:

1. **Grid & Linear Interpolation Operator (`cpp/include/nhc/grid.hpp`):** `AdaptiveGrid` ($\tau$-scaled log-moneyness $k/\sqrt{\tau}$) and `InterpOperator` $M$ (natural cubic spline in $k$ + linear in total variance) depend only on quoted market strikes/maturities, not on Heston. Only the short-maturity grid resolution ($\tau \in [1\text{d}, 1\text{M}]$) should be densified to resolve steep rough skews.
2. **Inference & Forward-Mode Jacobian (`cpp/include/nhc/mlp_eigen.hpp`):** `MlpSurrogateT<float>` evaluates the full 180-node surface and its exact $180 \times d$ Jacobian in one forward pass of $d$ tangents (`~15 µs` per pass for $d=5$). Moving to rBergomi ($d=4$) or Rough Heston ($d=6$) changes only the input dimension of the versioned `NHC1` binary format.
3. **Sobolev Training (`python/nhc/train/surrogate.py`):** In Rough Heston, exact gradients from the lifted multi-factor Riccati ODE or adjoint Adams solver supply Sobolev labels directly; in rBergomi, pathwise / AAD sensitivities from Monte Carlo provide unbiased gradient labels (reducing required simulation budgets via differential ML).
4. **OOD Circuit-Breaker (`cpp/include/nhc/ood.hpp`):** All 5 rules (parameter box margin, Mahalanobis distance on input IV surface, inverse-ensemble variance, LM convergence/bounds, and market support coverage) transfer unchanged. The only adaptation is Rule 3 (exact re-pricing audit): when exact pricing is expensive Monte Carlo, synchronous per-tick audit is replaced by either asynchronous/sampled MC verification or fallback to a fast multi-factor Markovian approximation.

---

## 4. Empirical Caveat on SPX: Do Rough Models Actually Win?

Before investing in rough-volatility dataset generation and neural deployment for SPX production desks, the empirical literature raises an important caution:

- **Abi Jaber & Li (2024/25, arXiv:2401.03345)** compare rough, path-dependent, and Markovian stochastic volatility models empirically on SPX option surfaces and find that **rough volatility models do not consistently outperform their Markovian / multi-factor equivalents on short-dated maturities**.
- In practice, multi-factor Markovian models (e.g., 2-factor Heston / lifted Bergomi / Markovian stochastic volatility with jumps) can replicate the empirical short-end SPX term structure of skew while preserving **fast semi-analytic or low-dimensional PDE/ODE pricing**, simpler hedging dynamics, and straightforward exact online fallback without Monte Carlo noise.

---

## 5. Engineering Decision Matrix

| Regime / Desk Requirement | Recommended Architecture | Rationale |
|---|---|---|
| **Standard Heston** (real-time single surface, `< 5 ms` budget) | **`E0` / `E1` Classical C++20 COS+LM** | `2.0–2.4 ms` p50, zero approximation error, zero OOD risk, exact no-arbitrage by construction. |
| **Standard Heston** (high-frequency batch / scenario grid, `< 0.5 ms` budget) | **`E2` Sobolev Grid Surrogate + Circuit-Breaker** | `0.43 ms` p50 (incl. exact COS audit), `5.5×` speedup over `E0`, automatic fallback to `E0` during stress regimes. |
| **SPX Production Desk** (requiring `< 0.2 vp` fit across short-dated smiles) | **Multi-factor Markovian SV / SVJ (before rough)** | Addresses the dominant structural Heston misspecification (H4) without sacrificing analytic tractability (Abi Jaber & Li, 2024/25). |
| **Rough Volatility (rBergomi / Rough Heston)** | **Sobolev Grid Surrogate + Inverse Warm-Start + Circuit-Breaker** | Classical MC/Volterra calibration is too slow for interactive use; neural surrogate is genuinely required. |

---

## References (Verified)

- Abi Jaber, E., & Li, S. (2024/25). *Volatility models in practice: Rough, path-dependent or Markovian?* arXiv:2401.03345.
- Baschetti, F., Bormetti, G., & Rossi, P. (2024). *Deep calibration with random grids.* Quantitative Finance 24(9), 1263–1285. arXiv:2306.11061.
- Bayer, C., & Stemper, B. (2018). *Deep calibration of rough stochastic volatility models.* arXiv:1810.03399.
- Bayer, C., Horvath, B., Muguruza, A., Stemper, B., & Tomas, M. (2019). *On deep calibration of (rough) stochastic volatility models.* arXiv:1908.08806.
- Cui, Y., del Baño Rollin, S., & Germano, G. (2017). *Full and fast calibration of the Heston stochastic volatility model.* European Journal of Operational Research 263(2), 625–638. arXiv:1511.08718.
- Fang, F., & Oosterlee, C. W. (2009). *A novel pricing method for European options based on Fourier-cosine series expansions.* SIAM Journal on Scientific Computing 31(2), 826–848.
- Horvath, B., Muguruza, A., & Tomas, M. (2021). *Deep learning volatility.* Quantitative Finance 21(1), 11–27. arXiv:1901.09647.
