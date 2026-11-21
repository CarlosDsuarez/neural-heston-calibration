# Phase 1 — Classical COS Pricer & Analytic Jacobian Accuracy Notes (Gate G1)

**Hardware & Compiler:** Apple M5 (10 cores, 16 GB RAM, single-threaded measurement), AppleClang 21.0.0, `-std=c++20 -O3 -fno-math-errno` (strict IEEE-754, **no** `-ffast-math`).  
**Oracle:** QuantLib 1.44 `AnalyticHestonEngine(model, 1e-12, 50000)` and PyTorch 2.15.0 `complex128` autodiff.

## 1. Implementation Notes vs. Specification Section 3

1. **Second cumulant $c_2$ (Fang–Oosterlee Table 11 erratum):** Symbolic differentiation of $\ln\phi_X(u)$ in SymPy and 50-digit verification in `mpmath` confirmed that the $\sigma^2 \bar v$ term in $c_2$ is `vbar * (-5 + 4*ek + ek^2)` (error `1.39e-17` vs `mpmath`), whereas the classic Table 11 erratum (`-7 + 6*ek + ek^2`) introduces an $O(10^{-3})$ error. For $\kappa\tau < 0.05$, we switch to a verified 4th-order Taylor expansion in $x = \kappa\tau$ to prevent $1/\kappa^3$ catastrophic cancellation.
2. **Cancellation-free $\xi - d$:** Using the exact algebraic identity $(\xi - d)/\sigma^2 = -(u^2 + iu)/(\xi + d)$ avoids subtraction cancellation both as $\sigma \to 0$ (Black–Scholes limit accurate to $< 10^{-7}$ even at $\sigma = 10^{-6}$) and near $u = 0$.
3. **Shifted-window COS with OTM / Put-Parity hybrid:** Using a per-strike shifted truncation interval $[a_m, b_m] = [x + c_1 - W_{\text{half}}, x + c_1 + W_{\text{half}}]$ of constant width $W = 2 W_{\text{half}}$ per maturity keeps $\phi_X(u_k)$ and all 5 derivatives **independent of strike** while avoiding window widening on short maturities. Direct COS calls are used only when $K \ge F$ and $b_m \le 0.5$ ($e^{b_m} \le 1.65$); otherwise we price the put ($e^{a_m} < 1$) and recover the call via put-call parity. Because $\nabla_\Theta C \equiv \nabla_\Theta P$, the analytic Jacobian always uses the put formulation and never suffers $e^{b_m}$ amplification.

## 2. Gate G1 Measurements

### 2.1 Normal Region (1,000 random cases, $\tau \in [0.022, 3.0]$, $\sigma \in [0.1, 0.8]$, 164 Feller-violated)

| Config $(N, L)$ | Max $|P_{\text{COS}} - P_{\text{QL}}| / S_0$ | p99 | p95 | Median |
|---|---:|---:|---:|---:|
| $(128, 12)$ | `6.55e-06` | `3.02e-07` | `6.13e-09` | `1.92e-15` |
| $(256, 16)$ | `2.21e-07` | `4.41e-09` | `2.05e-11` | `4.26e-16` |
| $(512, 20)$ | **`3.92e-09`** | `2.73e-11` | `2.30e-14` | `3.55e-16` |

- **Gate G1 Price Criterion ($\le 10^{-8}$):** **PASS** at $(512, 20)$ (`max = 3.92e-09`); at calibration default $(128, 12)$, median error is `1.92e-15` and p95 is `6.13e-09` (with worst Feller-violated corners at `6.55e-06`, equivalent to `< 0.002` vol points).

### 2.2 Analytic Derivatives & Price Jacobian

- **CF derivatives $\partial \ln\phi / \partial \Theta$ vs. PyTorch `complex128` autodiff** (1,000 random points, $\sigma \in [0.1, 3.5]$, $u \in [0.05, 40]$, $\tau \in [0.02, 3.0]$): max relative error = **`1.24e-10`**.
- **Price Jacobian vs. central finite differences** (30 random multi-maturity chains, 25 options each, $N=512, L=20$): max relative error **$\le 10^{-6}$** (Gate G1 **PASS**; strongly Feller-violated $7\times$ stress test verified to `6.01e-08` at $N=1024, L=28$).

### 2.3 Documented Boundary Regimes (Where Fixed $(N, L)$ Degrades)

| Regime (100 random cases each) | $(N=128, L=12)$ max / med | $(N=512, L=20)$ max / med | Diagnosis |
|---|---:|---:|---|
| Ultra-short maturity ($\tau \in [1\text{d}, 7\text{d}]$) | `9.34e-16` / `9.63e-17` | `2.31e-15` / `1.09e-16` | Shifted-window COS handles ultra-short $\tau$ to machine precision |
| Deep wings ($|k| \in [0.6, 1.2]\sqrt\tau$) | `1.40e-07` / `1.79e-14` | `7.85e-12` / `1.98e-16` | Shifted-window COS maintains full precision in wings |
| Extreme stress vol-of-vol ($\sigma \in [1.5, 4.5]$, Feller violated up to $50\times+$) | `8.59e-04` / `9.69e-06` | `1.08e-04` / `2.40e-08` | Heavy exponential tails & slow CF decay require $L \ge 24\text{--}30$, $N \ge 1024\text{--}2048$ |

## 3. Analytic vs. Finite-Difference Jacobian Speedup (Section 3.4)

Measured in pure C++ (`steady_clock`, 2,000 runs after 200 warmups, 1 core Apple M5, reused `CosWorkspace`, 200-option chain = 10 maturities × 20 strikes, $N=128, L=12$):

- `price_chain`: **p50 = 162.25 µs**, p99 = 191.17 µs
- `price_chain_with_jacobian`: **p50 = 212.00 µs**, p99 = 235.58 µs (+30.6% over pricing alone)
- **Measured speedup vs. forward FD (6×):** **4.59×**
- **Measured speedup vs. central FD (11×):** **8.42×**
