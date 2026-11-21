# Changelog & Gate Verification Log

Every phase gate from `NEURAL_HESTON_SPEC.md` is recorded here with measured values, hardware info, and commit hash. No performance claim is entered without a reproducible measurement.

## [Unreleased]

### Phase 0 — Foundations (Gate G0)
- **Status:** PASS (GitHub Actions CI run `44945961288` green across `gcc`, `clang`, `clang-format`, `ruff`)
- **Environment:** Apple M5 (10 cores, 16 GB RAM), AppleClang 21.0.0, CMake 4.5.1, Eigen 5.0.1, Catch2 v3.9.1, nanobind 3.1.0, Python 3.12.13, PyTorch 2.15.0, QuantLib 1.44.
- **Deliverables:**
  - Private repository `neural-heston-calibration`, MIT license, `.gitignore` (data/, build/, large binaries excluded).
  - `CMakeLists.txt` (C++20, `-O3 -fno-math-errno`, no `-ffast-math` on core targets, Eigen 5.0.1 & Catch2 v3.9.1 pinned via FetchContent, OpenMP detection, nanobind `nhc_core`).
  - `pyproject.toml` + `requirements.lock` pinned dependencies.
  - GitHub Actions CI (`gcc` + `clang` build, `ctest`, `pytest`, `clang-format`, `ruff`).
- **Gate G0:** `ctest` (2/2 passed), `pytest` (2/2 passed), `clang-format` & `ruff` clean, CI green.

### Phase 1 — Classical COS Pricer & Analytic Jacobian (Gate G1)
- **Status:** PASS (see `docs/phase1_cos_accuracy.md` for full breakdown)
- **Deliverables:**
  - `cpp/include/nhc/heston_cf.hpp`: Little Heston Trap CF, cancellation-free $(\xi-d)/\sigma^2$, verified exact $c_1, c_2$ (fixing Fang–Oosterlee Table 11 erratum) + small-$\kappa\tau$ Taylor branch, 64-byte aligned SoA `CosWorkspace`, single-pass analytic $\partial\ln\phi/\partial\Theta$.
  - `cpp/include/nhc/cos_pricer.hpp`: Shifted-window COS with OTM/put-parity hybrid and fused 6-way price+Jacobian dot product.
  - nanobind bindings (`eval_cf_and_dlogphi`, `price_option`, `price_chain`, `price_chain_with_jacobian`).
  - C++ Catch2 suite (`cpp/tests/test_cos_pricer.cpp`, 8/8 passing) and Python T1.4 oracle suite (`tests/test_phase1_pricer.py`, 6/6 passing).
- **Gate G1 Measured Values (Apple M5, AppleClang 21.0.0, `-O3 -fno-math-errno`):**
  - Price max error vs QuantLib (1,000 random normal cases, $\tau \in [0.022, 3.0]$, 164 Feller-violated): **`3.92e-09`** at $(N=512, L=20)$ ($\le 10^{-8}$ PASS); median `1.92e-15` at $(N=128, L=12)$.
  - Analytic $\partial\ln\phi/\partial\Theta$ vs PyTorch `complex128` autodiff (1,000 points): max rel error **`1.24e-10`**.
  - Price Jacobian vs central FD: max rel error **$\le 10^{-6}$** (PASS).
  - Measured Jacobian speedup (200-option chain, $N=128$, 1 core): `212 µs` analytic vs `162 µs` price-only (**4.59×** vs forward FD, **8.42×** vs central FD).
  - Documented boundary region: extreme vol-of-vol ($\sigma \in [1.5, 4.5]$, heavy Feller violation) has max `8.59e-04` at $(128, 12)$ and `1.08e-04` at $(512, 20)$; ultra-short $\tau \in [1\text{d}, 7\text{d}]$ and deep wings achieve machine precision (`9.34e-16` / `7.85e-12`).
