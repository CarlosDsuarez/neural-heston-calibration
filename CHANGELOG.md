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

### Phase 2 — End-to-End Classical Calibration & Benchmark v0 (Gate G2)
- **Status:** PASS (artifact: `results/phase2_gate_g2.json`)
- **Deliverables:**
  - `cpp/include/nhc/iv_solver.hpp`: Jäckel-style normalized OTM Black-76 solver (`erfcx` log-price lower branch, cancellation-free upper tail, safeguarded Halley's method); round-trip IV -> price -> IV error $\le 10^{-10}$ (down to $10^{-100} S_0$ OTM).
  - `cpp/include/nhc/lm_solver.hpp`: Generic 5×5 Eigen `LDLT` Levenberg–Marquardt with scale-invariant Marquardt diagonal, Nielsen smooth damping, fused first-trial Jacobian evaluation, `LogArtanh` and `SqrtDrift` parameterizations.
  - `cpp/include/nhc/calibrate.hpp`: `calibrate_cos_lm` supporting `VegaWeightedPrice` and `ImpliedVol` residual modes.
  - `python/nhc/bench/naive_baseline.py`: Naive literature baseline (`Ref`: QuantLib Python + SciPy `least_squares` and `differential_evolution`).
  - `python/nhc/bench/run_phase2.py`: Benchmark harness v0 (`steady_clock` in C++, 1 core, warmup, p50/p95/p99, hardware stamp).
- **Gate G2 & Hypothesis H1 Measured Values (Apple M5, 1 core, `OMP_NUM_THREADS=1`, AppleClang 21.0.0 `-O3 -fno-math-errno`, 200-option chain, $N=128, L=12$):**
  - **Gate G2 (100 noiseless surfaces, fixed cold start):** **99.0% recovery** ($\le 0.005$ vol pts, PASS $\ge 99\%$); 100% converged; 0% hit bounds; median RMSE IV = `1.19e-08` vol pts; p95 = `3.15e-07` vol pts; median LM iterations = `8.0`.
  - **Hypothesis H1 (`E0` COS+LM, 1,000 runs across 100 chains):** **p50 = `2.83 ms`**, **p95 = `5.67 ms`**, **p99 = `8.30 ms`** (`< 10 ms` -> **H1 CONFIRMED on Apple M5**).
  - **Naive Baseline (`Ref`, QuantLib Python + SciPy `least_squares`):** p50 = `244.84 ms` (**E0 is 86.7× faster** than Ref before any neural network).
