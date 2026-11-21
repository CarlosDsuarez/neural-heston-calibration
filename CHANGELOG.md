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

### Phase 3 — Data Pipeline, Adaptive Grid & Parallel OpenMP Generator (Gate G3)
- **Status:** PASS (artifact: `results/phase3_gate_g3.json`)
- **Deliverables:**
  - `cpp/include/nhc/grid.hpp`: `AdaptiveGrid` (10 maturities $\in [0.025, 2.0]$ × 18 scaled log-moneyness nodes $\in [-0.55, +0.35]$ = 180 nodes) + fixed linear interpolation operator $M$ (`InterpOperator`, natural cubic spline in $k$ + linear in total variance between adjacent maturities, coverage accounting, Jacobian propagation).
  - `python/nhc/data/spx_ingest.py`: SPX ingestion pipeline (near-ATM put-call parity OLS forward/discount extraction, bid/ask/spread/liquidity/tau/OTM filters, inverse-IV-spread weights, exact per-rule discard audit, chronological train/val/test split, and clearly tagged `synthetic_bates_spx_proxy` Bates jump-diffusion fallback when vendor CSVs are absent).
  - `cpp/src/gen_dataset.cpp` (`nhc_gen_dataset`): Parallel OpenMP (`#pragma omp parallel for schedule(static)`) synthetic surface + analytic IV Jacobian generator with automatic high-res COS retry on extreme stress regimes.
  - `python/nhc/data/synthetic.py`: Stratified Sobol sampler (65% normal / 25% stress / 10% empirical-perturbed from training-only COS+LM anchors), manifest writer, and committed `< 1 MB` fixture `data/mini_synthetic/`.
- **Gate G3 Measured Values (Apple M5, 10 OpenMP threads, AppleClang 21.0.0 `-O3 -fno-math-errno`):**
  - **120,000 valid surfaces** (180 nodes × 5 analytic IV Jacobian columns each = 129.6M labels) generated in **`15.52 s`** (**`8,196.7 surfaces/s`**; valid rate `95.88%`).
  - **NaN/Inf count:** **`0`** (PASS).
  - **Max IV round-trip error across all 21.6M nodes:** **`1.20e-14`** ($\le 10^{-8}$ PASS).
  - Separate 10k validation and 10k test datasets generated locally (gitignored); `data/mini_synthetic/dataset.npz` (~800 KiB, $\le 1$ MB) committed for CI & `make repro-small`.

### Phase 4 — Neural Surrogate (Baseline vs. Sobolev, Learning Curve & Architecture Sweep, Gate G4)
- **Status:** PARTIAL PASS — RMSE & p99 goals PASSED; Sobolev Jacobian reduction measured at `1.70×` (`64×4`) / `2.23×` (`128×4`) vs `3.0×` target and reported honestly (artifact: `results/phase4_gate_g4.json`)
- **Deliverables:**
  - `python/nhc/eval/noarb.py`: Dense 40-point-per-slice static no-arbitrage auditor (calendar monotonicity + Durrleman butterfly, tol `1e-6`).
  - `python/nhc/train/surrogate.py`: `GridSurrogateMLP` with smooth activations (`SiLU`, `Softplus`, `ELU`, `tanh`), analytic forward-mode 5-tangent Jacobian (verified to `< 1e-12` vs `torch.autograd.functional.jacobian`), Sobolev loss, optional differentiable no-arb penalty, and total-variance ablation.
  - `python/nhc/train/run_phase4.py`: Full multi-core Phase 4 experiment runner (68 trained models across T4.2, T4.3, T4.4, and final champions on 240k surfaces).
- **Gate G4 Measured Values (Apple M5, PyTorch 2.15.0 CPU, 10k held-out test set):**
  - **Champion `sobolev_64x4`** (24,564 params, $\lambda_{\text{grad}}=1.0$, 240k): in-domain RMSE = **`0.0420 vol pts`** ($\le 0.05$ PASS), p99 = **`0.1315 vol pts`** ($\le 0.20$ PASS), full-domain RMSE = `0.0459 vol pts`, median Jacobian rel error = **`0.93%`** vs `1.59%` baseline (**`1.70×`** reduction).
  - **Champion `sobolev_128x4`** (73,524 params, $\lambda_{\text{grad}}=1.0$, 240k): in-domain RMSE = **`0.0359 vol pts`** ($\le 0.05$ PASS), p99 = **`0.1064 vol pts`** ($\le 0.20$ PASS), full-domain RMSE = `0.0419 vol pts`, median Jacobian rel error = **`0.71%`** (**`2.23×`** reduction vs `baseline_64x4`).
  - **Key Empirical Findings:**
    - *Sobolev vs. explicit no-arb penalty (T4.2, 120k):* Sobolev ($\lambda_{\text{grad}}=1.0$) cut RMSE from `0.1267` to `0.0780 vp` AND reduced raw butterfly violations 5× (`0.94%` $\to$ `0.19%`), outperforming explicit $\lambda_{\text{arb}}=1.0$ (`0.1813 vp`, `1.65%` bfly).
    - *Total-variance output ablation (T4.2):* `0.4763 vp` RMSE vs `0.0838 vp` for direct $\sigma_{IV}$ (5.7× worse due to short-maturity variance scaling).
    - *Architecture sweep (T4.4, 120k):* `SiLU` (`0.0780 vp`) beat `ELU` (`0.0912`), `tanh` (`0.0972`), and `Softplus` (`0.1059`); width `128` reached `0.0594 vp` even at 120k / 80 epochs.
    - *5-seed learning curve (T4.3, 10k..240k):* Sobolev delivered ~4× sample efficiency (`30k` Sobolev `0.0986 ± 0.0042 vp` beats `120k` Baseline `0.1473 ± 0.0054 vp`) and consistent `~1.9–2.0×` Jacobian error reduction across all sample sizes.
