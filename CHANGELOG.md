# Changelog & Gate Verification Log

Every phase gate from `NEURAL_HESTON_SPEC.md` is recorded here with measured values, hardware info, and commit hash. No performance claim is entered without a reproducible measurement.

## [0.1.0] — 2026-11-21

### Phase 8 — Closure, Documentation & Clean-Clone Reproducibility (Gate G8)
- **Status:** PASS (verified from clean clone in `/tmp/nhc-clean-test`)
- **Deliverables:**
  - `docs/rough_viability.md`: <= 2-page technical note analyzing rBergomi / Rough Heston vs. standard Heston, stack transferability, and verified empirical SPX findings (Abi Jaber & Li, arXiv:2401.03345).
  - `README.md`: Complete English README following Section 10.4 (TL;DR, Why this project, Method, Results with measured Apple M5 tables & figures, mandatory Limitations, Reproduce, verified References & Citation).
  - `CITATION.cff`: Versioned `0.1.0` software citation metadata (MIT license).
  - `Makefile`: `setup`, `build`, `test`, `lint`, `data`, `repro-small`, `repro`, and `clean` targets (with optional `SPX_DIR=...` support).
  - `data/README.md`: Documented real SPX CSV/JSON ingestion schema for `--spx-dir`.
- **Gate G8:** Verified clean-clone `make repro-small` end-to-end in **`32.8 s`** (limit `< 30 min`, Apple M5: fresh clone -> `.venv` + C++20/nanobind build -> `nhc.bench.run --small` + `nhc.bench.report` -> reproduced `results/summary_tables.md` + 4 figures).

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

### Phase 5 — Inverse Warm-Start Ensemble, Pointwise Ablation & NHC1 `.bin` Export (Gate G5)
- **Status:** PASS (artifact: `results/phase5_gate_g5.json`, exported models: `results/models/`)
- **Deliverables:**
  - `python/nhc/train/inverse.py`: `InverseMLP` ($180 \to 128 \times 3 \to 5$, `SiLU` + `tanh` box output) supporting both `param_mse` and `surrogate_recon` (through frozen Sobolev surrogate) losses, 5-seed ensemble uncertainty, and validation-calibrated Mahalanobis + ensemble-variance OOD thresholds.
  - `python/nhc/train/pointwise.py`: 7D $\to$ 1D random-grid pointwise ablation (`64×4` `SiLU`, Baschetti et al. 2024).
  - `python/nhc/export/to_bin.py`: Versioned `NHC1` little-endian binary exporter with FNV-1a 64-bit grid hash, training config hash, git commit hash, and `.json` sidecar metadata.
  - `python/nhc/train/run_phase5.py`: Parallel training, exact COS evaluation, and export of 9 NHC1 models (`surrogate_baseline_64x4`, `surrogate_sobolev_64x4`, `surrogate_sobolev_128x4`, 5 `inverse_recon_s0..4` members, `pointwise_64x4`) + OOD calibration files.
- **Gate G5 & Hypothesis H3 Measured Values (200 held-out in-domain test surfaces, exact COS audit):**
  - *Inverse loss comparison (exact COS surface RMSE, median):* `param_mse` = **`0.088 vp`** vs `surrogate_recon` (single) = **`0.041 vp`** (2.15× better) vs `surrogate_recon` (5-seed ensemble) = **`0.034 vp`**.
  - **Hypothesis H3 (warm-start LM iteration reduction):** Cold start median = **`8.0`** iters $\to$ inverse warm-start median = **`3.0`** iters (**`62.5%` reduction**, $\ge 40\%$ -> **H3 CONFIRMED on synthetic test set**).
  - *Pointwise random-grid ablation (10k test):* in-domain RMSE = **`0.0262 vol pts`**, p99 = **`0.0883 vol pts`**.
  - *OOD calibration (99.5% validation quantile):* `mahal_threshold = 8.08`, `ens_var_threshold = 4.7e-4`.

### Phase 6 — C++20/Eigen Inference Engine, Forward-Mode Jacobian & OOD Circuit-Breaker (Gate G6)
- **Status:** PASS (artifact: `results/phase6_gate_g6.json`)
- **Deliverables:**
  - `cpp/include/nhc/mlp_eigen.hpp`: Zero-dependency Eigen 5.0.1 loader (`NHC1` with strict FNV-1a 64-bit grid-hash validation), `MlpSurrogateT<float/double>` with fused 5-tangent forward-mode Jacobian, `MlpInverseEnsembleT<float>` (5-member warm-start + disagreement variance), and `MlpPointwiseT<float>` (batched GEMM + reverse-mode Jacobian across all options simultaneously).
  - `cpp/include/nhc/ood.hpp`: `CircuitBreaker` implementing all 5 Section 8.1 OOD triggers (box margin, Mahalanobis distance, ensemble variance, COS-vs-surrogate discrepancy > 0.15 vol pts, LM failure/bounds, market support/coverage) + `project_chain_to_grid`.
  - `cpp/include/nhc/calibrate.hpp`: End-to-end C++ engines `calibrate_surrogate_lm` (E2), `calibrate_pointwise_lm` (E3), `calibrate_inverse_only` (E4), and `calibrate_hybrid` (E5), each with stage-by-stage timing breakdown and mandatory exact COS audit option.
  - `cpp/tests/test_mlp_ood.cpp` (17/17 C++ tests passing) & `tests/test_phase6_cpp_inference.py` (20/20 pytest passing).
- **Gate G6 Measured Values (Apple M5, 1 core, `OMP_NUM_THREADS=1`, 200-option off-grid chain, 1,000 C++ `steady_clock` runs):**
  - **T6.1 Parity (PyTorch vs. C++ Eigen across SiLU/Softplus/ELU/tanh, forward + Jacobian):** `<= 1e-9` abs (`float64`), `<= 1e-5` rel (`float32`) — **PASS**.
  - **OOD Circuit-Breaker:** **100% (6/6)** synthetic OOD scenarios across all 5 rules triggered fallback to exact COS+LM; `5%` fallback rate on clean in-domain chains — **PASS**.
  - **Pure-C++ Latency Summary (p50 / p99, exact COS-audited median RMSE):**
    - `E0` COS+LM cold: **`2.14 ms` / `4.72 ms`** (RMSE `1.45e-8 vp`, 7.5 iters)
    - `E1` COS+LM warm: **`1.33 ms` / `1.70 ms`** (`1.61×`, RMSE `1.05e-8 vp`, 4.0 iters)
    - `E2` Surrogate `64×4` Sobolev (no audit / + COS audit): **`0.192 ms`** (`11.2×`) / **`0.411 ms`** (`5.21×`, p99 `0.608 ms`, RMSE `0.056 vp`; breakdown: `12.7 µs` interp $M$ + `182.8 µs` neural LM + `215.8 µs` COS audit)
    - `E2` Surrogate `128×4` Sobolev (no audit / + COS audit): **`0.369 ms`** (`5.79×`) / **`0.587 ms`** (`3.65×`, p99 `0.909 ms`, RMSE **`0.046 vp`**)
    - `E3` Pointwise `64×4` (batched GEMM, no audit / + COS audit): **`3.02 ms`** (`0.71×` — **1.41× slower than E0!**) / **`3.14 ms`** (RMSE `0.0086 vp`)
    - `E4` Inverse 5-seed ensemble (no audit / + COS audit): **`0.0186 ms`** (`18.6 µs`) / **`0.235 ms`** (RMSE `0.423 vp`)
    - `E5` Hybrid (Inverse + 5 COS iters): **`1.33 ms` / `1.65 ms`** (`1.61×`, RMSE `1.05e-8 vp`)

### Phase 7 — Full Benchmark Suite, Error Budget & Reproducible Report (Gate G7)
- **Status:** PASS (artifacts: `results/benchmark_full.json`, `results/summary_tables.md`, `results/figures/fig1..4.png`)
- **Deliverables:**
  - `cpp/include/nhc/lm_solver.hpp` & `types.hpp`: Added relative cost-reduction stopping criterion (`tol_cost = 1e-7`, 2 consecutive steps) preventing LM spinning on active bounds / float32 surrogate floor while preserving 99%+ noiseless recovery.
  - `python/nhc/bench/run.py`: Single-command end-to-end benchmark executing `Ref` + `E0`–`E5` across chronological held-out test days (calm + stress), paired Wilcoxon signed-rank tests, 1,000-resample bootstrap 95% CIs, maturity/moneyness/regime slices, intertemporal parameter stability `||ΔΘ||₁` & ATM delta/vega rotation, static no-arbitrage audit, 5-component error budget decomposition (Section 7.3), and formal H1–H5 verdicts. Supports `--spx-dir` for user-supplied real SPX CSV/JSON files and `--small` for fast reproduction.
  - `python/nhc/bench/report.py`: Deterministic Markdown table & 4-figure (`fig1_latency_vs_accuracy.png`, `fig2_learning_curve.png`, `fig3_error_budget.png`, `fig4_daily_stability.png`) generator.
  - `tests/test_phase7_benchmark.py`: End-to-end `--small` smoke test for benchmark + report + figures (21/21 pytest passing).
- **Gate G7 Measured Values (Apple M5, 1 core, `OMP_NUM_THREADS=1`, `-std=c++20 -O3 -fno-math-errno`, `synthetic_bates_spx_proxy` 20 test days = 15 calm + 5 stress, 200 OTM quotes/day, all RMSEs exact COS-audited):**
  - `Ref` (QuantLib+SciPy): RMSE `0.4699 vp`, p50 **`156.67 ms`**, p99 `247.21 ms`
  - `E0` (COS+LM cold): RMSE `0.4699 vp` (calm `0.324`, stress `0.906`), p50 **`2.35 ms`** (**66.8× faster than Ref**), p99 **`4.51 ms`**, `100%` converged, 8.0 median iters
  - `E1` (COS+LM warm): RMSE `0.4699 vp`, p50 **`1.97 ms`**, p99 `3.96 ms`, 5.0 median iters
  - `E2` (Sobolev `64×4`, +COS audit): RMSE `0.4692 vp` (Wilcoxon $p=0.596$ vs E0), p50 **`0.426 ms`** (**5.51× faster than E0**, **368× faster than Ref**), p99 `1.07 ms`, raw & COS no-arb `0.00% / 0.00%`
  - `E2` (+Circuit-Breaker): RMSE `0.4700 vp`, p50 `0.681 ms`, p99 `3.52 ms`, `20.0%` fallback rate (4/5 stress days, 0% calm)
  - `E2` (Sobolev `128×4`, +COS audit): RMSE `0.4695 vp`, p50 **`0.579 ms`** (**4.06× faster than E0**), p99 `1.42 ms`
  - `E3` (Pointwise `64×4`, batched GEMM): RMSE `0.4689 vp`, p50 **`3.59 ms`** (**1.53× slower than E0**), p99 `6.13 ms`
  - `E4` (Inverse 5-seed ensemble only): RMSE `0.5185 vp`, p50 **`0.019 ms`** (`19 µs`), p99 `0.032 ms`
  - `E5` (Hybrid: Inverse + 5 COS iters): RMSE `0.4699 vp`, p50 **`1.67 ms`**, p99 `3.61 ms`
- **Error Budget (H4):** Surrogate `(a) 0.0176 vp`, Interp $M$ `(b) 0.0847 vp`, Optimizer dispersion `(c) 0.0000 vp`, Bid-ask microstructure `(e) 0.0298 vp`, Structural Heston misspecification `(d) 0.4699 vp` (**26.6×** surrogate error).
- **Hypotheses Verdicts:** **H1 `CONFIRMED`**, **H2 `CONFIRMED`**, **H3 `INCONCLUSIVE`** (62.5% iter reduction on synthetic Heston Gate G5; attenuated to 37.5% on misspecified Bates market proxy), **H4 `CONFIRMED`** (on `synthetic_bates_spx_proxy`), **H5 `CONFIRMED`**.
