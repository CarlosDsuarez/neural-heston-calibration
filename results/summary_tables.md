# Benchmark Results Summary (Phase 7 — Gate G7)

## Hardware, Compiler & Data Stamp
- **CPU / OS:** `Apple M5` (arm64, Darwin 27.2.0, single-threaded `OMP_NUM_THREADS=1`)
- **Compiler Flags:** `-std=c++20 -O3 -fno-math-errno (strict IEEE-754, no -ffast-math)` · Eigen `5.0.1` · Python `3.12.13`
- **Git Commit:** `cd27553` · Generated UTC: `2026-11-21T22:11:57.934240+00:00`
- **Market Series Source:** **`synthetic_bates_spx_proxy`** (train=60, val=20, test=20 days, mean 200.0 OTM quotes/day)

## 1. Main Benchmark Table (Section 7.5 — All Final RMSEs Audited with Exact COS)

| Engine | RMSE IV (vol pts, mean) | 95% CI | p50 (ms) | p99 (ms) | Conv. (no bounds) | Iter. LM (med) | Fallback | No-Arb (bfly/cal %) |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Ref (QuantLib+SciPy) | `0.4699` | `[0.370, 0.583]` | `156.665` | `247.207` | `100.0%` | `9.0` | `0.0%` | COS-audited |
| E0 COS+LM cold | `0.4699` | `[0.370, 0.583]` | `2.347` | `4.507` | `100.0%` | `8.0` | `0.0%` | COS 0.00% / 0.00% |
| E1 COS+LM warm | `0.4699` | `[0.370, 0.583]` | `1.973` | `3.962` | `100.0%` | `5.0` | `0.0%` | COS-audited |
| E2 Surrogate 64×4 (Sobolev) | `0.4692` | `[0.372, 0.577]` | `0.426` | `1.074` | `100.0%` | `6.5` | `0.0%` | raw 0.00% / 0.00% · COS 0.00% / 0.00% |
| E2 Surrogate 64×4 (+Circuit-Breaker) | `0.4700` | `[0.372, 0.579]` | `0.681` | `3.516` | `100.0%` | `7.5` | `20.0%` | COS-audited |
| E2 Surrogate 64×4 (Baseline) | `0.4693` | `[0.373, 0.576]` | `0.410` | `0.968` | `100.0%` | `6.0` | `0.0%` | COS-audited |
| E2 Surrogate 128×4 (Sobolev) | `0.4695` | `[0.372, 0.578]` | `0.579` | `1.418` | `100.0%` | `6.0` | `0.0%` | COS-audited |
| E3 Pointwise 64×4 (Random Grid) | `0.4689` | `[0.369, 0.581]` | `3.595` | `6.125` | `100.0%` | `6.5` | `0.0%` | COS-audited |
| E4 Inverse 5-Seed Ensemble only | `0.5185` | `[0.418, 0.635]` | `0.019` | `0.032` | `100.0%` | `0.0` | `0.0%` | COS-audited |
| E5 Hybrid (Inverse + 5 COS iters) | `0.4699` | `[0.370, 0.583]` | `1.667` | `3.608` | `65.0%` | `5.0` | `0.0%` | COS-audited |

## 2. Accuracy Sliced by Maturity, Moneyness & Regime (Mean Exact COS RMSE, vol pts)

| Engine | < 1M | 1M–6M | > 6M | Put Wing | ATM | Call Wing | Calm Days | Stress Days | ||ΔΘ||₁ (med) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ref (QuantLib+SciPy) | `0.669` | `0.502` | `0.271` | `0.460` | `0.514` | `0.417` | `0.324` | `0.906` | `0.724` |
| E0 COS+LM cold | `0.669` | `0.502` | `0.271` | `0.460` | `0.514` | `0.416` | `0.324` | `0.906` | `0.724` |
| E1 COS+LM warm | `0.669` | `0.502` | `0.271` | `0.460` | `0.514` | `0.416` | `0.324` | `0.906` | `0.724` |
| E2 Surrogate 64×4 (Sobolev) | `0.697` | `0.480` | `0.269` | `0.435` | `0.523` | `0.444` | `0.329` | `0.890` | `0.841` |
| E2 Surrogate 64×4 (+Circuit-Breaker) | `0.697` | `0.481` | `0.269` | `0.440` | `0.521` | `0.441` | `0.328` | `0.896` | `0.841` |
| E2 Surrogate 64×4 (Baseline) | `0.698` | `0.479` | `0.271` | `0.433` | `0.524` | `0.447` | `0.330` | `0.887` | `0.823` |
| E2 Surrogate 128×4 (Sobolev) | `0.696` | `0.483` | `0.265` | `0.435` | `0.522` | `0.445` | `0.329` | `0.891` | `0.859` |
| E3 Pointwise 64×4 (Random Grid) | `0.668` | `0.498` | `0.277` | `0.455` | `0.519` | `0.417` | `0.324` | `0.903` | `0.724` |
| E4 Inverse 5-Seed Ensemble only | `0.686` | `0.544` | `0.365` | `0.526` | `0.532` | `0.483` | `0.371` | `0.960` | `0.619` |
| E5 Hybrid (Inverse + 5 COS iters) | `0.669` | `0.502` | `0.271` | `0.460` | `0.514` | `0.416` | `0.324` | `0.906` | `0.724` |

## 3. Error Budget Decomposition (Section 7.3 — Hypothesis H4)

| Component | Measured RMSE (vol pts) | Notes |
|---|---:|---|
| (a) Surrogate approximation (Sobolev 64×4, in-box days) | `0.0176` | Against exact COS on 180-node grid (20 in-box days; all days incl. OOB extrapolation: `0.0176` vp) |
| (b) Linear interpolation operator $M$ | `0.0847` | Cubic spline in $k$ + linear in total variance |
| (c) Optimizer / multi-start dispersion (10 starts) | `0.0000` | Std dev of RMSE across 10 dispersed cold starts |
| (d) Structural Heston vs. Market (`synthetic_bates_spx_proxy`) | **`0.4699`** | Dominant component (**26.6×** surrogate error) |
| (e) Quote bid-ask / microstructure noise (est.) | `0.0298` | Estimated via high-frequency 2nd-difference MAD |

## 4. Paired Daily Wilcoxon Signed-Rank Tests vs. E0

| Comparison (Engine − E0) | Days ($n$) | Median ΔRMSE (vol pts) | Wilcoxon $p$-value |
|---|---:|---:|---:|
| `Ref_naive` − `E0` | 20 | `+0.0000` | `3.623e-02` |
| `E1_cos_warm` − `E0` | 20 | `+0.0000` | `6.215e-01` |
| `E2_sobolev_64x4` − `E0` | 20 | `+0.0029` | `5.958e-01` |
| `E2_sobolev_64x4_safe` − `E0` | 20 | `+0.0016` | `2.343e-01` |
| `E2_baseline_64x4` − `E0` | 20 | `+0.0043` | `5.958e-01` |
| `E2_sobolev_128x4` − `E0` | 20 | `+0.0027` | `5.958e-01` |
| `E3_pointwise_64x4` − `E0` | 20 | `-0.0005` | `1.690e-03` |
| `E4_inverse_ens5` − `E0` | 20 | `+0.0427` | `3.815e-06` |
| `E5_hybrid_5cos` − `E0` | 20 | `+0.0000` | `7.562e-01` |

## 5. Falsifiable Hypotheses Verdicts (H1–H5)

| ID | Hypothesis | Measured Evidence | Verdict |
|---|---|---|---|
| **H1** | COS + analytic Jacobian + LM (1 core, ~200 options): p99 < 10 ms | p50=2.35 ms, p99=4.51 ms (synthetic Gate G2 p99=8.30 ms) | **`CONFIRMED`** |
| **H2** | Grid surrogate + LM (Eigen float32) >= 3x faster than E0 (1 core) with in-domain IV RMSE <= 0.05 vol pts | 64x4 (+COS audit): 5.51x faster (0.426 vs 2.35 ms), in-domain RMSE=0.0420 vp; 128x4 (+COS audit): 4.06x faster (0.579 ms), in-domain RMSE=0.0359 vp | **`CONFIRMED`** |
| **H3** | Inverse warm-start reduces LM iterations >= 40% (median) vs cold start | Market proxy E1 (same LM config): 8.0 -> 5.0 iters (37.5% reduction); E5 hybrid (5-iter cap): 5.0 iters; Synthetic Heston test (Gate G5): 8.0 -> 3.0 iters (62.5% reduction) | **`INCONCLUSIVE`** |
| **H4** | On SPX/misspecified market, Heston structural error dominates surrogate error by >= 10x (RMSE) | Structural Heston RMSE=0.470 vp vs Surrogate RMSE=0.0176 vp (ratio=26.6x; note: measured on synthetic_bates_spx_proxy) | **`CONFIRMED`** |
| **H5** | Pointwise random-grid surrogate (E3) is slower than exact COS (E0) in standard Heston | E3 p50=3.59 ms vs E0 p50=2.35 ms (1.53x slower even with batched Eigen GEMM + reverse-mode Jacobian; Phase 6 synthetic ratio=1.41x) | **`CONFIRMED`** |
