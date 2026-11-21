# Changelog & Gate Verification Log

Every phase gate from `NEURAL_HESTON_SPEC.md` is recorded here with measured values, hardware info, and commit hash. No performance claim is entered without a reproducible measurement.

## [Unreleased]

### Phase 0 — Foundations (Gate G0)
- **Status:** Verified locally & submitted to CI
- **Environment:** Apple M5 (10 cores, 16 GB RAM), AppleClang 21.0.0, CMake 4.5.1, Eigen 5.0.1, Catch2 v3.9.1, nanobind 3.1.0, Python 3.12.13, PyTorch 2.15.0, QuantLib 1.44.
- **Deliverables:**
  - Private repository `neural-heston-calibration`, MIT license, `.gitignore` (data/, build/, large binaries excluded).
  - `CMakeLists.txt` (C++20, `-O3 -fno-math-errno`, no `-ffast-math` on core targets, Eigen 5.0.1 & Catch2 v3.9.1 pinned via FetchContent, OpenMP detection, nanobind `nhc_core`).
  - `pyproject.toml` + `requirements.lock` pinned dependencies.
  - GitHub Actions CI (`gcc` + `clang` build, `ctest`, `pytest`, `clang-format`, `ruff`).
- **Gate G0:** `ctest` (2/2 passed), `pytest` (2/2 passed), `clang-format` & `ruff` clean.
