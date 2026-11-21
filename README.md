# neural-heston-calibration

Heston calibration in C++20/Python: analytic-gradient COS + Levenberg–Marquardt baseline vs. Sobolev-trained neural surrogate & inverse warm-start, with an honest benchmark on SPX.

> **Status:** Early development (Phase 0 — Foundations). No benchmark numbers are reported yet. See [`NEURAL_HESTON_SPEC.md`](NEURAL_HESTON_SPEC.md) for the full specification and [`CHANGELOG.md`](CHANGELOG.md) for gate verification history.

## Build & Test

```bash
# C++ unit tests
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

# Python editable install & tests
uv venv --python 3.12
uv pip install -e ".[dev]" --no-build-isolation
pytest
```
