.PHONY: help setup build test lint data repro-small repro clean

export PATH := /opt/homebrew/bin:/opt/homebrew/sbin:$(PATH)

VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
BUILD_CPP := build-cpp
CMAKE_GEN := $(shell command -v ninja >/dev/null 2>&1 && echo "-G Ninja" || echo "")
SPX_DIR ?=
SPX_ARGS := $(if $(SPX_DIR),--spx-dir $(SPX_DIR),)

help:
	@echo "Targets:"
	@echo "  make setup        Create .venv, install Python deps, build C++20 nanobind extension"
	@echo "  make build        Build C++ tests/CLI (build-cpp) and editable nhc_core extension"
	@echo "  make test         Run C++ Catch2 (ctest) and Python (pytest) test suites"
	@echo "  make lint         Run clang-format --dry-run, ruff check, and ruff format --check"
	@echo "  make repro-small  Fast Gate G8 reproduction (< 5 min): builds stack & reproduces summary table + figures"
	@echo "  make repro        Full end-to-end reproduction (240k data -> train -> export -> full benchmark)"
	@echo "                    Optional: make repro SPX_DIR=/path/to/spx_csvs"
	@echo "  make clean        Remove build artifacts (preserves .venv and committed results/)"

setup:
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			uv venv --python 3.12 $(VENV) || uv venv $(VENV); \
		else \
			python3 -m venv $(VENV); \
		fi; \
	fi
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install --python $(VENV_PYTHON) scikit-build-core nanobind==3.1.0; \
		uv pip install --python $(VENV_PYTHON) --no-build-isolation -e ".[dev]"; \
	else \
		$(VENV_PYTHON) -m pip install --upgrade pip; \
		$(VENV_PYTHON) -m pip install scikit-build-core nanobind==3.1.0; \
		$(VENV_PYTHON) -m pip install --no-build-isolation -e ".[dev]"; \
	fi

build: setup
	cmake -S . -B $(BUILD_CPP) $(CMAKE_GEN) \
		-DCMAKE_BUILD_TYPE=Release \
		-DNHC_BUILD_PYTHON=OFF \
		-DNHC_BUILD_TESTS=ON \
		-DNHC_BUILD_TOOLS=ON
	cmake --build $(BUILD_CPP) --parallel

test: build
	ctest --test-dir $(BUILD_CPP) --output-on-failure
	$(VENV_PYTHON) -m pytest -v

lint:
	find cpp -type f \( -name '*.hpp' -o -name '*.cpp' \) -print0 | xargs -0 clang-format --dry-run --Werror
	$(VENV)/bin/ruff check python tests
	$(VENV)/bin/ruff format --check python tests

data: build
	$(VENV_PYTHON) -m nhc.data.synthetic
	$(VENV_PYTHON) -c "from pathlib import Path; \
from nhc.data.spx_ingest import generate_misspecified_spx_proxy_series, split_snapshots_temporal; \
from nhc.data.synthetic import calibrate_empirical_anchors, generate_dataset; \
snaps = generate_misspecified_spx_proxy_series(n_days=60, seed=20261121); \
tr, _, _ = split_snapshots_temporal(snaps); \
anchors = calibrate_empirical_anchors(tr); \
generate_dataset(240_000, Path('data/synthetic_240k'), seed=20261124, empirical_anchors=anchors, compressed=False, include_jacobians=True)"

repro-small: setup
	@echo "==> [Gate G8] Running reproducible small benchmark using committed NHC1 models..."
	OMP_NUM_THREADS=1 $(VENV_PYTHON) -m nhc.bench.run --small $(SPX_ARGS) --out results/benchmark_small.json
	$(VENV_PYTHON) -m nhc.bench.report \
		--bench results/benchmark_small.json \
		--phase4 results/phase4_gate_g4.json \
		--md-out results/summary_tables.md \
		--fig-dir results/figures
	@echo "==> Reproduced results/summary_tables.md and results/figures/:"
	@cat results/summary_tables.md

repro: data
	@echo "==> [Full Repro] Training Phase 4 surrogates (Baseline + Sobolev + 5-seed learning curve)..."
	$(VENV_PYTHON) -m nhc.train.run_phase4
	@echo "==> [Full Repro] Training Phase 5 inverse ensemble, pointwise ablation & exporting NHC1 models..."
	$(VENV_PYTHON) -m nhc.train.run_phase5
	@echo "==> [Full Repro] Running Phase 7 full benchmark (Ref + E0-E5) & generating report..."
	OMP_NUM_THREADS=1 $(VENV_PYTHON) -m nhc.bench.run $(SPX_ARGS) --out results/benchmark_full.json
	$(VENV_PYTHON) -m nhc.bench.report \
		--bench results/benchmark_full.json \
		--phase4 results/phase4_gate_g4.json \
		--md-out results/summary_tables.md \
		--fig-dir results/figures

clean:
	rm -rf $(BUILD_CPP) build .pytest_cache .ruff_cache
