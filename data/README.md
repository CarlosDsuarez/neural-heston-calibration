# Data Directory Policy & SPX Schema

Per Section 5.3 of `NEURAL_HESTON_SPEC.md`:

- **No vendor market data is committed to this repository.** Almost all option data licenses prohibit redistribution.
- Everything inside `data/` is git-ignored except `data/README.md` and `data/mini_synthetic/` ($\le 1\text{ MB}$ synthetic fixture used by CI tests and `make repro-small`).
- Generated datasets (120k / 240k synthetic surfaces) and any locally stored SPX snapshots live under `data/` locally and remain untracked.

## Using Your Own Real SPX Data (`SPX_DIR`)

Place one CSV or Cboe delayed-quotes JSON file per trading day inside a local directory (e.g. `data/spx_local/`) and run:

```bash
make repro-small SPX_DIR=data/spx_local
# or full benchmark:
OMP_NUM_THREADS=1 .venv/bin/python -m nhc.bench.run --spx-dir data/spx_local
```

### Supported Formats (`python/nhc/data/spx_ingest.py`)

1. **Single-day CSV (`*.csv`):** Case-insensitive columns:
   - `quote_date` (optional; defaults to filename stem `YYYY-MM-DD`)
   - `spot` or `underlying_last`
   - `tau` (years, ACT/365) or `dte` (calendar days)
   - `strike`
   - `is_call` (`True`/`False`) or `option_type` (`C`/`P` or `call`/`put`)
   - `bid`, `ask`
   - `volume`, `open_interest` (optional; defaults to passing liquidity filter if absent)
2. **Cboe delayed-quotes JSON (`*.json`):** Standard `data.current_price` + `data.options[]` with OCC option symbols (`...YYMMDD[C/P]SSSSSSSS`), `bid`, `ask`, `volume`, and `open_interest`.

Every ingested day passes through the exact same near-ATM put-call parity OLS forward/discount extractor, bid/ask/spread/liquidity/$\tau$/OTM filters, and chronological 60/20/20 train/val/test split.
