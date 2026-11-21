"""SPX European option chain ingestion, cleaning, put-call parity forward/discount extraction,
and misspecified Bates jump-diffusion SPX proxy generator (clearly tagged as synthetic proxy
when real vendor SPX CSVs are not present on disk).

Implements Section 5.2 of NEURAL_HESTON_SPEC.md:
  1. Implicit forward F and discount factor D per maturity via near-ATM put-call parity OLS:
       C - P = a - D * K  =>  D = -slope, F = a / D.
  2. Filters: bid > 0, ask > bid, relative spread <= 20%, liquidity (volume or OI), tau >= 7/365.
  3. OTM only (put if K < F, call if K >= F).
  4. Weights: inverse IV spread w_i = 1 / max(spread_iv_i, 0.001) (or vega-weighted).
  5. Transparent per-rule discard audit counts.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import nhc


@dataclass
class FilterConfig:
    max_rel_spread: float = 0.20
    min_tau: float = 7.0 / 365.0
    max_tau: float = 2.5
    min_volume_or_oi: int = 1
    iv_spread_floor: float = 0.001
    weight_mode: str = "inv_iv_spread"  # "inv_iv_spread" or "vega"
    pcp_moneyness_band: float = 0.12  # near-ATM band |K/S0 - 1| for put-call parity regression


@dataclass
class FilterAudit:
    total_raw: int = 0
    discarded_tau: int = 0
    discarded_bid_ask: int = 0
    discarded_spread: int = 0
    discarded_liquidity: int = 0
    discarded_itm: int = 0
    discarded_iv: int = 0
    kept_otm: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class SpxDaySnapshot:
    date: str
    spot: float
    market: nhc.Market
    chain: list[nhc.Quote]
    audit: FilterAudit
    source_type: str  # "real_spx_csv", "real_cboe_json", or "synthetic_bates_spx_proxy"
    regime: str = "unknown"  # "calm", "elevated", "stress", or "market"
    per_maturity_pcp: dict[float, tuple[float, float]] | None = None  # tau -> (F, D)


def extract_forward_discount_pcp(
    df_mat: pd.DataFrame,
    spot: float,
    tau: float,
    moneyness_band: float = 0.12,
    fallback_r: float = 0.04,
    fallback_q: float = 0.013,
) -> tuple[float, float, float, float]:
    """Extracts implied discount factor D and forward F via near-ATM put-call parity OLS:
        C(K) - P(K) = a - D * K  =>  D = -slope, F = a / D.
    Returns (F, D, r_implied, q_implied).
    """
    calls = df_mat[df_mat["is_call"]].copy()
    puts = df_mat[~df_mat["is_call"]].copy()
    if len(calls) == 0 or len(puts) == 0:
        D = math.exp(-fallback_r * tau)
        F = spot * math.exp((fallback_r - fallback_q) * tau)
        return F, D, fallback_r, fallback_q

    calls["mid"] = 0.5 * (calls["bid"] + calls["ask"])
    puts["mid"] = 0.5 * (puts["bid"] + puts["ask"])
    merged = pd.merge(
        calls[["strike", "mid"]],
        puts[["strike", "mid"]],
        on="strike",
        suffixes=("_c", "_p"),
    )
    merged = merged[
        (merged["strike"] >= spot * (1.0 - moneyness_band))
        & (merged["strike"] <= spot * (1.0 + moneyness_band))
    ]

    if len(merged) < 3:
        D = math.exp(-fallback_r * tau)
        F = spot * math.exp((fallback_r - fallback_q) * tau)
        return F, D, fallback_r, fallback_q

    K = merged["strike"].to_numpy(dtype=np.float64)
    y = (merged["mid_c"] - merged["mid_p"]).to_numpy(dtype=np.float64)
    # OLS: y = a + slope * K
    X = np.column_stack([np.ones_like(K), K])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, slope = float(coef[0]), float(coef[1])
    D = -slope
    if not (0.5 < D < 1.05) or a <= 0.0:
        D = math.exp(-fallback_r * tau)
        F = spot * math.exp((fallback_r - fallback_q) * tau)
        return F, D, fallback_r, fallback_q

    F = a / D
    r_imp = -math.log(D) / tau
    q_imp = r_imp - math.log(F / spot) / tau
    return F, D, r_imp, q_imp


def ingest_spx_dataframe(
    df: pd.DataFrame,
    date: str,
    spot: float,
    cfg: FilterConfig | None = None,
    source_type: str = "real_spx_csv",
    regime: str = "market",
) -> SpxDaySnapshot:
    """Cleans a raw SPX option chain DataFrame into a calibrated-ready `SpxDaySnapshot`.

    Required columns in `df`:
      `tau` (years), `strike`, `is_call` (bool), `bid`, `ask`, `volume`, `open_interest`.
    """
    if cfg is None:
        cfg = FilterConfig()

    audit = FilterAudit(total_raw=len(df))
    work = df.copy()

    # 1. Maturity filter
    mask_tau = (work["tau"] >= cfg.min_tau) & (work["tau"] <= cfg.max_tau)
    audit.discarded_tau = int((~mask_tau).sum())
    work = work[mask_tau]

    # 2. Positive bid & ask > bid
    mask_ba = (work["bid"] > 0.0) & (work["ask"] > work["bid"])
    audit.discarded_bid_ask = int((~mask_ba).sum())
    work = work[mask_ba]

    # 3. Relative spread filter <= 20%
    mid = 0.5 * (work["bid"] + work["ask"])
    rel_spread = (work["ask"] - work["bid"]) / mid
    mask_sp = rel_spread <= cfg.max_rel_spread
    audit.discarded_spread = int((~mask_sp).sum())
    work = work[mask_sp].copy()
    work["mid"] = 0.5 * (work["bid"] + work["ask"])

    # 4. Minimum liquidity filter (volume or open interest)
    mask_liq = (work["volume"] >= cfg.min_volume_or_oi) | (
        work["open_interest"] >= cfg.min_volume_or_oi
    )
    audit.discarded_liquidity = int((~mask_liq).sum())
    work = work[mask_liq]

    # 5. Extract per-maturity F and D via put-call parity OLS & estimate flat term (r, q)
    per_mat: dict[float, tuple[float, float]] = {}
    r_vals: list[float] = []
    q_vals: list[float] = []
    mat_groups = sorted(work["tau"].unique())
    for tau in mat_groups:
        sub = work[work["tau"] == tau]
        F, D, r_i, q_i = extract_forward_discount_pcp(
            sub, spot, float(tau), moneyness_band=cfg.pcp_moneyness_band
        )
        per_mat[float(tau)] = (F, D)
        r_vals.append(r_i)
        q_vals.append(q_i)

    r_med = float(np.median(r_vals)) if r_vals else 0.04
    q_med = float(np.median(q_vals)) if q_vals else 0.013
    market = nhc.Market(spot, r_med, q_med)

    # 6. OTM filter + IV inversion + weights
    quotes: list[nhc.Quote] = []
    raw_weights: list[float] = []
    for tau in mat_groups:
        F_mat, _ = per_mat[float(tau)]
        sub = work[work["tau"] == tau].sort_values("strike")
        for row in sub.itertuples(index=False):
            strike = float(row.strike)
            is_call = bool(row.is_call)
            is_otm = (is_call and strike >= F_mat) or ((not is_call) and strike < F_mat)
            if not is_otm:
                audit.discarded_itm += 1
                continue

            px_mid = float(row.mid)
            iv_mid = nhc.implied_volatility(px_mid, spot, strike, r_med, q_med, float(tau), is_call)
            if not math.isfinite(iv_mid) or iv_mid <= 1e-4 or iv_mid > 4.0:
                audit.discarded_iv += 1
                continue

            vega = nhc.black_scholes_vega(spot, strike, r_med, q_med, float(tau), iv_mid)
            if not math.isfinite(vega) or vega <= 1e-6:
                audit.discarded_iv += 1
                continue

            iv_bid = nhc.implied_volatility(
                float(row.bid), spot, strike, r_med, q_med, float(tau), is_call
            )
            iv_ask = nhc.implied_volatility(
                float(row.ask), spot, strike, r_med, q_med, float(tau), is_call
            )
            if math.isfinite(iv_bid) and math.isfinite(iv_ask) and iv_ask >= iv_bid:
                iv_spread = max(iv_ask - iv_bid, cfg.iv_spread_floor)
            else:
                iv_spread = max(
                    (float(row.ask) - float(row.bid)) / max(vega, 1e-4), cfg.iv_spread_floor
                )

            w = (1.0 / iv_spread) if cfg.weight_mode == "inv_iv_spread" else vega
            raw_weights.append(w)
            quotes.append(
                nhc.Quote(
                    tau=float(tau),
                    strike=strike,
                    mid=px_mid,
                    iv=float(iv_mid),
                    vega=float(vega),
                    weight=float(w),
                    is_call=is_call,
                )
            )

    # Normalize weights to RMS = 1 so LM residual scale matches IV units cleanly
    if quotes:
        w_arr = np.array(raw_weights, dtype=np.float64)
        w_rms = float(np.sqrt(np.mean(w_arr * w_arr)))
        if w_rms > 0.0:
            for q in quotes:
                q.weight = q.weight / w_rms

    audit.kept_otm = len(quotes)
    return SpxDaySnapshot(
        date=date,
        spot=spot,
        market=market,
        chain=quotes,
        audit=audit,
        source_type=source_type,
        regime=regime,
        per_maturity_pcp=per_mat,
    )


def load_cboe_delayed_json(path: Path, cfg: FilterConfig | None = None) -> SpxDaySnapshot:
    """Parses a locally saved Cboe delayed quotes JSON file (e.g. `_SPX.json`) into SpxDaySnapshot."""
    payload = json.loads(path.read_text())
    data = payload.get("data", payload)
    spot = float(data["current_price"])
    ts = str(payload.get("timestamp", data.get("last_trade_time", path.stem)))[:10]
    eval_dt = pd.Timestamp(ts)

    rows = []
    for opt in data.get("options", []):
        sym = str(opt.get("option", ""))
        # Standard OCC suffix: ...YYMMDD[C/P]SSSSSSSS
        if len(sym) < 15:
            continue
        tail = sym[-15:]
        exp_str = "20" + tail[:6]
        cp = tail[6]
        strike = int(tail[7:]) / 1000.0
        exp_dt = pd.Timestamp(exp_str)
        tau = (exp_dt - eval_dt).days / 365.0
        rows.append(
            {
                "tau": tau,
                "strike": strike,
                "is_call": cp == "C",
                "bid": float(opt.get("bid", 0.0) or 0.0),
                "ask": float(opt.get("ask", 0.0) or 0.0),
                "volume": int(opt.get("volume", 0) or 0),
                "open_interest": int(opt.get("open_interest", 0) or 0),
            }
        )
    df = pd.DataFrame(rows)
    return ingest_spx_dataframe(df, date=ts, spot=spot, cfg=cfg, source_type="real_cboe_json")


def load_spx_csv(path: Path, cfg: FilterConfig | None = None) -> SpxDaySnapshot:
    """Loads a cleaned or OptionsDX/Cboe-style single-day SPX CSV file."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    date = str(df[cols["quote_date"]].iloc[0])[:10] if "quote_date" in cols else path.stem
    spot = (
        float(df[cols["underlying_last"]].iloc[0])
        if "underlying_last" in cols
        else float(df[cols["spot"]].iloc[0])
    )
    norm = pd.DataFrame(
        {
            "tau": df[cols["tau"]].astype(float)
            if "tau" in cols
            else df[cols["dte"]].astype(float) / 365.0,
            "strike": df[cols["strike"]].astype(float),
            "is_call": df[cols["is_call"]].astype(bool)
            if "is_call" in cols
            else df[cols["option_type"]].astype(str).str.upper().str.startswith("C"),
            "bid": df[cols["bid"]].astype(float),
            "ask": df[cols["ask"]].astype(float),
            "volume": df[cols["volume"]].fillna(0).astype(int) if "volume" in cols else 10,
            "open_interest": df[cols["open_interest"]].fillna(0).astype(int)
            if "open_interest" in cols
            else 100,
        }
    )
    return ingest_spx_dataframe(norm, date=date, spot=spot, cfg=cfg, source_type="real_spx_csv")


def _bates_cos_prices(
    S0: float,
    r: float,
    q: float,
    tau: float,
    strikes: np.ndarray,
    is_call: np.ndarray,
    v0: float,
    kappa: float,
    vbar: float,
    sigma: float,
    rho: float,
    lam_j: float,
    mu_j: float,
    sigma_j: float,
    N: int = 256,
    L: float = 16.0,
) -> np.ndarray:
    """Bates (1996) jump-diffusion pricer via COS (Heston CF * Merton jump CF).

    Used ONLY to generate a genuinely misspecified non-Heston SPX market proxy
    (incorporating short-dated jump skew that 1-factor Heston cannot exactly replicate)
    so the Phase 7 error budget (H4) measures real structural misspecification even when
    vendor SPX CSVs are unavailable.
    """
    k_bar = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    drift = (r - q - lam_j * k_bar) * tau

    # Variance estimate including jump contribution for COS window
    v_eff = max(v0, vbar)
    c2_approx = v_eff * tau + lam_j * tau * (mu_j * mu_j + sigma_j * sigma_j)
    c1_approx = (r - q) * tau - 0.5 * v_eff * tau
    w_half = L * math.sqrt(max(c2_approx, 1e-8))
    W = 2.0 * w_half
    du = math.pi / W
    u = np.arange(N, dtype=np.float64) * du

    iu = 1j * u
    s2 = sigma * sigma
    xi = kappa - rho * sigma * iu
    d = np.sqrt(xi * xi + s2 * (u * u + iu))
    d = np.where(d.real < 0, -d, d)
    beta = xi + d
    alpha_over_s2 = -(u * u + iu) / beta
    g = (s2 * alpha_over_s2) / beta
    E = np.exp(-d * tau)
    D = 1.0 - g * E
    log_phi_heston = (
        iu * drift
        + (kappa * vbar) * (alpha_over_s2 * tau - 2.0 * np.log(D / (1.0 - g)) / s2)
        + v0 * alpha_over_s2 * (1.0 - E) / D
    )
    log_phi_jump = lam_j * tau * (np.exp(iu * mu_j - 0.5 * sigma_j * sigma_j * u * u) - 1.0)
    log_phi = log_phi_heston + log_phi_jump
    log_phi[0] = 0.0
    phi = np.exp(log_phi)

    a0 = c1_approx - w_half
    A = (2.0 / W) * np.real(phi * np.exp(-1j * u * a0))
    A0 = A[0]
    u1 = u[1:]
    C = A[1:] / (1.0 + u1 * u1)
    S = C / u1
    I_put = float(np.sum(C))
    ks = np.arange(1, N, dtype=np.float64)

    df_r = math.exp(-r * tau)
    df_q = math.exp(-q * tau)
    S_disc = S0 * df_q

    out = np.empty(len(strikes), dtype=np.float64)
    for idx, (K, ic) in enumerate(zip(strikes, is_call, strict=True)):
        x = math.log(S0 / float(K))
        am = x + c1_approx - w_half
        bm = x + c1_approx + w_half
        if am >= 0.0:
            put_p = 0.0
        elif bm <= 0.0:
            put_p = max(float(K) * df_r - S_disc, 0.0)
        else:
            theta = -math.pi * am / W
            sig = float(np.sum(S * np.sin(ks * theta) - C * np.cos(ks * theta)))
            put_p = (
                df_r
                * float(K)
                * (0.5 * A0 * (-am - 1.0 + math.exp(am)) + sig + math.exp(am) * I_put)
            )
        call_p = put_p + S_disc - float(K) * df_r
        out[idx] = max(call_p if ic else put_p, 0.0)
    return out


def generate_misspecified_spx_proxy_series(
    n_days: int = 60,
    seed: int = 20261121,
    cfg: FilterConfig | None = None,
) -> list[SpxDaySnapshot]:
    """Generates a multi-day, multi-regime SPX-like European option series using Bates
    jump-diffusion (genuinely misspecified vs 1-factor Heston) + realistic bid-ask spreads,
    passes every day through the EXACT `ingest_spx_dataframe` pipeline (put-call parity OLS,
    bid-ask/liquidity/OTM filters, discard audit), and explicitly tags every snapshot as
    `source_type='synthetic_bates_spx_proxy'`.

    Regimes across the chronological series (for zero-leakage train/val/test split):
      - First 60% of days (train): calm + moderate volatility evolution (VIX ~ 13-22)
      - Next 20% of days (val):    moderate / elevated transition (VIX ~ 16-26)
      - Final 20% of days (test):  includes a multi-day volatility stress shock (VIX ~ 30-48)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_days)

    # Underlying evolving state (OU-like daily evolution with stress shock in test window)
    spot = 5800.0
    r = 0.043
    q = 0.013
    v0 = 0.022  # ~14.8% vol
    vbar = 0.036
    kappa = 2.4
    sigma = 0.65
    rho = -0.78

    # Maturity calendar (10 expiries: 10d to 1.8y, plus 1 ultra-short 4d expiry to exercise tau filter)
    dte_list = [4, 12, 24, 45, 75, 115, 175, 260, 380, 540, 680]
    n_train_end = int(0.60 * n_days)
    n_val_end = int(0.80 * n_days)

    snapshots: list[SpxDaySnapshot] = []
    for t_idx, dt in enumerate(dates):
        if t_idx < n_train_end:
            regime = "calm"
            target_v0, target_sig, lam_j, mu_j, sig_j = 0.026, 0.70, 0.45, -0.11, 0.09
        elif t_idx < n_val_end:
            regime = "elevated"
            target_v0, target_sig, lam_j, mu_j, sig_j = 0.048, 0.95, 0.75, -0.13, 0.11
        else:
            # Test period includes a realistic ~25% multi-day volatility spike episode
            n_test_days = max(1, n_days - n_val_end)
            offset = t_idx - n_val_end
            shock_start = max(1, int(0.45 * n_test_days))
            shock_end = shock_start + max(2, int(0.25 * n_test_days))
            in_shock = shock_start <= offset < shock_end
            if in_shock:
                regime = "stress"
                target_v0, target_sig, lam_j, mu_j, sig_j = 0.16, 1.45, 1.35, -0.14, 0.12
            else:
                regime = "calm"
                target_v0, target_sig, lam_j, mu_j, sig_j = 0.032, 0.75, 0.55, -0.11, 0.09

        # Smooth day-to-day parameter evolution (tests intertemporal stability realistically)
        v0 = float(np.clip(0.82 * v0 + 0.18 * target_v0 + rng.normal(0.0, 0.004), 0.009, 0.45))
        vbar = float(
            np.clip(
                0.90 * vbar + 0.10 * max(0.032, 0.65 * target_v0) + rng.normal(0.0, 0.002),
                0.015,
                0.30,
            )
        )
        kappa = float(np.clip(kappa + rng.normal(0.0, 0.08), 0.8, 5.5))
        sigma = float(np.clip(0.85 * sigma + 0.15 * target_sig + rng.normal(0.0, 0.04), 0.30, 2.2))
        rho = float(np.clip(rho + rng.normal(0.0, 0.015), -0.92, -0.55))
        spot = float(
            spot
            * math.exp(rng.normal(-0.003 if regime == "stress" else 0.0003, math.sqrt(v0 / 252.0)))
        )

        raw_rows = []
        for dte in dte_list:
            tau = dte / 365.0
            F = spot * math.exp((r - q) * tau)
            # 20 strikes per expiry, with both calls and puts on near-ATM strikes for PCP OLS
            u_grid = np.linspace(-0.48, 0.30, 20)
            strikes = np.round(F * np.exp(u_grid * math.sqrt(tau)) / 5.0) * 5.0
            strikes = np.unique(strikes)

            # Evaluate both calls and puts at every strike
            both_K = np.concatenate([strikes, strikes])
            both_ic = np.concatenate(
                [np.ones(len(strikes), dtype=bool), np.zeros(len(strikes), dtype=bool)]
            )
            true_px = _bates_cos_prices(
                spot, r, q, tau, both_K, both_ic, v0, kappa, vbar, sigma, rho, lam_j, mu_j, sig_j
            )

            for K, ic, px in zip(both_K, both_ic, true_px, strict=True):
                if px < 0.15:
                    # Deep OTM illiquid quote -> exercises bid=0 / wide spread filters
                    bid = 0.0
                    ask = max(0.10, float(px) * 1.5)
                    vol_i, oi_i = 0, 5
                else:
                    # Realistic SPX bid-ask spread (~0.4% ATM, widening in wings & stress)
                    moneyness_pen = abs(math.log(float(K) / F)) / max(math.sqrt(tau), 0.15)
                    rel_half_sp = (0.003 + 0.012 * moneyness_pen) * (
                        1.6 if regime == "stress" else 1.0
                    )
                    # Add tiny microstructure noise to mid (~15% of half-spread)
                    noisy_mid = float(px) * (1.0 + float(rng.normal(0.0, 0.15 * rel_half_sp)))
                    bid = max(0.05, noisy_mid * (1.0 - rel_half_sp))
                    ask = noisy_mid * (1.0 + rel_half_sp)
                    vol_i = int(rng.integers(5, 800))
                    oi_i = int(rng.integers(20, 5000))

                raw_rows.append(
                    {
                        "tau": tau,
                        "strike": float(K),
                        "is_call": bool(ic),
                        "bid": bid,
                        "ask": ask,
                        "volume": vol_i,
                        "open_interest": oi_i,
                    }
                )

        snap = ingest_spx_dataframe(
            pd.DataFrame(raw_rows),
            date=dt.strftime("%Y-%m-%d"),
            spot=spot,
            cfg=cfg,
            source_type="synthetic_bates_spx_proxy",
            regime=regime,
        )
        snapshots.append(snap)

    return snapshots


def split_snapshots_temporal(
    snapshots: list[SpxDaySnapshot],
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[list[SpxDaySnapshot], list[SpxDaySnapshot], list[SpxDaySnapshot]]:
    """Strict chronological train < val < test partition (zero temporal leakage)."""
    ordered = sorted(snapshots, key=lambda s: s.date)
    n = len(ordered)
    i_train = max(1, int(round(train_frac * n)))
    i_val = max(i_train + 1, int(round((train_frac + val_frac) * n)))
    return ordered[:i_train], ordered[i_train:i_val], ordered[i_val:]


def calibrate_empirical_anchors(
    train_snapshots: list[SpxDaySnapshot],
    cos_cfg: nhc.CosConfig | None = None,
) -> np.ndarray:
    """Calibrates ONLY training-date snapshots with COS+LM to produce empirical Theta anchors
    for Stratum 3 (10% of the synthetic dataset, Section 5.1).
    """
    if cos_cfg is None:
        cos_cfg = nhc.CosConfig(N=128, L=12.0)
    lm_cfg = nhc.LmConfig()
    lm_cfg.max_iter = 60
    init = nhc.HestonParams(0.04, 2.0, 0.04, 0.6, -0.75)

    anchors = []
    for snap in train_snapshots:
        res = nhc.calibrate_cos_lm(snap.market, snap.chain, init, lm_cfg, cos_cfg)
        if res.converged and math.isfinite(res.rmse_iv):
            th = res.theta
            anchors.append([th.v0, th.kappa, th.vbar, th.sigma, th.rho])
            init = th  # warm-start next chronological training day
    return np.array(anchors, dtype=np.float64)
