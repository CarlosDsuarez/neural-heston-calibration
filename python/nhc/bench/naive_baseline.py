"""Naive literature baseline (Ref): QuantLib AnalyticHestonEngine in Python + SciPy optimizer.

This reproduces the commonly benchmarked "classical" baseline in neural calibration papers
(Python loop over QuantLib pricers + finite-difference SciPy least_squares or differential
evolution) so we can measure BOTH:
  - Ref vs E0 (how much proper C++20 COS + analytic Jacobian + LM alone gives), and
  - E0 vs E1..E5 (how much neural methods genuinely add on top of a strong classical baseline).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import QuantLib as ql
from scipy.optimize import differential_evolution, least_squares

import nhc


@dataclass
class NaiveResult:
    theta: nhc.HestonParams
    rmse_iv: float
    elapsed_ms: float
    nfev: int
    converged: bool


def _ql_price_chain(
    theta: nhc.HestonParams,
    market: nhc.Market,
    chain: list[nhc.Quote],
    eval_date: ql.Date,
) -> np.ndarray:
    ql.Settings.instance().evaluationDate = eval_date
    dc = ql.Actual365Fixed()
    rf = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, market.r, dc))
    div = ql.YieldTermStructureHandle(ql.FlatForward(eval_date, market.q, dc))
    spot = ql.QuoteHandle(ql.SimpleQuote(market.S0))
    proc = ql.HestonProcess(
        rf, div, spot, theta.v0, theta.kappa, theta.vbar, theta.sigma, theta.rho
    )
    model = ql.HestonModel(proc)
    engine = ql.AnalyticHestonEngine(model, 64)

    prices = np.empty(len(chain), dtype=np.float64)
    for i, q in enumerate(chain):
        days = max(1, int(round(q.tau * 365.0)))
        mat = eval_date + days
        opt_type = ql.Option.Call if q.is_call else ql.Option.Put
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(opt_type, q.strike), ql.EuropeanExercise(mat))
        opt.setPricingEngine(engine)
        prices[i] = opt.NPV()
    return prices


def calibrate_naive_scipy_lsq(
    market: nhc.Market,
    chain: list[nhc.Quote],
    init: nhc.HestonParams,
    cos_cfg_eval: nhc.CosConfig | None = None,
    max_nfev: int = 100,
) -> NaiveResult:
    """QuantLib Python + SciPy `least_squares` (finite-difference TRF, vega-weighted price)."""
    if cos_cfg_eval is None:
        cos_cfg_eval = nhc.CosConfig(N=128, L=12.0)

    eval_date = ql.Date(15, 6, 2026)
    mkt_prices = np.array([q.mid for q in chain], dtype=np.float64)
    weights = np.array([q.weight if q.weight > 0.0 else 1.0 for q in chain], dtype=np.float64)
    vegas = np.array(
        [
            max(
                q.vega
                if q.vega > 0.0
                else nhc.black_scholes_vega(market.S0, q.strike, market.r, market.q, q.tau, q.iv),
                1e-4 * market.S0 * 0.01,
            )
            for q in chain
        ],
        dtype=np.float64,
    )
    scale = weights / vegas

    lb = np.array([1e-4, 1e-2, 1e-4, 1e-2, -0.995], dtype=np.float64)
    ub = np.array([2.0, 20.0, 2.0, 6.0, 0.995], dtype=np.float64)
    x0 = np.clip(
        np.array([init.v0, init.kappa, init.vbar, init.sigma, init.rho], dtype=np.float64),
        lb + 1e-6,
        ub - 1e-6,
    )

    def residuals(x: np.ndarray) -> np.ndarray:
        hp = nhc.HestonParams(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))
        px = _ql_price_chain(hp, market, chain, eval_date)
        return scale * (px - mkt_prices)

    t0 = time.perf_counter()
    sol = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        method="trf",
        xtol=1e-8,
        ftol=1e-8,
        max_nfev=max_nfev,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    theta = nhc.HestonParams(
        float(sol.x[0]), float(sol.x[1]), float(sol.x[2]), float(sol.x[3]), float(sol.x[4])
    )
    rmse_iv = float(nhc.evaluate_cos_iv_rmse(theta, market, chain, cos_cfg_eval))
    return NaiveResult(
        theta=theta,
        rmse_iv=rmse_iv,
        elapsed_ms=elapsed_ms,
        nfev=int(sol.nfev),
        converged=bool(sol.success),
    )


def calibrate_naive_de(
    market: nhc.Market,
    chain: list[nhc.Quote],
    cos_cfg_eval: nhc.CosConfig | None = None,
    maxiter: int = 15,
    popsize: int = 5,
    seed: int = 42,
) -> NaiveResult:
    """QuantLib Python + SciPy `differential_evolution` (gradient-free global optimizer)."""
    if cos_cfg_eval is None:
        cos_cfg_eval = nhc.CosConfig(N=128, L=12.0)

    eval_date = ql.Date(15, 6, 2026)
    mkt_prices = np.array([q.mid for q in chain], dtype=np.float64)
    vegas = np.array(
        [
            max(
                q.vega
                if q.vega > 0.0
                else nhc.black_scholes_vega(market.S0, q.strike, market.r, market.q, q.tau, q.iv),
                1e-4 * market.S0 * 0.01,
            )
            for q in chain
        ],
        dtype=np.float64,
    )
    bounds = [(0.005, 0.5), (0.1, 10.0), (0.005, 0.5), (0.05, 3.0), (-0.98, -0.05)]

    def objective(x: np.ndarray) -> float:
        hp = nhc.HestonParams(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))
        px = _ql_price_chain(hp, market, chain, eval_date)
        r = (px - mkt_prices) / vegas
        return float(math.sqrt(np.mean(r * r)))

    t0 = time.perf_counter()
    sol = differential_evolution(
        objective, bounds=bounds, maxiter=maxiter, popsize=popsize, seed=seed, polish=False
    )
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    theta = nhc.HestonParams(
        float(sol.x[0]), float(sol.x[1]), float(sol.x[2]), float(sol.x[3]), float(sol.x[4])
    )
    rmse_iv = float(nhc.evaluate_cos_iv_rmse(theta, market, chain, cos_cfg_eval))
    return NaiveResult(
        theta=theta,
        rmse_iv=rmse_iv,
        elapsed_ms=elapsed_ms,
        nfev=int(sol.nfev),
        converged=bool(sol.success),
    )
