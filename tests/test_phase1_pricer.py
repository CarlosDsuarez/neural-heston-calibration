"""Phase 1 verification suite (T1.4 & Gate G1):
- T1.4a: 1,000 random cases vs QuantLib AnalyticHestonEngine (including Feller violated, tau in [0.02, 3]).
- T1.4b: COS convergence N = 64 -> 128 -> 256 -> 512.
- T1.4c: Analytic d(ln phi)/dTheta vs PyTorch complex128 autodiff on 1,000 random points (Sec. 3.4).
- T1.4d: Price Jacobian vs central finite differences (Gate G1 <= 1e-6 relative).
"""

from __future__ import annotations

import math

import numpy as np
import QuantLib as ql
import torch

import nhc


def _ql_price(
    S0: float,
    r: float,
    q: float,
    days: int,
    K: float,
    is_call: bool,
    p: nhc.HestonParams,
) -> tuple[float, float]:
    today = ql.Date(15, 6, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    mat = today + days
    tau_exact = dc.yearFraction(today, mat)
    proc = ql.HestonProcess(
        ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
        ql.QuoteHandle(ql.SimpleQuote(S0)),
        p.v0,
        p.kappa,
        p.vbar,
        p.sigma,
        p.rho,
    )
    opt_type = ql.Option.Call if is_call else ql.Option.Put
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(opt_type, K), ql.EuropeanExercise(mat))
    opt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc), 1e-12, 50000))
    return float(opt.NPV()), float(tau_exact)


def test_t14a_quantlib_oracle_1000_random_cases() -> None:
    """Gate G1: max normalized price error <= 1e-8 on 1,000 random normal cases
    (tau in [0.02, 3.0], including Feller-violated parameter sets).
    """
    rng = np.random.default_rng(20261121)
    cfg = nhc.CosConfig(N=512, L=20.0, use_c4=False)
    errs = np.empty(1000, dtype=np.float64)
    feller_viol_count = 0

    for i in range(1000):
        S0 = 100.0
        r = float(rng.uniform(0.0, 0.05))
        q = float(rng.uniform(0.0, 0.03))
        days = int(rng.integers(8, 1096))  # [0.0219, 3.000] years
        v0 = float(rng.uniform(0.01, 0.25))
        vbar = float(rng.uniform(0.01, 0.25))
        kappa = float(rng.uniform(0.5, 8.0))
        sigma = float(rng.uniform(0.1, 0.8))
        rho = float(rng.uniform(-0.95, -0.15))

        p = nhc.HestonParams(v0=v0, kappa=kappa, vbar=vbar, sigma=sigma, rho=rho)
        if not p.satisfies_feller():
            feller_viol_count += 1

        tau_approx = days / 365.0
        F = S0 * math.exp((r - q) * tau_approx)
        k_lm = float(rng.uniform(-0.40 * math.sqrt(tau_approx), 0.30 * math.sqrt(tau_approx)))
        K = F * math.exp(k_lm)
        is_call = bool(rng.integers(0, 2))

        ql_p, tau_exact = _ql_price(S0, r, q, days, K, is_call, p)
        cos_p = nhc.price_option(p, nhc.Market(S0, r, q), tau_exact, K, is_call, cfg)
        errs[i] = abs(cos_p - ql_p) / S0

    assert feller_viol_count >= 100, "Test must cover Feller-violated cases"
    assert float(np.max(errs)) <= 1e-8, (
        f"Gate G1 violated: max err/S0 = {np.max(errs):.3e}, p99 = {np.percentile(errs, 99):.3e}"
    )


def test_t14b_cos_convergence_N() -> None:
    """COS convergence N = 64 -> 128 -> 256 -> 512 vs QuantLib reference."""
    S0, r, q = 100.0, 0.03, 0.01
    p = nhc.HestonParams(v0=0.05, kappa=1.8, vbar=0.04, sigma=0.6, rho=-0.75)
    market = nhc.Market(S0, r, q)

    cases = [
        (15, 95.0, False),
        (15, 100.0, True),
        (90, 90.0, False),
        (90, 110.0, True),
        (365, 80.0, False),
        (365, 120.0, True),
        (730, 100.0, True),
    ]
    ql_refs = []
    chain = []
    for days, K, is_call in cases:
        qp, tau_ex = _ql_price(S0, r, q, days, K, is_call, p)
        ql_refs.append(qp)
        chain.append(nhc.Quote(tau=tau_ex, strike=K, is_call=is_call))
    ql_refs_arr = np.array(ql_refs)

    errors_by_N: dict[int, float] = {}
    for N in (64, 128, 256, 512):
        prices = nhc.price_chain(p, market, chain, nhc.CosConfig(N=N, L=20.0))
        errors_by_N[N] = float(np.max(np.abs(prices - ql_refs_arr) / S0))

    assert errors_by_N[128] <= errors_by_N[64] + 1e-12
    assert errors_by_N[256] <= errors_by_N[128] + 1e-12
    assert errors_by_N[512] <= 1e-10


def test_t14c_cf_derivatives_vs_pytorch_complex128_autodiff() -> None:
    """Section 3.4 mandatory validation: d(ln phi)/dTheta vs PyTorch complex128 autodiff
    across 1,000 random parameter/frequency/maturity points (including extreme Feller violations).
    """
    rng = np.random.default_rng(777)
    max_rel_err = 0.0

    for _ in range(1000):
        u = float(rng.uniform(0.05, 40.0))
        tau = float(rng.uniform(0.02, 3.0))
        drift = float(rng.uniform(-0.02, 0.06)) * tau
        v0 = float(rng.uniform(0.005, 0.50))
        kappa = float(rng.uniform(0.1, 10.0))
        vbar = float(rng.uniform(0.005, 0.50))
        sigma = float(rng.uniform(0.1, 3.5))
        rho = float(rng.uniform(-0.98, -0.05))

        theta_t = torch.tensor(
            [v0, kappa, vbar, sigma, rho], dtype=torch.float64, requires_grad=True
        )
        v0_t, kappa_t, vbar_t, sigma_t, rho_t = theta_t.unbind()
        iu = torch.complex(
            torch.tensor(0.0, dtype=torch.float64), torch.tensor(u, dtype=torch.float64)
        )
        s2 = sigma_t * sigma_t
        xi = kappa_t - rho_t * sigma_t * iu
        d = torch.sqrt(xi * xi + s2 * (u * u + iu))
        if d.real.item() < 0.0:
            d = -d
        alpha = xi - d
        g = alpha / (xi + d)
        E = torch.exp(-d * tau)
        D = 1.0 - g * E
        log_phi = (
            iu * drift
            + (kappa_t * vbar_t / s2) * (alpha * tau - 2.0 * torch.log(D / (1.0 - g)))
            + (v0_t / s2) * alpha * (1.0 - E) / D
        )
        grad_re = torch.autograd.grad(log_phi.real, theta_t, retain_graph=True)[0]
        grad_im = torch.autograd.grad(log_phi.imag, theta_t)[0]
        ad_grad = grad_re.detach().numpy() + 1j * grad_im.detach().numpy()

        p = nhc.HestonParams(v0=v0, kappa=kappa, vbar=vbar, sigma=sigma, rho=rho)
        phi_cpp, dlogphi_cpp = nhc.eval_cf_and_dlogphi(u, tau, drift, p)
        cpp_grad = np.array(dlogphi_cpp, dtype=np.complex128)

        phi_torch = complex(torch.exp(log_phi).detach().item())
        assert abs(phi_cpp - phi_torch) <= 1e-12 * max(1.0, abs(phi_torch))

        rel = np.abs(cpp_grad - ad_grad) / np.maximum(np.abs(ad_grad), 1e-10)
        max_rel_err = max(max_rel_err, float(np.max(rel)))

    assert max_rel_err <= 1e-9, f"CF autodiff max relative error too large: {max_rel_err:.3e}"


def test_t14d_price_jacobian_vs_finite_differences() -> None:
    """Gate G1: analytic price Jacobian relative error <= 1e-6 vs central finite differences."""
    rng = np.random.default_rng(2026)
    market = nhc.Market(100.0, 0.03, 0.01)
    cfg = nhc.CosConfig(N=512, L=20.0, use_c4=False)

    taus = [0.05, 0.25, 0.75, 1.5, 2.5]
    chain: list[nhc.Quote] = []
    for tau in taus:
        F = market.S0 * math.exp((market.r - market.q) * tau)
        for lm in (-0.35, -0.15, 0.0, 0.15, 0.25):
            K = F * math.exp(lm * math.sqrt(tau))
            chain.append(nhc.Quote(tau=tau, strike=K, is_call=(lm >= 0.0)))

    max_rel_err = 0.0
    for _ in range(30):
        p = nhc.HestonParams(
            v0=float(rng.uniform(0.01, 0.25)),
            kappa=float(rng.uniform(0.5, 6.0)),
            vbar=float(rng.uniform(0.01, 0.25)),
            sigma=float(rng.uniform(0.15, 0.8)),
            rho=float(rng.uniform(-0.95, -0.15)),
        )
        prices, J = nhc.price_chain_with_jacobian(p, market, chain, cfg)
        assert prices.shape == (len(chain),)
        assert J.shape == (len(chain), 5)
        assert np.all(np.isfinite(prices))
        assert np.all(np.isfinite(J))

        fields = ["v0", "kappa", "vbar", "sigma", "rho"]
        for col, name in enumerate(fields):
            val = getattr(p, name)
            h = 5e-5 * max(0.1, abs(val))
            pp = nhc.HestonParams(p.v0, p.kappa, p.vbar, p.sigma, p.rho)
            pm = nhc.HestonParams(p.v0, p.kappa, p.vbar, p.sigma, p.rho)
            setattr(pp, name, val + h)
            setattr(pm, name, val - h)
            pr_p = nhc.price_chain(pp, market, chain, cfg)
            pr_m = nhc.price_chain(pm, market, chain, cfg)
            fd = (pr_p - pr_m) / (2.0 * h)
            denom = np.maximum(np.abs(fd), 1e-4 * market.S0)
            rel = float(np.max(np.abs(J[:, col] - fd) / denom))
            max_rel_err = max(max_rel_err, rel)

    assert max_rel_err <= 1e-6, f"Gate G1 Jacobian violated: max_rel_err={max_rel_err:.3e}"
