#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include "nhc/cos_pricer.hpp"
#include "nhc/heston_cf.hpp"
#include "nhc/types.hpp"

TEST_CASE("AlignedAllocator produces 64-byte aligned pointers", "[heston_cf]") {
  nhc::CosWorkspace ws;
  ws.ensure_size(256, true);
  REQUIRE(reinterpret_cast<std::uintptr_t>(ws.u.data()) % 64 == 0);
  REQUIRE(reinterpret_cast<std::uintptr_t>(ws.C.data()) % 64 == 0);
  REQUIRE(reinterpret_cast<std::uintptr_t>(ws.S.data()) % 64 == 0);
  REQUIRE(reinterpret_cast<std::uintptr_t>(ws.dC[0].data()) % 64 == 0);
  REQUIRE(reinterpret_cast<std::uintptr_t>(ws.dS[4].data()) % 64 == 0);
}

TEST_CASE("Heston CF obeys phi(0)=1, |phi(u)|<=1, and Black-Scholes limit", "[heston_cf]") {
  const nhc::HestonParams p{0.04, 2.0, 0.04, 0.5, -0.7};
  std::complex<double> phi;
  std::array<std::complex<double>, 5> dlogphi;

  nhc::eval_cf_and_dlogphi(0.0, 1.0, 0.02, p, phi, dlogphi);
  REQUIRE(std::abs(phi - std::complex<double>(1.0, 0.0)) < 1e-15);
  for (int j = 0; j < 5; ++j) {
    REQUIRE(std::abs(dlogphi[j]) == 0.0);
  }

  for (double u : {0.5, 2.0, 10.0, 50.0, 200.0}) {
    nhc::eval_cf_and_dlogphi(u, 1.0, 0.02, p, phi, dlogphi);
    REQUIRE(std::isfinite(phi.real()));
    REQUIRE(std::isfinite(phi.imag()));
    REQUIRE(std::abs(phi) <= 1.0 + 1e-12);
  }

  // Black-Scholes limit: sigma -> 1e-6, v0 = vbar = vol^2
  const double vol = 0.25;
  const nhc::HestonParams bs_p{vol * vol, 2.0, vol * vol, 1e-6, -0.5};
  const nhc::Market m{100.0, 0.03, 0.01};
  const double tau = 0.75;
  const double K = 105.0;
  const double cos_c = nhc::price_option(bs_p, m, tau, K, true, nhc::CosConfig{128, 12.0});

  const double d1 =
      (std::log(m.S0 / K) + (m.r - m.q + 0.5 * vol * vol) * tau) / (vol * std::sqrt(tau));
  const double d2 = d1 - vol * std::sqrt(tau);
  auto ncdf = [](double x) { return 0.5 * std::erfc(-x / std::numbers::sqrt2); };
  const double bs_c = m.S0 * std::exp(-m.q * tau) * ncdf(d1) - K * std::exp(-m.r * tau) * ncdf(d2);

  REQUIRE(std::abs(cos_c - bs_c) / m.S0 < 1e-7);
}

TEST_CASE("Put-call parity holds to machine precision across strikes and regimes", "[cos_pricer]") {
  const nhc::Market m{100.0, 0.04, 0.015};
  const std::vector<nhc::HestonParams> regimes = {
      {0.04, 2.0, 0.04, 0.4, -0.7},  // Feller satisfied
      {0.06, 1.2, 0.05, 1.2, -0.8},  // Feller violated
      {0.25, 0.5, 0.20, 2.0, -0.9},  // Stress
  };
  const std::vector<double> taus = {0.02, 0.25, 1.0, 2.5};
  const std::vector<double> strikes = {70.0, 85.0, 100.0, 115.0, 140.0};

  for (const auto& p : regimes) {
    for (double tau : taus) {
      const double rhs = m.S0 * std::exp(-m.q * tau);
      const double df = std::exp(-m.r * tau);
      for (double K : strikes) {
        const double c = nhc::price_option(p, m, tau, K, true, nhc::CosConfig{256, 14.0});
        const double put = nhc::price_option(p, m, tau, K, false, nhc::CosConfig{256, 14.0});
        REQUIRE(std::abs((c - put) - (rhs - K * df)) / m.S0 < 1e-12);
      }
    }
  }
}

TEST_CASE("COS convergence N=64->128->256->512", "[cos_pricer]") {
  const nhc::HestonParams p{0.05, 1.8, 0.04, 0.6, -0.75};
  const nhc::Market m{100.0, 0.03, 0.01};
  nhc::Chain chain;
  for (double tau : {0.05, 0.5, 2.0}) {
    for (double K : {85.0, 100.0, 115.0}) {
      nhc::Quote q{};
      q.tau = tau;
      q.strike = K;
      q.is_call = (K >= 100.0);
      chain.push_back(q);
    }
  }

  std::vector<double> p_ref(chain.size());
  nhc::price_chain(p, m, chain, nhc::CosConfig{1024, 16.0}, p_ref.data());

  double prev_err = 1.0;
  for (int N : {64, 128, 256, 512}) {
    std::vector<double> p_cur(chain.size());
    nhc::price_chain(p, m, chain, nhc::CosConfig{N, 16.0}, p_cur.data());
    double max_err = 0.0;
    for (std::size_t i = 0; i < chain.size(); ++i) {
      max_err = std::max(max_err, std::abs(p_cur[i] - p_ref[i]) / m.S0);
    }
    if (N >= 128) {
      REQUIRE(max_err <= std::max(prev_err, 1e-13));
    }
    prev_err = max_err;
  }
  REQUIRE(prev_err < 1e-12);
}

TEST_CASE("Analytic d(ln phi)/dTheta matches central finite differences", "[heston_cf]") {
  std::mt19937_64 rng(20261121);
  std::uniform_real_distribution<double> u_dist(0.1, 30.0);
  std::uniform_real_distribution<double> tau_dist(0.02, 3.0);
  std::uniform_real_distribution<double> v0_dist(0.01, 0.30);
  std::uniform_real_distribution<double> kappa_dist(0.2, 8.0);
  std::uniform_real_distribution<double> vbar_dist(0.01, 0.30);
  std::uniform_real_distribution<double> sigma_dist(0.1, 2.5);
  std::uniform_real_distribution<double> rho_dist(-0.95, -0.1);

  double max_rel_err = 0.0;
  for (int i = 0; i < 200; ++i) {
    const double u = u_dist(rng);
    const double tau = tau_dist(rng);
    const double drift = 0.02 * tau;
    const nhc::HestonParams p{v0_dist(rng), kappa_dist(rng), vbar_dist(rng), sigma_dist(rng),
                              rho_dist(rng)};

    std::complex<double> phi;
    std::array<std::complex<double>, 5> dlogphi;
    nhc::eval_cf_and_dlogphi(u, tau, drift, p, phi, dlogphi);

    auto get_field = [](nhc::HestonParams& hp, int idx) -> double& {
      switch (idx) {
        case 0:
          return hp.v0;
        case 1:
          return hp.kappa;
        case 2:
          return hp.vbar;
        case 3:
          return hp.sigma;
        default:
          return hp.rho;
      }
    };

    for (int j = 0; j < 5; ++j) {
      // 4th-order central finite difference to eliminate O(h^2) truncation error
      nhc::HestonParams p1 = p, m1 = p, p2 = p, m2 = p;
      const double eps = 1e-4 * std::max(0.1, std::abs(get_field(p1, j)));
      get_field(p1, j) += eps;
      get_field(m1, j) -= eps;
      get_field(p2, j) += 2.0 * eps;
      get_field(m2, j) -= 2.0 * eps;
      const auto lp_p1 = nhc::eval_log_cf(std::complex<double>(u, 0.0), tau, drift, p1);
      const auto lp_m1 = nhc::eval_log_cf(std::complex<double>(u, 0.0), tau, drift, m1);
      const auto lp_p2 = nhc::eval_log_cf(std::complex<double>(u, 0.0), tau, drift, p2);
      const auto lp_m2 = nhc::eval_log_cf(std::complex<double>(u, 0.0), tau, drift, m2);
      const auto fd = (-lp_p2 + 8.0 * lp_p1 - 8.0 * lp_m1 + lp_m2) / (12.0 * eps);
      const double denom = std::max(1e-8, std::abs(fd));
      max_rel_err = std::max(max_rel_err, std::abs(dlogphi[j] - fd) / denom);
    }
  }
  REQUIRE(max_rel_err < 1e-6);
}

TEST_CASE("Price Jacobian matches central finite differences on multi-maturity chain",
          "[cos_pricer]") {
  const nhc::Market m{100.0, 0.035, 0.012};
  nhc::Chain chain;
  for (double tau : {0.05, 0.25, 0.75, 2.0}) {
    const double F = m.S0 * std::exp((m.r - m.q) * tau);
    for (double lm : {-0.3, -0.15, 0.0, 0.1, 0.25}) {
      nhc::Quote q{};
      q.tau = tau;
      q.strike = F * std::exp(lm * std::sqrt(tau));
      q.is_call = (lm >= 0.0);
      chain.push_back(q);
    }
  }

  struct Case {
    nhc::HestonParams params;
    nhc::CosConfig cfg;
  };
  const std::vector<Case> cases = {
      {{0.04, 2.0, 0.04, 0.4, -0.7}, {512, 20.0, false}},   // Feller satisfied
      {{0.05, 1.5, 0.04, 0.9, -0.8}, {1024, 28.0, false}},  // Strongly Feller violated (7x)
  };

  for (const auto& c : cases) {
    const auto& p = c.params;
    const auto& cfg = c.cfg;
    std::vector<double> prices(chain.size());
    std::vector<double> J(chain.size() * 5);
    nhc::price_chain_with_jacobian(p, m, chain, cfg, prices.data(), J.data());

    std::vector<double> p_only(chain.size());
    nhc::price_chain(p, m, chain, cfg, p_only.data());
    for (std::size_t i = 0; i < chain.size(); ++i) {
      REQUIRE(std::abs(prices[i] - p_only[i]) < 1e-13);
    }

    auto get_field = [](nhc::HestonParams& hp, int idx) -> double& {
      switch (idx) {
        case 0:
          return hp.v0;
        case 1:
          return hp.kappa;
        case 2:
          return hp.vbar;
        case 3:
          return hp.sigma;
        default:
          return hp.rho;
      }
    };

    double max_rel = 0.0;
    for (int j = 0; j < 5; ++j) {
      nhc::HestonParams pp = p;
      nhc::HestonParams pm = p;
      const double h = 5e-5 * std::max(0.1, std::abs(get_field(pp, j)));
      get_field(pp, j) += h;
      get_field(pm, j) -= h;
      std::vector<double> pr_p(chain.size()), pr_m(chain.size());
      nhc::price_chain(pp, m, chain, cfg, pr_p.data());
      nhc::price_chain(pm, m, chain, cfg, pr_m.data());
      for (std::size_t i = 0; i < chain.size(); ++i) {
        const double fd = (pr_p[i] - pr_m[i]) / (2.0 * h);
        const double analytic = J[i * 5 + static_cast<std::size_t>(j)];
        const double denom = std::max(std::abs(fd), 1e-4 * m.S0);
        max_rel = std::max(max_rel, std::abs(analytic - fd) / denom);
      }
    }
    REQUIRE(max_rel < 1e-6);
  }
}
