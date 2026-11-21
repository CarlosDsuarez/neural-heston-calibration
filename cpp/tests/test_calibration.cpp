#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "nhc/calibrate.hpp"
#include "nhc/cos_pricer.hpp"
#include "nhc/iv_solver.hpp"
#include "nhc/lm_solver.hpp"

TEST_CASE("IV solver round-trip IV -> price -> IV <= 1e-10 across strikes/vols/maturities",
          "[iv_solver]") {
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> vol_dist(0.02, 2.5);
  std::uniform_real_distribution<double> tau_dist(0.01, 3.0);
  std::uniform_real_distribution<double> lm_dist(-0.6, 0.5);
  std::uniform_real_distribution<double> r_dist(0.0, 0.06);
  std::uniform_real_distribution<double> q_dist(0.0, 0.03);

  double max_err = 0.0;
  int tested = 0;

  for (int i = 0; i < 2000; ++i) {
    const double S0 = 100.0;
    const double r = r_dist(rng);
    const double q = q_dist(rng);
    const double tau = tau_dist(rng);
    const double vol = vol_dist(rng);
    const double F = S0 * std::exp((r - q) * tau);
    const double K = F * std::exp(lm_dist(rng) * std::sqrt(tau));
    const bool is_call = (i % 2 == 0);

    const double px = nhc::black_scholes_price(S0, K, r, q, tau, vol, is_call);
    const double vega = nhc::black_scholes_vega(S0, K, r, q, tau, vol);
    const bool is_otm = (is_call && K >= F) || (!is_call && K <= F);
    // For ITM options, IEEE-754 price storage limits achievable IV precision to
    // ~eps_mach * price / vega; require well-conditioned ITM vega >= 1e-3 * S0.
    // For OTM options, erfcx log-space inversion works down to px >= 1e-100 * S0.
    if ((!is_otm && vega < 1e-3 * S0) || (is_otm && px < 1e-100 * S0)) {
      continue;
    }

    const double iv = nhc::implied_volatility(px, S0, K, r, q, tau, is_call);
    REQUIRE(std::isfinite(iv));
    max_err = std::max(max_err, std::abs(iv - vol));
    ++tested;
  }

  REQUIRE(tested > 1500);
  REQUIRE(max_err <= 1e-10);
}

TEST_CASE("IV solver detects arbitrage violations cleanly", "[iv_solver]") {
  const double iv_neg = nhc::implied_volatility(-1.0, 100.0, 100.0, 0.02, 0.0, 1.0, true);
  REQUIRE(std::isnan(iv_neg));
  const double iv_below_intr = nhc::implied_volatility(5.0, 100.0, 80.0, 0.0, 0.0, 1.0, true);
  REQUIRE(std::isnan(iv_below_intr));
  const double iv_above_spot = nhc::implied_volatility(105.0, 100.0, 100.0, 0.0, 0.0, 1.0, true);
  REQUIRE(std::isnan(iv_above_spot));
}

TEST_CASE("LM recovers noiseless Heston surface in both residual modes and transforms",
          "[calibrate]") {
  const nhc::Market m{100.0, 0.03, 0.01};
  const nhc::HestonParams truth{0.055, 2.4, 0.042, 0.65, -0.78};
  const nhc::CosConfig cos_cfg{128, 12.0, false};

  nhc::Chain chain;
  for (double tau : {0.05, 0.15, 0.35, 0.75, 1.5, 2.0}) {
    const double F = m.S0 * std::exp((m.r - m.q) * tau);
    for (int k = 0; k < 15; ++k) {
      const double lm = -0.45 + (0.30 - (-0.45)) * static_cast<double>(k) / 14.0;
      const double K = F * std::exp(lm * std::sqrt(tau));
      nhc::Quote q{};
      q.tau = tau;
      q.strike = K;
      q.is_call = (lm >= 0.0);
      q.weight = 1.0;
      chain.push_back(q);
    }
  }

  std::vector<double> true_prices(chain.size());
  nhc::price_chain(truth, m, chain, cos_cfg, true_prices.data());
  for (std::size_t i = 0; i < chain.size(); ++i) {
    chain[i].mid = true_prices[i];
    chain[i].iv = nhc::implied_volatility(true_prices[i], m.S0, chain[i].strike, m.r, m.q,
                                          chain[i].tau, chain[i].is_call);
    chain[i].vega =
        nhc::black_scholes_vega(m.S0, chain[i].strike, m.r, m.q, chain[i].tau, chain[i].iv);
  }

  const nhc::HestonParams cold_init{0.04, 1.5, 0.06, 0.4, -0.5};

  for (auto mode : {nhc::ResidualMode::VegaWeightedPrice, nhc::ResidualMode::ImpliedVol}) {
    for (auto tr : {nhc::ParamTransform::LogArtanh, nhc::ParamTransform::SqrtDrift}) {
      nhc::LmConfig lm_cfg;
      lm_cfg.residual_mode = mode;
      const auto res = nhc::calibrate_cos_lm(m, chain, cold_init, lm_cfg, cos_cfg, tr);
      REQUIRE(res.converged);
      // 0.005 vol points == 5e-5 decimal IV
      REQUIRE(res.rmse_iv <= 5e-5);
    }
  }
}
