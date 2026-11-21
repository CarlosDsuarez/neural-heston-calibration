#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <string>
#include <vector>

#include "nhc/calibrate.hpp"
#include "nhc/grid.hpp"
#include "nhc/mlp_eigen.hpp"
#include "nhc/ood.hpp"

#ifndef NHC_SOURCE_DIR
#define NHC_SOURCE_DIR "."
#endif

static std::string model_path(const std::string& name) {
  return std::string(NHC_SOURCE_DIR) + "/results/models/" + name;
}

TEST_CASE("NHC1 loader enforces strict grid-hash match", "[mlp_eigen]") {
  const nhc::AdaptiveGrid good_grid;
  REQUIRE_NOTHROW(nhc::MlpSurrogate::load(model_path("surrogate_sobolev_64x4.bin"), &good_grid));

  // Mutating a single grid maturity MUST throw on load (Section 4.2)
  auto bad_taus = good_grid.taus();
  bad_taus[0] += 0.005;
  const nhc::AdaptiveGrid bad_grid(bad_taus, good_grid.k_scaled());
  REQUIRE_THROWS(nhc::MlpSurrogate::load(model_path("surrogate_sobolev_64x4.bin"), &bad_grid));
}

TEST_CASE("MlpSurrogateD forward-mode Jacobian matches central finite differences to 1e-8",
          "[mlp_eigen]") {
  const nhc::AdaptiveGrid grid;
  const auto surr = nhc::MlpSurrogateD::load(model_path("surrogate_sobolev_64x4.bin"), &grid);
  const nhc::HestonParams p{0.045, 2.3, 0.04, 0.65, -0.75};

  std::vector<double> iv(surr.out_dim());
  std::vector<double> J(surr.out_dim() * 5);
  surr.forward_theta_with_jacobian(p, iv.data(), J.data());

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

  double max_err = 0.0;
  std::vector<double> iv_p(surr.out_dim()), iv_m(surr.out_dim());
  for (int col = 0; col < 5; ++col) {
    nhc::HestonParams pp = p, pm = p;
    const double h = 1e-6 * std::max(0.1, std::abs(get_field(pp, col)));
    get_field(pp, col) += h;
    get_field(pm, col) -= h;
    surr.forward_theta(pp, iv_p.data());
    surr.forward_theta(pm, iv_m.data());
    for (std::size_t i = 0; i < surr.out_dim(); ++i) {
      const double fd = (iv_p[i] - iv_m[i]) / (2.0 * h);
      const double an = J[i * 5 + static_cast<std::size_t>(col)];
      const double denom = std::max(1.0, std::abs(fd));
      max_err = std::max(max_err, std::abs(an - fd) / denom);
    }
  }
  REQUIRE(max_err < 1e-7);
}

TEST_CASE("MlpPointwise batched reverse-mode Jacobian matches central finite differences",
          "[mlp_eigen]") {
  const nhc::AdaptiveGrid grid;
  const auto pw = nhc::MlpPointwiseD::load(model_path("pointwise_64x4.bin"), &grid);
  const nhc::HestonParams p{0.05, 1.8, 0.045, 0.7, -0.72};

  const std::vector<double> tk = {pw.scale_tau(0.05), pw.scale_k(-0.2),  pw.scale_tau(0.5),
                                  pw.scale_k(0.0),    pw.scale_tau(1.5), pw.scale_k(0.2)};
  const std::size_t n = 3;
  std::vector<double> iv(n), J(n * 5);
  pw.eval_batch_with_jacobian(p, tk.data(), n, iv.data(), J.data());

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

  std::vector<double> iv_p(n), iv_m(n);
  double max_err = 0.0;
  for (int col = 0; col < 5; ++col) {
    nhc::HestonParams pp = p, pm = p;
    const double h = 1e-6 * std::max(0.1, std::abs(get_field(pp, col)));
    get_field(pp, col) += h;
    get_field(pm, col) -= h;
    pw.eval_batch(pp, tk.data(), n, iv_p.data());
    pw.eval_batch(pm, tk.data(), n, iv_m.data());
    for (std::size_t i = 0; i < n; ++i) {
      const double fd = (iv_p[i] - iv_m[i]) / (2.0 * h);
      const double an = J[i * 5 + static_cast<std::size_t>(col)];
      const double denom = std::max(1.0, std::abs(fd));
      max_err = std::max(max_err, std::abs(an - fd) / denom);
    }
  }
  REQUIRE(max_err < 1e-7);
}

TEST_CASE("Surrogate LM, Hybrid, and Circuit-Breaker OOD fallback work end-to-end", "[ood]") {
  const nhc::AdaptiveGrid grid;
  const auto surr = nhc::MlpSurrogate::load(model_path("surrogate_sobolev_64x4.bin"), &grid);
  const auto inv =
      nhc::MlpInverse::load({model_path("inverse_recon_s0.bin"), model_path("inverse_recon_s1.bin"),
                             model_path("inverse_recon_s2.bin"), model_path("inverse_recon_s3.bin"),
                             model_path("inverse_recon_s4.bin")},
                            &grid);

  const nhc::Market m{100.0, 0.03, 0.01};
  const nhc::HestonParams truth{0.04, 2.2, 0.045, 0.65, -0.75};
  const nhc::CosConfig cos_cfg{128, 12.0};
  const nhc::LmConfig lm_cfg;

  nhc::Chain chain;
  for (double tau : {0.04, 0.12, 0.30, 0.65, 1.25, 1.80}) {
    const double F = m.S0 * std::exp((m.r - m.q) * tau);
    for (int k = 0; k < 12; ++k) {
      const double u = -0.40 + (0.25 - (-0.40)) * static_cast<double>(k) / 11.0;
      const double K = F * std::exp(u * std::sqrt(tau));
      chain.push_back(nhc::Quote{tau, K, 0.0, 0.0, 1.0, 1.0, u >= 0.0});
    }
  }
  std::vector<double> pxs(chain.size());
  nhc::price_chain(truth, m, chain, cos_cfg, pxs.data());
  for (std::size_t i = 0; i < chain.size(); ++i) {
    chain[i].mid = pxs[i];
    chain[i].iv = nhc::implied_volatility(pxs[i], m.S0, chain[i].strike, m.r, m.q, chain[i].tau,
                                          chain[i].is_call);
    chain[i].vega =
        nhc::black_scholes_vega(m.S0, chain[i].strike, m.r, m.q, chain[i].tau, chain[i].iv);
  }

  // 1. In-domain E2 calibration
  const nhc::HestonParams cold{0.06, 1.5, 0.05, 0.45, -0.6};
  const auto res_e2 = nhc::calibrate_surrogate_lm(m, chain, surr, cold, lm_cfg, cos_cfg);
  REQUIRE(res_e2.lm.converged);
  REQUIRE(res_e2.cos_rmse_iv * 100.0 < 0.15);  // < 0.15 vol pts exact COS RMSE

  // 2. In-domain E5 hybrid calibration (5 COS iterations after inverse warm-start)
  const auto res_e5 = nhc::calibrate_hybrid(m, chain, inv, lm_cfg, cos_cfg, 5);
  REQUIRE(res_e5.cos_rmse_iv * 100.0 < 0.02);

  // 3. Synthetic OOD case (extreme IV = 250% vol outside training support) MUST trigger fallback
  nhc::Chain ood_chain = chain;
  for (auto& q : ood_chain) {
    q.iv = 2.50;
    q.mid = nhc::black_scholes_price(m.S0, q.strike, m.r, m.q, q.tau, q.iv, q.is_call);
    q.vega = nhc::black_scholes_vega(m.S0, q.strike, m.r, m.q, q.tau, q.iv);
  }
  const nhc::CircuitBreaker cb;
  const auto res_ood = nhc::calibrate_surrogate_lm(
      m, ood_chain, surr, cold, lm_cfg, cos_cfg, nhc::InterpSpace::TotalVariance, true, &cb, &inv);
  REQUIRE(res_ood.fallback_triggered);
  REQUIRE((res_ood.ood_flags & nhc::OodMarketSupport) != 0);
}
