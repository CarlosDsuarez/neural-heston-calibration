#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "nhc/cos_pricer.hpp"
#include "nhc/grid.hpp"
#include "nhc/iv_solver.hpp"

TEST_CASE("AdaptiveGrid natural cubic spline reproduces exact nodal values and linear functions",
          "[grid]") {
  const nhc::AdaptiveGrid grid;
  REQUIRE(grid.num_taus() == 10);
  REQUIRE(grid.num_k() == 18);
  REQUIRE(grid.size() == 180);

  const auto& u = grid.k_scaled();
  std::vector<double> w(grid.num_k());

  // Exact interpolation at every knot
  for (std::size_t idx = 0; idx < u.size(); ++idx) {
    grid.spline_weights_1d(u[idx], w.data());
    for (std::size_t j = 0; j < u.size(); ++j) {
      const double expected = (j == idx) ? 1.0 : 0.0;
      REQUIRE(std::abs(w[j] - expected) < 1e-13);
    }
  }

  // Exact reproduction of any affine function f(u) = 0.25 - 0.3 * u
  for (double q : {-0.50, -0.23, 0.0, 0.12, 0.31}) {
    grid.spline_weights_1d(q, w.data());
    double val = 0.0;
    double w_sum = 0.0;
    for (std::size_t j = 0; j < u.size(); ++j) {
      val += w[j] * (0.25 - 0.3 * u[j]);
      w_sum += w[j];
    }
    REQUIRE(std::abs(w_sum - 1.0) < 1e-13);
    REQUIRE(std::abs(val - (0.25 - 0.3 * q)) < 1e-13);
  }
}

TEST_CASE(
    "InterpOperator excludes out-of-range quotes, counts coverage, and matches exact Heston "
    "IV Jacobian",
    "[grid]") {
  const nhc::AdaptiveGrid grid;
  const nhc::Market m{5800.0, 0.045, 0.013};
  const nhc::HestonParams p{0.035, 2.2, 0.04, 0.6, -0.75};
  const nhc::CosConfig cfg{128, 12.0};

  // Build canonical grid IVs and IV Jacobian on unit market (S0=1, r=0, q=0)
  const nhc::Chain canon = grid.make_canonical_chain();
  const nhc::Market unit_m{1.0, 0.0, 0.0};
  std::vector<double> canon_px(canon.size()), canon_Jp(canon.size() * 5);
  nhc::price_chain_with_jacobian(p, unit_m, canon, cfg, canon_px.data(), canon_Jp.data());

  std::vector<double> grid_iv(canon.size()), grid_Jiv(canon.size() * 5);
  for (std::size_t i = 0; i < canon.size(); ++i) {
    grid_iv[i] = nhc::implied_volatility(canon_px[i], 1.0, canon[i].strike, 0.0, 0.0, canon[i].tau,
                                         canon[i].is_call);
    const double vega =
        nhc::black_scholes_vega(1.0, canon[i].strike, 0.0, 0.0, canon[i].tau, grid_iv[i]);
    for (int col = 0; col < 5; ++col) {
      grid_Jiv[i * 5 + static_cast<std::size_t>(col)] =
          canon_Jp[i * 5 + static_cast<std::size_t>(col)] / vega;
    }
  }

  // Mix of in-range and out-of-range market quotes on SPX-like market (S0=5800)
  nhc::Chain mkt_chain;
  // Out-of-range maturity (< 0.025)
  mkt_chain.push_back(nhc::Quote{0.01, 5800.0, 0.0, 0.0, 1.0, 1.0, true});
  // In-range interior quotes (off-grid taus and off-grid strikes)
  for (double tau : {0.07, 0.28, 0.62, 1.25}) {
    const double F = m.S0 * std::exp((m.r - m.q) * tau);
    for (double u : {-0.25, -0.08, 0.0, 0.12}) {
      mkt_chain.push_back(
          nhc::Quote{tau, F * std::exp(u * std::sqrt(tau)), 0.0, 0.0, 1.0, 1.0, u >= 0.0});
    }
  }
  // Out-of-range maturity (> 2.0) and out-of-range far wing
  mkt_chain.push_back(nhc::Quote{2.5, 5800.0, 0.0, 0.0, 1.0, 1.0, true});
  mkt_chain.push_back(
      nhc::Quote{0.5, 5800.0 * std::exp(-0.9 * std::sqrt(0.5)), 0.0, 0.0, 1.0, 1.0, false});

  const nhc::InterpOperator op(grid, m, mkt_chain);
  REQUIRE(op.total_quotes() == 19);
  REQUIRE(op.num_in_range() == 16);
  REQUIRE(op.num_excluded() == 3);
  REQUIRE(op.coverage() == Catch::Approx(16.0 / 19.0));

  // Verify interpolation error on smooth Heston surface is tiny (<= 0.02 vol pts == 2e-4)
  std::vector<double> interp_iv(op.num_in_range()), interp_J(op.num_in_range() * 5);
  op.apply_with_jacobian(grid_iv.data(), grid_Jiv.data(), interp_iv.data(), interp_J.data(),
                         nhc::InterpSpace::TotalVariance);

  nhc::Chain in_chain;
  for (std::size_t qi : op.in_range_indices()) {
    in_chain.push_back(mkt_chain[qi]);
  }
  std::vector<double> exact_px(in_chain.size());
  nhc::price_chain(p, m, in_chain, cfg, exact_px.data());

  double max_iv_err = 0.0;
  double sum_sq_err = 0.0;
  for (std::size_t r = 0; r < in_chain.size(); ++r) {
    const double exact_iv = nhc::implied_volatility(exact_px[r], m.S0, in_chain[r].strike, m.r, m.q,
                                                    in_chain[r].tau, in_chain[r].is_call);
    const double err = std::abs(interp_iv[r] - exact_iv);
    max_iv_err = std::max(max_iv_err, err);
    sum_sq_err += err * err;
  }
  const double rmse_iv_err = std::sqrt(sum_sq_err / static_cast<double>(in_chain.size()));
  // Interpolation RMSE < 0.05 vol pts (5e-4) and max < 0.15 vol pts (1.5e-3)
  REQUIRE(rmse_iv_err < 5e-4);
  REQUIRE(max_iv_err < 1.5e-3);

  // Verify apply_with_jacobian matches finite differences of apply wrt grid_iv
  const double eps = 1e-6;
  std::vector<double> iv_p(op.num_in_range()), iv_m(op.num_in_range());
  std::vector<double> g_p = grid_iv, g_m = grid_iv;
  // Perturb along column 0 of grid_Jiv (directional derivative check)
  for (std::size_t i = 0; i < grid_iv.size(); ++i) {
    g_p[i] += eps * grid_Jiv[i * 5 + 0];
    g_m[i] -= eps * grid_Jiv[i * 5 + 0];
  }
  op.apply(g_p.data(), iv_p.data(), nhc::InterpSpace::TotalVariance);
  op.apply(g_m.data(), iv_m.data(), nhc::InterpSpace::TotalVariance);
  for (std::size_t r = 0; r < op.num_in_range(); ++r) {
    const double fd = (iv_p[r] - iv_m[r]) / (2.0 * eps);
    REQUIRE(std::abs(interp_J[r * 5 + 0] - fd) < 1e-7);
  }
}
