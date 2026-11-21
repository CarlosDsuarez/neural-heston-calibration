#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "nhc/cos_pricer.hpp"
#include "nhc/iv_solver.hpp"
#include "nhc/lm_solver.hpp"
#include "nhc/types.hpp"

namespace nhc {

// Exact IV RMSE (in decimal annualized vol, e.g. 0.0005 = 0.05 vol points) between
// exact COS model prices for `theta` and market quotes in `chain`.
[[nodiscard]] inline double evaluate_cos_iv_rmse(const HestonParams& theta, const Market& market,
                                                 const Chain& chain, const CosConfig& cos_cfg,
                                                 CosWorkspace* external_ws = nullptr) {
  if (chain.empty()) {
    return 0.0;
  }
  const std::size_t n = chain.size();
  std::vector<double> model_prices(n);
  price_chain(theta, market, chain, cos_cfg, model_prices.data(), external_ws);

  double sum_sq = 0.0;
  std::size_t valid = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& q = chain[i];
    double mkt_iv = q.iv;
    if (!(mkt_iv > 0.0)) {
      mkt_iv = implied_volatility(q.mid, market.S0, q.strike, market.r, market.q, q.tau, q.is_call);
    }
    const double mod_iv = implied_volatility(model_prices[i], market.S0, q.strike, market.r,
                                             market.q, q.tau, q.is_call);
    if (std::isfinite(mkt_iv) && std::isfinite(mod_iv)) {
      const double diff = mod_iv - mkt_iv;
      sum_sq += diff * diff;
      ++valid;
    }
  }
  if (valid == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::sqrt(sum_sq / static_cast<double>(valid));
}

// Classical calibration baseline (E0 / E1): COS pricer + analytic Jacobian + 5x5 LM.
// Supports both ResidualMode::VegaWeightedPrice (fast, no IV inversion in loop) and
// ResidualMode::ImpliedVol (exact IV residual in loop).
[[nodiscard]] inline LmResult calibrate_cos_lm(const Market& market, const Chain& chain,
                                               HestonParams init, const LmConfig& lm_cfg,
                                               const CosConfig& cos_cfg,
                                               ParamTransform transform = ParamTransform::LogArtanh,
                                               CosWorkspace* external_ws = nullptr) {
  const std::size_t n = chain.size();
  if (n == 0) {
    LmResult empty{};
    empty.theta = init;
    empty.converged = true;
    return empty;
  }

  CosWorkspace local_ws;
  CosWorkspace& ws = external_ws ? *external_ws : local_ws;

  // Precompute market targets, weights, and vegas once outside the LM loop.
  std::vector<double> mkt_price(n);
  std::vector<double> mkt_iv(n);
  std::vector<double> inv_vega_w(n);
  std::vector<double> weights(n);

  for (std::size_t i = 0; i < n; ++i) {
    const auto& q = chain[i];
    const double w = (q.weight > 0.0) ? q.weight : 1.0;
    weights[i] = w;

    double iv = q.iv;
    double px = q.mid;
    if (!(px > 0.0) && iv > 0.0) {
      px = black_scholes_price(market.S0, q.strike, market.r, market.q, q.tau, iv, q.is_call);
    } else if (!(iv > 0.0) && px > 0.0) {
      iv = implied_volatility(px, market.S0, q.strike, market.r, market.q, q.tau, q.is_call);
    }
    mkt_price[i] = px;
    mkt_iv[i] = iv;

    double vega = q.vega;
    if (!(vega > 0.0) && std::isfinite(iv) && iv > 0.0) {
      vega = black_scholes_vega(market.S0, q.strike, market.r, market.q, q.tau, iv);
    }
    const double vega_safe = std::max(vega, lm_cfg.vega_floor * market.S0 * 0.01);
    inv_vega_w[i] = w / vega_safe;
  }

  std::vector<double> model_prices(n);
  std::vector<double> J_price(n * 5);

  auto eval_rj = [&](const HestonParams& theta, Eigen::VectorXd& r, JacMat& J_theta) {
    price_chain_with_jacobian(theta, market, chain, cos_cfg, model_prices.data(), J_price.data(),
                              &ws);
    if (lm_cfg.residual_mode == ResidualMode::VegaWeightedPrice) {
      for (std::size_t i = 0; i < n; ++i) {
        const auto ii = static_cast<Eigen::Index>(i);
        const double sc = inv_vega_w[i];
        r[ii] = sc * (model_prices[i] - mkt_price[i]);
        const double* row = J_price.data() + i * 5;
        for (int j = 0; j < 5; ++j) {
          J_theta(ii, j) = sc * row[j];
        }
      }
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        const auto ii = static_cast<Eigen::Index>(i);
        const auto& q = chain[i];
        const double mod_iv = implied_volatility(model_prices[i], market.S0, q.strike, market.r,
                                                 market.q, q.tau, q.is_call);
        if (std::isfinite(mod_iv) && std::isfinite(mkt_iv[i])) {
          const double vega_mod =
              std::max(black_scholes_vega(market.S0, q.strike, market.r, market.q, q.tau, mod_iv),
                       lm_cfg.vega_floor * market.S0 * 0.01);
          const double sc = weights[i] / vega_mod;
          r[ii] = weights[i] * (mod_iv - mkt_iv[i]);
          const double* row = J_price.data() + i * 5;
          for (int j = 0; j < 5; ++j) {
            J_theta(ii, j) = sc * row[j];
          }
        } else {
          const double sc = inv_vega_w[i];
          r[ii] = sc * (model_prices[i] - mkt_price[i]);
          const double* row = J_price.data() + i * 5;
          for (int j = 0; j < 5; ++j) {
            J_theta(ii, j) = sc * row[j];
          }
        }
      }
    }
  };

  auto eval_r = [&](const HestonParams& theta, Eigen::VectorXd& r) {
    price_chain(theta, market, chain, cos_cfg, model_prices.data(), &ws);
    if (lm_cfg.residual_mode == ResidualMode::VegaWeightedPrice) {
      for (std::size_t i = 0; i < n; ++i) {
        r[static_cast<Eigen::Index>(i)] = inv_vega_w[i] * (model_prices[i] - mkt_price[i]);
      }
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        const auto& q = chain[i];
        const double mod_iv = implied_volatility(model_prices[i], market.S0, q.strike, market.r,
                                                 market.q, q.tau, q.is_call);
        if (std::isfinite(mod_iv) && std::isfinite(mkt_iv[i])) {
          r[static_cast<Eigen::Index>(i)] = weights[i] * (mod_iv - mkt_iv[i]);
        } else {
          r[static_cast<Eigen::Index>(i)] = inv_vega_w[i] * (model_prices[i] - mkt_price[i]);
        }
      }
    }
  };

  LmResult result = lm_solve_5d(n, init, lm_cfg, eval_rj, eval_r, transform);
  // Compute final exact IV RMSE using already-populated mkt_iv (one final COS evaluation at
  // result.theta so rejected trial evaluations cannot overwrite model_prices).
  price_chain(result.theta, market, chain, cos_cfg, model_prices.data(), &ws);
  double sum_sq = 0.0;
  std::size_t valid = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto& q = chain[i];
    const double mod_iv = implied_volatility(model_prices[i], market.S0, q.strike, market.r,
                                             market.q, q.tau, q.is_call);
    if (std::isfinite(mkt_iv[i]) && std::isfinite(mod_iv)) {
      const double diff = mod_iv - mkt_iv[i];
      sum_sq += diff * diff;
      ++valid;
    }
  }
  result.rmse_iv = (valid > 0) ? std::sqrt(sum_sq / static_cast<double>(valid))
                               : std::numeric_limits<double>::quiet_NaN();
  return result;
}

}  // namespace nhc
