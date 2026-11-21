#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "nhc/cos_pricer.hpp"
#include "nhc/grid.hpp"
#include "nhc/iv_solver.hpp"
#include "nhc/lm_solver.hpp"
#include "nhc/mlp_eigen.hpp"
#include "nhc/ood.hpp"
#include "nhc/types.hpp"

namespace nhc {

struct NeuralCalibResult {
  LmResult lm{};
  double surr_rmse_iv = std::numeric_limits<double>::quiet_NaN();
  double cos_rmse_iv = std::numeric_limits<double>::quiet_NaN();
  double coverage = 1.0;
  std::uint32_t ood_flags = OodNone;
  bool fallback_triggered = false;
  double mahal_distance = 0.0;
  double ensemble_var = 0.0;
  double interp_us = 0.0;
  double neural_lm_us = 0.0;
  double cos_audit_us = 0.0;
  double total_us = 0.0;
};

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

// ---------------------------------------------------------------------------
// E2: Grid Surrogate + Fixed Linear Interpolation Operator M + 5x5 LM
// Optionally runs Circuit-Breaker pre/post checks and falls back to exact COS+LM if OOD.
// ---------------------------------------------------------------------------
template <typename Scalar = float>
[[nodiscard]] inline NeuralCalibResult calibrate_surrogate_lm(
    const Market& market, const Chain& chain, const MlpSurrogateT<Scalar>& surrogate,
    HestonParams init, const LmConfig& lm_cfg, const CosConfig& cos_cfg = CosConfig{},
    InterpSpace space = InterpSpace::TotalVariance, bool audit_cos = true,
    const CircuitBreaker* cb = nullptr,
    std::type_identity_t<const MlpInverseEnsembleT<Scalar>*> inv_for_ood = nullptr,
    CosWorkspace* external_ws = nullptr) {
  using Clock = std::chrono::steady_clock;
  const auto t_start = Clock::now();
  NeuralCalibResult out{};

  // Stage 1: Construct fixed linear interpolation operator M (Section 3.9)
  const auto t_interp0 = Clock::now();
  const InterpOperator op(surrogate.grid(), market, chain);
  out.coverage = op.coverage();
  const std::size_t n_in = op.num_in_range();

  std::vector<double> proj_grid;
  double ens_var = -1.0;
  if (cb != nullptr) {
    proj_grid.resize(surrogate.out_dim());
    project_chain_to_grid(surrogate.grid(), market, chain, proj_grid.data());
    if (inv_for_ood != nullptr) {
      const auto pred = inv_for_ood->predict(proj_grid.data());
      ens_var = pred.ensemble_var;
      out.ensemble_var = ens_var;
    }
  }
  const auto t_interp1 = Clock::now();
  out.interp_us = std::chrono::duration<double, std::micro>(t_interp1 - t_interp0).count();

  // Pre-calibration OOD check (Conditions 2 & 5)
  if (cb != nullptr) {
    out.ood_flags = cb->check_pre(chain, out.coverage, proj_grid.data(), proj_grid.size(), ens_var,
                                  &out.mahal_distance);
    if (out.ood_flags != OodNone || n_in < 5) {
      const auto t_fb0 = Clock::now();
      out.fallback_triggered = true;
      out.lm = calibrate_cos_lm(market, chain, init, lm_cfg, cos_cfg, ParamTransform::LogArtanh,
                                external_ws);
      out.cos_rmse_iv = out.lm.rmse_iv;
      const auto t_fb1 = Clock::now();
      out.cos_audit_us = std::chrono::duration<double, std::micro>(t_fb1 - t_fb0).count();
      out.total_us = std::chrono::duration<double, std::micro>(t_fb1 - t_start).count();
      return out;
    }
  }

  if (n_in < 5) {
    out.ood_flags |= OodMarketSupport;
    out.lm = calibrate_cos_lm(market, chain, init, lm_cfg, cos_cfg, ParamTransform::LogArtanh,
                              external_ws);
    out.cos_rmse_iv = out.lm.rmse_iv;
    out.fallback_triggered = true;
    out.total_us = std::chrono::duration<double, std::micro>(Clock::now() - t_start).count();
    return out;
  }

  // Stage 2: Neural LM loop using MlpSurrogate + operator M (zero COS calls in loop)
  const auto t_lm0 = Clock::now();
  std::vector<double> target_iv(n_in);
  std::vector<double> weights(n_in);
  const auto& idx_map = op.in_range_indices();
  for (std::size_t r = 0; r < n_in; ++r) {
    const auto& q = chain[idx_map[r]];
    double iv = q.iv;
    if (!(iv > 0.0) && q.mid > 0.0) {
      iv = implied_volatility(q.mid, market.S0, q.strike, market.r, market.q, q.tau, q.is_call);
    }
    target_iv[r] = std::isfinite(iv) ? iv : 0.20;
    weights[r] = (q.weight > 0.0) ? q.weight : 1.0;
  }

  const std::size_t G = surrogate.out_dim();
  typename MlpSurrogateT<Scalar>::Workspace mlp_ws;
  std::vector<double> grid_iv(G);
  std::vector<double> grid_J(G * 5);
  std::vector<double> mkt_iv_pred(n_in);
  std::vector<double> mkt_J(n_in * 5);

  // Constrain LM cleanly to the surrogate's trained parameter box
  const Vec5& zmin = surrogate.z_min();
  const Vec5& zmax = surrogate.z_max();
  ParamBounds surr_bounds{
      std::exp(zmin[0]),  std::exp(zmax[0]),  std::exp(zmin[1]), std::exp(zmax[1]),
      std::exp(zmin[2]),  std::exp(zmax[2]),  std::exp(zmin[3]), std::exp(zmax[3]),
      std::tanh(zmin[4]), std::tanh(zmax[4]),
  };

  auto eval_rj = [&](const HestonParams& theta, Eigen::VectorXd& r, JacMat& J_theta) {
    surrogate.forward_theta_with_jacobian(theta, grid_iv.data(), grid_J.data(), &mlp_ws);
    op.apply_with_jacobian(grid_iv.data(), grid_J.data(), mkt_iv_pred.data(), mkt_J.data(), space);
    for (std::size_t i = 0; i < n_in; ++i) {
      const auto ii = static_cast<Eigen::Index>(i);
      const double w = weights[i];
      r[ii] = w * (mkt_iv_pred[i] - target_iv[i]);
      const double* row = mkt_J.data() + i * 5;
      for (int j = 0; j < 5; ++j) {
        J_theta(ii, j) = w * row[j];
      }
    }
  };

  auto eval_r = [&](const HestonParams& theta, Eigen::VectorXd& r) {
    surrogate.forward_theta(theta, grid_iv.data(), &mlp_ws);
    op.apply(grid_iv.data(), mkt_iv_pred.data(), space);
    for (std::size_t i = 0; i < n_in; ++i) {
      r[static_cast<Eigen::Index>(i)] = weights[i] * (mkt_iv_pred[i] - target_iv[i]);
    }
  };

  out.lm = lm_solve_5d(n_in, init, lm_cfg, eval_rj, eval_r, ParamTransform::LogArtanh, surr_bounds);

  // Unweighted surrogate IV RMSE on in-range quotes
  surrogate.forward_theta(out.lm.theta, grid_iv.data(), &mlp_ws);
  op.apply(grid_iv.data(), mkt_iv_pred.data(), space);
  double sq = 0.0;
  for (std::size_t i = 0; i < n_in; ++i) {
    const double d = mkt_iv_pred[i] - target_iv[i];
    sq += d * d;
  }
  out.surr_rmse_iv = std::sqrt(sq / static_cast<double>(n_in));
  out.lm.rmse_iv = out.surr_rmse_iv;
  const auto t_lm1 = Clock::now();
  out.neural_lm_us = std::chrono::duration<double, std::micro>(t_lm1 - t_lm0).count();

  // Stage 3: Exact COS audit (always available; required when CircuitBreaker is active)
  if (audit_cos || cb != nullptr) {
    const auto t_aud0 = Clock::now();
    // Evaluate COS RMSE on the in-range chain for apples-to-apples discrepancy check, plus full
    // chain
    out.cos_rmse_iv = evaluate_cos_iv_rmse(out.lm.theta, market, chain, cos_cfg, external_ws);
    out.lm.rmse_iv = out.cos_rmse_iv;

    if (cb != nullptr) {
      Chain in_chain;
      in_chain.reserve(n_in);
      for (std::size_t qi : idx_map) {
        in_chain.push_back(chain[qi]);
      }
      const double cos_in_rmse =
          evaluate_cos_iv_rmse(out.lm.theta, market, in_chain, cos_cfg, external_ws);
      const Vec5 x_final = surrogate.theta_to_x(out.lm.theta);
      out.ood_flags |= cb->check_post(x_final, out.lm, out.surr_rmse_iv, cos_in_rmse);
      if (out.ood_flags != OodNone) {
        out.fallback_triggered = true;
        out.lm = calibrate_cos_lm(market, chain, init, lm_cfg, cos_cfg, ParamTransform::LogArtanh,
                                  external_ws);
        out.cos_rmse_iv = out.lm.rmse_iv;
      }
    }
    const auto t_aud1 = Clock::now();
    out.cos_audit_us = std::chrono::duration<double, std::micro>(t_aud1 - t_aud0).count();
  }

  out.total_us = std::chrono::duration<double, std::micro>(Clock::now() - t_start).count();
  return out;
}

// ---------------------------------------------------------------------------
// E3: Pointwise Random-Grid Surrogate + 5x5 LM (Baschetti et al. 2024 ablation, H5)
// ---------------------------------------------------------------------------
template <typename Scalar = float>
[[nodiscard]] inline NeuralCalibResult calibrate_pointwise_lm(
    const Market& market, const Chain& chain, const MlpPointwiseT<Scalar>& pw, HestonParams init,
    const LmConfig& lm_cfg, const CosConfig& cos_cfg = CosConfig{}, bool audit_cos = true,
    CosWorkspace* external_ws = nullptr) {
  using Clock = std::chrono::steady_clock;
  const auto t_start = Clock::now();
  NeuralCalibResult out{};

  const auto t_prep0 = Clock::now();
  const std::size_t n = chain.size();
  std::vector<double> tk_tilde(n * 2);
  std::vector<double> target_iv(n);
  std::vector<double> weights(n);

  for (std::size_t i = 0; i < n; ++i) {
    const auto& q = chain[i];
    const double F = market.S0 * std::exp((market.r - market.q) * q.tau);
    const double k_sc = std::log(q.strike / F) / std::sqrt(q.tau);
    tk_tilde[i * 2 + 0] = pw.scale_tau(q.tau);
    tk_tilde[i * 2 + 1] = pw.scale_k(k_sc);
    double iv = q.iv;
    if (!(iv > 0.0) && q.mid > 0.0) {
      iv = implied_volatility(q.mid, market.S0, q.strike, market.r, market.q, q.tau, q.is_call);
    }
    target_iv[i] = std::isfinite(iv) ? iv : 0.20;
    weights[i] = (q.weight > 0.0) ? q.weight : 1.0;
  }
  const auto t_prep1 = Clock::now();
  out.interp_us = std::chrono::duration<double, std::micro>(t_prep1 - t_prep0).count();

  const auto t_lm0 = Clock::now();
  std::vector<double> pred_iv(n);
  std::vector<double> pred_J(n * 5);

  const ParamBounds pw_bounds{0.005, 1.0, 0.05, 12.0, 0.01, 0.50, 0.10, 4.50, -0.98, -0.10};

  auto eval_rj = [&](const HestonParams& theta, Eigen::VectorXd& r, JacMat& J_theta) {
    pw.eval_batch_with_jacobian(theta, tk_tilde.data(), n, pred_iv.data(), pred_J.data());
    for (std::size_t i = 0; i < n; ++i) {
      const auto ii = static_cast<Eigen::Index>(i);
      const double w = weights[i];
      r[ii] = w * (pred_iv[i] - target_iv[i]);
      const double* row = pred_J.data() + i * 5;
      for (int j = 0; j < 5; ++j) {
        J_theta(ii, j) = w * row[j];
      }
    }
  };

  auto eval_r = [&](const HestonParams& theta, Eigen::VectorXd& r) {
    pw.eval_batch(theta, tk_tilde.data(), n, pred_iv.data());
    for (std::size_t i = 0; i < n; ++i) {
      r[static_cast<Eigen::Index>(i)] = weights[i] * (pred_iv[i] - target_iv[i]);
    }
  };

  out.lm = lm_solve_5d(n, init, lm_cfg, eval_rj, eval_r, ParamTransform::LogArtanh, pw_bounds);

  pw.eval_batch(out.lm.theta, tk_tilde.data(), n, pred_iv.data());
  double sq = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double d = pred_iv[i] - target_iv[i];
    sq += d * d;
  }
  out.surr_rmse_iv = std::sqrt(sq / static_cast<double>(std::max<std::size_t>(n, 1)));
  out.lm.rmse_iv = out.surr_rmse_iv;
  const auto t_lm1 = Clock::now();
  out.neural_lm_us = std::chrono::duration<double, std::micro>(t_lm1 - t_lm0).count();

  if (audit_cos) {
    const auto t_aud0 = Clock::now();
    out.cos_rmse_iv = evaluate_cos_iv_rmse(out.lm.theta, market, chain, cos_cfg, external_ws);
    out.lm.rmse_iv = out.cos_rmse_iv;
    const auto t_aud1 = Clock::now();
    out.cos_audit_us = std::chrono::duration<double, std::micro>(t_aud1 - t_aud0).count();
  }

  out.total_us = std::chrono::duration<double, std::micro>(Clock::now() - t_start).count();
  return out;
}

// ---------------------------------------------------------------------------
// E4: Direct Inverse Ensemble Only (with exact COS audit)
// ---------------------------------------------------------------------------
template <typename Scalar = float>
[[nodiscard]] inline NeuralCalibResult calibrate_inverse_only(
    const Market& market, const Chain& chain, const MlpInverseEnsembleT<Scalar>& inverse,
    const CosConfig& cos_cfg = CosConfig{}, bool audit_cos = true,
    CosWorkspace* external_ws = nullptr) {
  using Clock = std::chrono::steady_clock;
  const auto t_start = Clock::now();
  NeuralCalibResult out{};

  const auto t_p0 = Clock::now();
  std::vector<double> grid_iv(inverse.in_dim());
  project_chain_to_grid(inverse.grid(), market, chain, grid_iv.data());
  const auto t_p1 = Clock::now();
  out.interp_us = std::chrono::duration<double, std::micro>(t_p1 - t_p0).count();

  const auto t_i0 = Clock::now();
  const auto pred = inverse.predict(grid_iv.data());
  out.lm.theta = pred.theta;
  out.lm.converged = true;
  out.lm.iters = 0;
  out.lm.func_evals = 1;
  out.ensemble_var = pred.ensemble_var;
  const auto t_i1 = Clock::now();
  out.neural_lm_us = std::chrono::duration<double, std::micro>(t_i1 - t_i0).count();

  if (audit_cos) {
    const auto t_a0 = Clock::now();
    out.cos_rmse_iv = evaluate_cos_iv_rmse(out.lm.theta, market, chain, cos_cfg, external_ws);
    out.lm.rmse_iv = out.cos_rmse_iv;
    const auto t_a1 = Clock::now();
    out.cos_audit_us = std::chrono::duration<double, std::micro>(t_a1 - t_a0).count();
  }

  out.total_us = std::chrono::duration<double, std::micro>(Clock::now() - t_start).count();
  return out;
}

// ---------------------------------------------------------------------------
// E5: Hybrid (Inverse Ensemble Warm-Start -> 3..5 iterations exact COS+LM, with OOD fallback)
// ---------------------------------------------------------------------------
template <typename Scalar = float>
[[nodiscard]] inline NeuralCalibResult calibrate_hybrid(
    const Market& market, const Chain& chain, const MlpInverseEnsembleT<Scalar>& inverse,
    const LmConfig& lm_cfg, const CosConfig& cos_cfg = CosConfig{}, int hybrid_iters = 5,
    const CircuitBreaker* cb = nullptr, HestonParams cold_fallback = {0.04, 1.5, 0.04, 0.5, -0.65},
    CosWorkspace* external_ws = nullptr) {
  using Clock = std::chrono::steady_clock;
  const auto t_start = Clock::now();
  NeuralCalibResult out{};

  const auto t_p0 = Clock::now();
  std::vector<double> grid_iv(inverse.in_dim());
  project_chain_to_grid(inverse.grid(), market, chain, grid_iv.data());
  const auto t_p1 = Clock::now();
  out.interp_us = std::chrono::duration<double, std::micro>(t_p1 - t_p0).count();

  const auto t_i0 = Clock::now();
  const auto pred = inverse.predict(grid_iv.data());
  out.ensemble_var = pred.ensemble_var;
  const auto t_i1 = Clock::now();
  out.neural_lm_us = std::chrono::duration<double, std::micro>(t_i1 - t_i0).count();

  const auto t_c0 = Clock::now();
  HestonParams start_theta = pred.theta;
  LmConfig run_cfg = lm_cfg;
  run_cfg.max_iter = hybrid_iters;

  if (cb != nullptr) {
    out.ood_flags = cb->check_pre(chain, 1.0, grid_iv.data(), grid_iv.size(), pred.ensemble_var,
                                  &out.mahal_distance);
    if (cb->check_box_margin(pred.z_tilde_mean)) {
      out.ood_flags |= OodBoxMargin;
    }
    if (out.ood_flags != OodNone) {
      out.fallback_triggered = true;
      start_theta = cold_fallback;
      run_cfg.max_iter = lm_cfg.max_iter;
    }
  }

  out.lm = calibrate_cos_lm(market, chain, start_theta, run_cfg, cos_cfg, ParamTransform::LogArtanh,
                            external_ws);
  out.cos_rmse_iv = out.lm.rmse_iv;
  const auto t_c1 = Clock::now();
  out.cos_audit_us = std::chrono::duration<double, std::micro>(t_c1 - t_c0).count();
  out.total_us = std::chrono::duration<double, std::micro>(t_c1 - t_start).count();
  return out;
}

}  // namespace nhc
