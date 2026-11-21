#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "nhc/types.hpp"

namespace nhc {

enum class ParamTransform : std::uint8_t {
  LogArtanh = 0,  // z = (ln v0, ln kappa, ln vbar, ln sigma, artanh rho)
  SqrtDrift = 1,  // z = (ln sqrt(v0), ln(kappa*vbar), ln sqrt(vbar), ln sigma, artanh rho)
};

struct ParamBounds {
  double v0_min = 1e-4, v0_max = 4.0;
  double kappa_min = 1e-3, kappa_max = 30.0;
  double vbar_min = 1e-4, vbar_max = 4.0;
  double sigma_min = 1e-3, sigma_max = 10.0;
  double rho_min = -0.999, rho_max = 0.999;
};

using Vec5 = Eigen::Matrix<double, 5, 1>;
using Mat5 = Eigen::Matrix<double, 5, 5>;
using JacMat = Eigen::Matrix<double, Eigen::Dynamic, 5, Eigen::RowMajor>;

[[nodiscard]] inline HestonParams clamp_params(const HestonParams& p,
                                               const ParamBounds& b = ParamBounds{}) noexcept {
  return HestonParams{
      std::clamp(p.v0, b.v0_min, b.v0_max),       std::clamp(p.kappa, b.kappa_min, b.kappa_max),
      std::clamp(p.vbar, b.vbar_min, b.vbar_max), std::clamp(p.sigma, b.sigma_min, b.sigma_max),
      std::clamp(p.rho, b.rho_min, b.rho_max),
  };
}

[[nodiscard]] inline bool is_at_bounds(const HestonParams& p,
                                       const ParamBounds& b = ParamBounds{}) noexcept {
  constexpr double rel = 1e-3;
  return (p.v0 <= b.v0_min * (1.0 + rel)) || (p.v0 >= b.v0_max * (1.0 - rel)) ||
         (p.kappa <= b.kappa_min * (1.0 + rel)) || (p.kappa >= b.kappa_max * (1.0 - rel)) ||
         (p.vbar <= b.vbar_min * (1.0 + rel)) || (p.vbar >= b.vbar_max * (1.0 - rel)) ||
         (p.sigma <= b.sigma_min * (1.0 + rel)) || (p.sigma >= b.sigma_max * (1.0 - rel)) ||
         (p.rho <= b.rho_min + 5e-4) || (p.rho >= b.rho_max - 5e-4);
}

[[nodiscard]] inline Vec5 params_to_z(const HestonParams& p_in,
                                      ParamTransform tr = ParamTransform::LogArtanh,
                                      const ParamBounds& b = ParamBounds{}) noexcept {
  const HestonParams p = clamp_params(p_in, b);
  Vec5 z;
  if (tr == ParamTransform::LogArtanh) {
    z[0] = std::log(p.v0);
    z[1] = std::log(p.kappa);
    z[2] = std::log(p.vbar);
    z[3] = std::log(p.sigma);
    z[4] = std::atanh(p.rho);
  } else {
    z[0] = 0.5 * std::log(p.v0);
    z[1] = std::log(p.kappa * p.vbar);
    z[2] = 0.5 * std::log(p.vbar);
    z[3] = std::log(p.sigma);
    z[4] = std::atanh(p.rho);
  }
  return z;
}

[[nodiscard]] inline HestonParams z_to_params(const Vec5& z,
                                              ParamTransform tr = ParamTransform::LogArtanh,
                                              const ParamBounds& b = ParamBounds{}) noexcept {
  HestonParams p;
  if (tr == ParamTransform::LogArtanh) {
    p.v0 = std::exp(std::clamp(z[0], std::log(b.v0_min), std::log(b.v0_max)));
    p.kappa = std::exp(std::clamp(z[1], std::log(b.kappa_min), std::log(b.kappa_max)));
    p.vbar = std::exp(std::clamp(z[2], std::log(b.vbar_min), std::log(b.vbar_max)));
    p.sigma = std::exp(std::clamp(z[3], std::log(b.sigma_min), std::log(b.sigma_max)));
    p.rho = std::clamp(std::tanh(z[4]), b.rho_min, b.rho_max);
  } else {
    const double v0 = std::exp(std::clamp(2.0 * z[0], std::log(b.v0_min), std::log(b.v0_max)));
    const double vbar =
        std::exp(std::clamp(2.0 * z[2], std::log(b.vbar_min), std::log(b.vbar_max)));
    const double kappa =
        std::clamp(std::exp(std::clamp(z[1], -20.0, 20.0)) / vbar, b.kappa_min, b.kappa_max);
    const double sigma = std::exp(std::clamp(z[3], std::log(b.sigma_min), std::log(b.sigma_max)));
    const double rho = std::clamp(std::tanh(z[4]), b.rho_min, b.rho_max);
    p = HestonParams{v0, kappa, vbar, sigma, rho};
  }
  return p;
}

// 5x5 Jacobian dTheta / dz for chain rule J_z = J_theta * (dTheta / dz).
[[nodiscard]] inline Mat5 dtheta_dz(const HestonParams& p,
                                    ParamTransform tr = ParamTransform::LogArtanh) noexcept {
  Mat5 D = Mat5::Zero();
  if (tr == ParamTransform::LogArtanh) {
    D(0, 0) = p.v0;
    D(1, 1) = p.kappa;
    D(2, 2) = p.vbar;
    D(3, 3) = p.sigma;
    D(4, 4) = 1.0 - p.rho * p.rho;
  } else {
    D(0, 0) = 2.0 * p.v0;
    D(1, 1) = p.kappa;
    D(1, 2) = -2.0 * p.kappa;
    D(2, 2) = 2.0 * p.vbar;
    D(3, 3) = p.sigma;
    D(4, 4) = 1.0 - p.rho * p.rho;
  }
  return D;
}

// Generic 5-parameter Levenberg-Marquardt solver in unconstrained z-space with
// scale-invariant Marquardt diagonal + Nielsen smooth damping update.
//
// `eval_residual_and_jac(theta, r, J_theta)` must populate r (size m) and J_theta (m x 5).
// `eval_residual_only(theta, r)` must populate r (size m) for trial-step acceptance.
template <typename EvalRJ, typename EvalR>
LmResult lm_solve_5d(std::size_t m, HestonParams init, const LmConfig& cfg,
                     EvalRJ&& eval_residual_and_jac, EvalR&& eval_residual_only,
                     ParamTransform tr = ParamTransform::LogArtanh,
                     const ParamBounds& bounds = ParamBounds{}) {
  LmResult res{};
  if (m == 0) {
    res.theta = clamp_params(init, bounds);
    res.converged = true;
    return res;
  }

  Vec5 z = params_to_z(init, tr, bounds);
  HestonParams theta = z_to_params(z, tr, bounds);

  Eigen::VectorXd r(static_cast<Eigen::Index>(m));
  Eigen::VectorXd r_try(static_cast<Eigen::Index>(m));
  JacMat J_theta(static_cast<Eigen::Index>(m), 5);
  JacMat J_try(static_cast<Eigen::Index>(m), 5);
  JacMat J_z(static_cast<Eigen::Index>(m), 5);

  eval_residual_and_jac(theta, r, J_theta);
  res.func_evals = 1;

  double cost = 0.5 * r.squaredNorm();
  if (!std::isfinite(cost)) {
    res.theta = theta;
    res.converged = false;
    return res;
  }

  Mat5 D_th = dtheta_dz(theta, tr);
  J_z.noalias() = J_theta * D_th;
  Mat5 H = J_z.transpose() * J_z;
  Vec5 g = J_z.transpose() * r;

  double lambda = std::max(cfg.lambda0, 1e-12);
  double nu = 2.0;

  for (int iter = 0; iter < cfg.max_iter; ++iter) {
    if (g.lpNorm<Eigen::Infinity>() <= cfg.tol_grad ||
        std::sqrt(2.0 * cost / static_cast<double>(m)) <= 1e-11) {
      res.converged = true;
      break;
    }

    // Scale-invariant Marquardt diagonal
    Vec5 diag_scale;
    for (int k = 0; k < 5; ++k) {
      diag_scale[k] = std::clamp(H(k, k), 1e-6, 1e12);
    }

    bool accepted = false;
    Vec5 dz = Vec5::Zero();

    // Inner damping loop (at most 12 rejections per outer iteration)
    for (int sub = 0; sub < 12; ++sub) {
      Mat5 A = H;
      for (int k = 0; k < 5; ++k) {
        A(k, k) += lambda * diag_scale[k] + 1e-12;
      }

      Eigen::LDLT<Mat5> ldlt(A);
      if (ldlt.info() != Eigen::Success) {
        lambda *= nu;
        nu *= 2.0;
        continue;
      }
      dz = ldlt.solve(-g);
      if (!dz.allFinite()) {
        lambda *= nu;
        nu *= 2.0;
        continue;
      }

      // Limit single-step jump in z-space to keep nonlinear steps well-behaved
      const double max_dz = dz.lpNorm<Eigen::Infinity>();
      if (max_dz > 2.5) {
        dz *= (2.5 / max_dz);
      }

      const Vec5 z_try_raw = z + dz;
      const HestonParams theta_try = z_to_params(z_try_raw, tr, bounds);
      const Vec5 z_try = params_to_z(theta_try, tr, bounds);
      const Vec5 dz_eff = z_try - z;

      if (dz_eff.norm() <= cfg.tol_step * (z.norm() + cfg.tol_step)) {
        res.converged = true;
        accepted = true;
        break;
      }

      // On the first trial step (sub == 0), fuse residual + Jacobian evaluation because analytic
      // Jacobian adds only ~30% over pricing and steps are accepted ~90% of the time.
      const bool fused_jac = (sub == 0);
      if (fused_jac) {
        eval_residual_and_jac(theta_try, r_try, J_try);
      } else {
        eval_residual_only(theta_try, r_try);
      }
      ++res.func_evals;

      const double cost_try = 0.5 * r_try.squaredNorm();
      // Predicted decrease under quadratic model: -g^T dz - 0.5 dz^T H dz
      const double pred = -g.dot(dz_eff) - 0.5 * dz_eff.dot(H * dz_eff);

      if (std::isfinite(cost_try) && cost_try < cost && pred > 0.0) {
        const double rho_gain = (cost - cost_try) / pred;
        z = z_try;
        theta = theta_try;
        r.swap(r_try);
        cost = cost_try;

        // Nielsen smooth update
        const double factor = 1.0 - std::pow(2.0 * rho_gain - 1.0, 3);
        lambda = std::clamp(lambda * std::max(1.0 / 3.0, factor), 1e-12, 1e12);
        nu = 2.0;
        accepted = true;
        ++res.iters;

        if (dz_eff.norm() <= cfg.tol_step * (z.norm() + cfg.tol_step)) {
          res.converged = true;
        } else {
          if (fused_jac) {
            J_theta.swap(J_try);
          } else {
            eval_residual_and_jac(theta, r, J_theta);
            ++res.func_evals;
          }
          D_th = dtheta_dz(theta, tr);
          J_z.noalias() = J_theta * D_th;
          H.noalias() = J_z.transpose() * J_z;
          g.noalias() = J_z.transpose() * r;
        }
        break;
      }

      lambda *= nu;
      nu *= 2.0;
      if (lambda > 1e12) {
        break;
      }
    }

    if (res.converged) {
      break;
    }
    if (!accepted) {
      // Stalled: mark converged only if residual norm is already small
      res.converged = (std::sqrt(2.0 * cost / static_cast<double>(m)) < 1e-5);
      break;
    }
  }

  res.theta = theta;
  res.rmse_weighted = std::sqrt(2.0 * cost / static_cast<double>(m));
  res.hit_bounds = is_at_bounds(theta, bounds);
  return res;
}

}  // namespace nhc
